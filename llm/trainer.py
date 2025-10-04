"""Training loop with distributed training, AMP, and checkpointing."""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from .config import Config, TrainingConfig
from .model import GPTModel
from .tokenizer import SPTokenizer
from .data import DataModule
from .evaluation import evaluate_model


def setup_distributed() -> bool:
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        torch.cuda.set_device(local_rank)
        return True
    
    return False


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_lr_scheduler(optimizer, scheduler_type: str, num_training_steps: int, warmup_steps: int):
    """Get learning rate scheduler."""
    if scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == "constant":
        from transformers import get_constant_schedule_with_warmup
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class Trainer:
    """Main training class with distributed support."""
    
    def __init__(
        self,
        config: Config,
        model: GPTModel,
        tokenizer: SPTokenizer,
        data_module: DataModule,
        output_dir: str = "artifacts/checkpoints"
    ):
        self.config = config
        self.training_config = config.training
        self.model = model
        self.tokenizer = tokenizer
        self.data_module = data_module
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup distributed training
        self.is_distributed = setup_distributed()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
        
        self.model = self.model.to(self.device)
        
        # Wrap model for distributed training
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        
        # Setup logging
        self._setup_logging()
        
        # Initialize training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Setup mixed precision
        self.scaler = GradScaler() if self.training_config.use_amp else None
        
        # Load checkpoint if specified
        if self.training_config.resume_from_checkpoint:
            self.load_checkpoint(self.training_config.resume_from_checkpoint)
    
    def _setup_logging(self):
        """Setup logging."""
        log_level = getattr(logging, self.training_config.log_level.upper())
        
        # Only log on main process in distributed training
        if self.rank == 0:
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(self.output_dir / 'training.log')
                ]
            )
        else:
            logging.basicConfig(level=logging.WARNING)
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Separate weight decay for different parameter types
        no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.training_config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        if self.training_config.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.training_config.learning_rate,
                betas=(self.training_config.beta1, self.training_config.beta2),
                eps=self.training_config.eps,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.training_config.optimizer}")
        
        # Calculate total training steps
        train_dataset = self.data_module.get_train_dataset()
        if hasattr(train_dataset, '__len__'):
            total_steps = len(train_dataset) * self.training_config.num_epochs // (
                self.training_config.batch_size * self.training_config.gradient_accumulation_steps * self.world_size
            )
        else:
            total_steps = self.training_config.max_steps or 100000
        
        # Setup scheduler
        if self.training_config.scheduler != "none":
            try:
                self.scheduler = get_lr_scheduler(
                    self.optimizer,
                    self.training_config.scheduler,
                    total_steps,
                    self.training_config.warmup_steps
                )
            except ImportError:
                self.logger.warning("transformers library not available, using manual scheduler")
                self.scheduler = None
        else:
            self.scheduler = None
        
        self.logger.info(f"Total training steps: {total_steps}")
        self.logger.info(f"Warmup steps: {self.training_config.warmup_steps}")
    
    def save_checkpoint(self, checkpoint_dir: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Only save on main process
        if self.rank != 0:
            return
        
        # Get model state dict (unwrap DDP if necessary)
        model_state_dict = (
            self.model.module.state_dict() 
            if hasattr(self.model, 'module') 
            else self.model.state_dict()
        )
        
        checkpoint = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.config.to_dict(),
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_dir / 'pytorch_model.bin')
        
        # Save config
        self.config.to_yaml(checkpoint_dir / 'config.yaml')
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        if is_best:
            # Also save as best checkpoint
            best_dir = self.output_dir / 'best'
            best_dir.mkdir(exist_ok=True)
            torch.save(checkpoint, best_dir / 'pytorch_model.bin')
            self.config.to_yaml(best_dir / 'config.yaml')
            self.tokenizer.save_pretrained(str(best_dir))
            self.logger.info(f"Best checkpoint saved to {best_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / 'pytorch_model.bin'
        else:
            checkpoint_file = checkpoint_path
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
        
        self.logger.info(f"Loading checkpoint from {checkpoint_file}")
        
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        # Load model state dict
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state dict
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state dict
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state dict
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        self.logger.info(f"Checkpoint loaded. Resuming from step {self.global_step}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Model parameters: {self.model.get_num_params():,}")
        
        # Get data loaders
        train_dataloader = self.data_module.get_train_dataloader(
            batch_size=self.training_config.batch_size,
            num_workers=self.training_config.dataloader_num_workers,
            pin_memory=self.training_config.dataloader_pin_memory,
            distributed=self.is_distributed
        )
        
        eval_dataloader = None
        if self.training_config.eval_dataset_paths:
            eval_dataloader = self.data_module.get_eval_dataloader(
                batch_size=self.training_config.batch_size,
                num_workers=self.training_config.dataloader_num_workers,
                pin_memory=self.training_config.dataloader_pin_memory,
                distributed=self.is_distributed
            )
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        log_interval = self.training_config.logging_steps
        
        for epoch in range(self.epoch, self.training_config.num_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if self.is_distributed and hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
            
            epoch_start_time = time.time()
            
            for step, batch in enumerate(train_dataloader):
                if self.training_config.max_steps and self.global_step >= self.training_config.max_steps:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                if self.training_config.use_amp:
                    with autocast():
                        outputs = self.model(**batch)
                        loss = outputs['loss'] / self.training_config.gradient_accumulation_steps
                else:
                    outputs = self.model(**batch)
                    loss = outputs['loss'] / self.training_config.gradient_accumulation_steps
                
                # Backward pass
                if self.training_config.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
                
                # Gradient accumulation
                if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.training_config.max_grad_norm > 0:
                        if self.training_config.use_amp:
                            self.scaler.unscale_(self.optimizer)
                        
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.training_config.max_grad_norm
                        )
                    
                    # Optimizer step
                    if self.training_config.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Scheduler step
                    if self.scheduler:
                        self.scheduler.step()
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % log_interval == 0:
                        avg_loss = total_loss / log_interval
                        lr = self.optimizer.param_groups[0]['lr']
                        
                        if self.rank == 0:
                            self.logger.info(
                                f"Epoch {epoch} | Step {self.global_step} | "
                                f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                            )
                        
                        total_loss = 0.0
                    
                    # Evaluation
                    if (self.training_config.eval_every_n_steps > 0 and 
                        self.global_step % self.training_config.eval_every_n_steps == 0 and 
                        eval_dataloader is not None):
                        
                        eval_loss = self.evaluate(eval_dataloader)
                        
                        if self.rank == 0:
                            self.logger.info(f"Eval loss: {eval_loss:.4f}")
                            
                            # Save best checkpoint
                            if eval_loss < self.best_loss:
                                self.best_loss = eval_loss
                                checkpoint_dir = self.output_dir / f'step-{self.global_step}'
                                self.save_checkpoint(checkpoint_dir, is_best=True)
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if (self.training_config.save_every_n_steps > 0 and 
                        self.global_step % self.training_config.save_every_n_steps == 0):
                        
                        checkpoint_dir = self.output_dir / f'step-{self.global_step}'
                        self.save_checkpoint(checkpoint_dir)
                        
                        # Clean up old checkpoints
                        self._cleanup_checkpoints()
            
            epoch_time = time.time() - epoch_start_time
            if self.rank == 0:
                self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        # Final checkpoint
        final_checkpoint_dir = self.output_dir / 'final'
        self.save_checkpoint(final_checkpoint_dir)
        
        if self.rank == 0:
            self.logger.info("Training completed!")
        
        cleanup_distributed()
    
    def evaluate(self, eval_dataloader: DataLoader) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.training_config.use_amp:
                    with autocast():
                        outputs = self.model(**batch)
                        loss = outputs['loss']
                else:
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _cleanup_checkpoints(self):
        """Clean up old checkpoints to save disk space."""
        if self.training_config.save_total_limit <= 0:
            return
        
        checkpoint_dirs = []
        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith('step-'):
                try:
                    step_num = int(item.name.split('-')[1])
                    checkpoint_dirs.append((step_num, item))
                except (ValueError, IndexError):
                    continue
        
        # Sort by step number and keep only the most recent ones
        checkpoint_dirs.sort(key=lambda x: x[0])
        
        while len(checkpoint_dirs) > self.training_config.save_total_limit:
            _, old_dir = checkpoint_dirs.pop(0)
            if old_dir.exists():
                import shutil
                shutil.rmtree(old_dir)
                if self.rank == 0:
                    self.logger.info(f"Removed old checkpoint: {old_dir}")


def train_model(config: Config) -> None:
    """Train model from config."""
    # Load tokenizer
    tokenizer = SPTokenizer.from_pretrained(config.training.tokenizer_path)
    
    # Update model config with tokenizer info
    config.model.vocab_size = tokenizer.get_vocab_size()
    
    # Create model
    model = GPTModel(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        layer_norm_eps=config.model.layer_norm_eps,
        use_flash_attention=config.model.use_flash_attention,
        rope_base=config.model.rope_base,
        tie_word_embeddings=config.model.tie_word_embeddings,
        gradient_checkpointing=config.model.gradient_checkpointing,
        initializer_range=config.model.initializer_range,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Create data module
    data_module = DataModule(
        tokenizer=tokenizer,
        train_dataset_paths=config.training.dataset_paths,
        eval_dataset_paths=config.training.eval_dataset_paths,
        max_seq_len=config.training.max_seq_len,
        pack_sequences=True,
        use_streaming=False,
    )
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        data_module=data_module,
        output_dir=config.training.output_dir
    )
    
    # Start training
    trainer.train()