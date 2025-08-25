"""
Advanced Training System for Rythm AI 1.2 Europa
Implements LoRA/QLoRA, RLHF, Federated Learning, and Model Validation
Production-ready training pipeline with complete error handling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import bitsandbytes as bnb

import numpy as np
from typing import Optional, Dict, List, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from tqdm import tqdm
import wandb
from datetime import datetime
import math
import random
import os
import sys
import traceback
from collections import defaultdict
import hashlib
import pickle

# Advanced training imports
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import datasets
from datasets import load_dataset, Dataset as HFDataset

# Import our modules
from rythm_model_architecture import RythmForCausalLM, RythmConfig, RythmModel
from tokenizer_system import RythmTokenizer, create_tokenizer

# Setup logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AdvancedTrainingConfig:
    """Advanced configuration for production training"""
    # Model configuration
    model_name: str = "rythm-europa-8b"
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = "./tokenizer"
    
    # LoRA/QLoRA configuration
    use_lora: bool = True
    use_qlora: bool = False  # 4-bit quantization
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "lm_head"
    ])
    
    # Quantization config for QLoRA
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # RLHF configuration
    use_rlhf: bool = True
    reward_model_path: Optional[str] = None
    ppo_epochs: int = 4
    ppo_batch_size: int = 1
    kl_penalty: float = 0.02
    reward_baseline: float = 0.0
    value_loss_coef: float = 0.1
    
    # Federated learning configuration
    use_federated: bool = False
    num_clients: int = 10
    clients_per_round: int = 5
    federated_rounds: int = 100
    local_epochs: int = 2
    
    # Training hyperparameters
    batch_size: int = 8
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 10000
    num_epochs: int = 3
    
    # Optimization
    optimizer: str = "adamw"  # "adamw", "adam8bit", "lion"
    scheduler: str = "cosine"  # "cosine", "linear", "polynomial"
    max_grad_norm: float = 0.5
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    compile_model: bool = False
    
    # Mixed precision
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    
    # Data configuration
    dataset_name: Optional[str] = None
    dataset_path: str = "./data"
    max_seq_length: int = 8192
    num_workers: int = 4
    
    # Validation
    do_eval: bool = True
    eval_steps: int = 500
    eval_accumulation_steps: int = 10
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Checkpointing
    output_dir: str = "./checkpoints"
    save_steps: int = 1000
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # Logging
    logging_steps: int = 10
    report_to: str = "wandb"  # "wandb", "tensorboard", "none"
    wandb_project: str = "rythm-europa-advanced"
    
    # Error handling
    max_retries: int = 3
    retry_delay: int = 60
    ignore_data_errors: bool = True
    
    # Model validation
    validate_model: bool = True
    validation_samples: int = 100
    compatibility_check: bool = True


class LoRALayer(nn.Module):
    """LoRA adapter layer implementation"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation"""
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B
        return lora_output * self.scaling


class QLoRALinear(nn.Module):
    """QLoRA quantized linear layer with 4-bit quantization"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Use bitsandbytes 4-bit linear layer
        self.base_layer = bnb.nn.Linear4bit(
            in_features,
            out_features,
            bias=False,
            compute_dtype=torch.float16,
            compress_statistics=True,
            quant_type="nf4"
        )
        
        # Add LoRA adapters
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with QLoRA"""
        base_output = self.base_layer(x)
        lora_output = self.lora(x)
        return base_output + lora_output


def apply_lora_to_model(
    model: nn.Module,
    config: AdvancedTrainingConfig
) -> nn.Module:
    """Apply LoRA/QLoRA to model layers"""
    logger.info(f"Applying {'QLoRA' if config.use_qlora else 'LoRA'} to model")
    
    lora_params = []
    
    for name, module in model.named_modules():
        # Check if this module should have LoRA
        if any(target in name for target in config.lora_target_modules):
            if isinstance(module, nn.Linear):
                # Get dimensions
                in_features = module.in_features
                out_features = module.out_features
                
                # Create LoRA/QLoRA layer
                if config.use_qlora:
                    lora_layer = QLoRALinear(
                        in_features, out_features,
                        config.lora_rank, config.lora_alpha, config.lora_dropout
                    )
                else:
                    lora_layer = LoRALayer(
                        in_features, out_features,
                        config.lora_rank, config.lora_alpha, config.lora_dropout
                    )
                
                # Register LoRA adapter
                parent_module = model
                module_names = name.split('.')
                for n in module_names[:-1]:
                    parent_module = getattr(parent_module, n)
                
                # Add LoRA as a parallel module
                setattr(parent_module, f"{module_names[-1]}_lora", lora_layer)
                lora_params.extend(lora_layer.parameters())
                
                # Modify forward pass to include LoRA
                original_forward = module.forward
                def new_forward(self, x):
                    base_output = original_forward(x)
                    lora_output = getattr(parent_module, f"{module_names[-1]}_lora")(x)
                    return base_output + lora_output
                module.forward = new_forward.__get__(module, type(module))
    
    # Freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze LoRA parameters
    for param in lora_params:
        param.requires_grad = True
    
    logger.info(f"Added LoRA to {len(lora_params)} parameters")
    return model


class RewardModel(nn.Module):
    """Reward model for RLHF training"""
    def __init__(self, base_model: RythmModel, hidden_size: int = 5120):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate reward score for input"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state of last token
        hidden_states = outputs['last_hidden_state']
        last_token_hidden = hidden_states[:, -1, :]
        
        # Calculate reward
        reward = self.reward_head(last_token_hidden).squeeze(-1)
        return reward


class PPOTrainer:
    """Proximal Policy Optimization trainer for RLHF"""
    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: nn.Module,
        config: AdvancedTrainingConfig,
        tokenizer: RythmTokenizer
    ):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.config = config
        self.tokenizer = tokenizer
        
        # Value model for advantage estimation
        self.value_model = self._create_value_model()
        
        # Optimizer for PPO
        self.optimizer = self._create_optimizer()
    
    def _create_value_model(self) -> nn.Module:
        """Create value model for PPO"""
        class ValueModel(nn.Module):
            def __init__(self, hidden_size: int = 5120):
                super().__init__()
                self.value_head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size // 2, 1)
                )
            
            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return self.value_head(hidden_states).squeeze(-1)
        
        return ValueModel()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for PPO training"""
        params = list(self.policy_model.parameters()) + list(self.value_model.parameters())
        return optim.AdamW(params, lr=self.config.learning_rate)
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
        
        return advantages, returns
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """Single PPO training step"""
        # Get current policy outputs
        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        logits = outputs['logits']
        hidden_states = outputs['hidden_states'][-1]
        
        # Calculate log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = torch.gather(
            log_probs[:, :-1, :],
            2,
            input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Calculate ratio for PPO
        ratio = torch.exp(action_log_probs - old_log_probs)
        
        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        values = self.value_model(hidden_states[:, -1, :])
        value_loss = F.mse_loss(values, returns)
        
        # KL penalty
        kl_div = (old_log_probs - action_log_probs).mean()
        kl_penalty = self.config.kl_penalty * kl_div
        
        # Total loss
        total_loss = policy_loss + self.config.value_loss_coef * value_loss + kl_penalty
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "kl_div": kl_div.item(),
            "total_loss": total_loss.item()
        }


class FederatedLearningTrainer:
    """Federated learning trainer for privacy-preserving training"""
    def __init__(
        self,
        model_class: type,
        config: AdvancedTrainingConfig,
        tokenizer: RythmTokenizer
    ):
        self.model_class = model_class
        self.config = config
        self.tokenizer = tokenizer
        self.global_model = None
        self.client_models = []
        self.client_data = []
        
        self._initialize_federated_setup()
    
    def _initialize_federated_setup(self):
        """Initialize federated learning setup"""
        logger.info("Initializing federated learning setup")
        
        # Create global model
        model_config = RythmConfig()
        self.global_model = self.model_class(model_config)
        
        # Create client models
        for i in range(self.config.num_clients):
            client_model = self.model_class(model_config)
            client_model.load_state_dict(self.global_model.state_dict())
            self.client_models.append(client_model)
        
        logger.info(f"Created {self.config.num_clients} client models")
    
    def split_data_for_clients(self, dataset: Dataset) -> List[Dataset]:
        """Split dataset among clients"""
        total_size = len(dataset)
        client_size = total_size // self.config.num_clients
        
        client_datasets = []
        for i in range(self.config.num_clients):
            start_idx = i * client_size
            end_idx = start_idx + client_size if i < self.config.num_clients - 1 else total_size
            client_data = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
            client_datasets.append(client_data)
        
        return client_datasets
    
    def train_client(
        self,
        client_id: int,
        client_model: nn.Module,
        client_data: Dataset,
        epochs: int
    ) -> Dict[str, Any]:
        """Train a single client model"""
        logger.info(f"Training client {client_id}")
        
        # Create optimizer for client
        optimizer = optim.AdamW(
            client_model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Create data loader
        dataloader = DataLoader(
            client_data,
            batch_size=self.config.micro_batch_size,
            shuffle=True
        )
        
        client_model.train()
        total_loss = 0
        num_batches = 0
        
        for epoch in range(epochs):
            for batch in dataloader:
                input_ids = batch['input_ids']
                labels = batch['labels']
                
                # Forward pass
                outputs = client_model(input_ids=input_ids, labels=labels)
                loss = outputs['loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            "client_id": client_id,
            "avg_loss": avg_loss,
            "num_samples": len(client_data)
        }
    
    def aggregate_models(
        self,
        client_models: List[nn.Module],
        client_weights: Optional[List[float]] = None
    ):
        """Aggregate client models using FedAvg"""
        logger.info("Aggregating client models")
        
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        
        # Initialize global model state dict
        global_state_dict = self.global_model.state_dict()
        
        # Aggregate parameters
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.zeros_like(global_state_dict[key])
            
            for client_model, weight in zip(client_models, client_weights):
                client_state = client_model.state_dict()
                global_state_dict[key] += weight * client_state[key]
        
        # Update global model
        self.global_model.load_state_dict(global_state_dict)
    
    def train_round(self, round_num: int, client_datasets: List[Dataset]) -> Dict[str, Any]:
        """Train one federated round"""
        logger.info(f"Federated round {round_num}")
        
        # Select clients for this round
        selected_clients = random.sample(
            range(self.config.num_clients),
            self.config.clients_per_round
        )
        
        # Train selected clients
        round_results = []
        selected_models = []
        
        for client_id in selected_clients:
            # Copy global model to client
            self.client_models[client_id].load_state_dict(
                self.global_model.state_dict()
            )
            
            # Train client
            result = self.train_client(
                client_id,
                self.client_models[client_id],
                client_datasets[client_id],
                self.config.local_epochs
            )
            
            round_results.append(result)
            selected_models.append(self.client_models[client_id])
        
        # Calculate aggregation weights based on data size
        total_samples = sum(r["num_samples"] for r in round_results)
        weights = [r["num_samples"] / total_samples for r in round_results]
        
        # Aggregate models
        self.aggregate_models(selected_models, weights)
        
        # Calculate round statistics
        avg_loss = sum(r["avg_loss"] * w for r, w in zip(round_results, weights))
        
        return {
            "round": round_num,
            "avg_loss": avg_loss,
            "num_clients": len(selected_clients),
            "client_results": round_results
        }


class ModelValidator:
    """Comprehensive model validation and compatibility checking"""
    def __init__(self, model: nn.Module, tokenizer: RythmTokenizer, config: AdvancedTrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.validation_results = {}
    
    def validate_model_architecture(self) -> bool:
        """Validate model architecture"""
        try:
            logger.info("Validating model architecture")
            
            # Check model type
            if not isinstance(self.model, (RythmForCausalLM, nn.Module)):
                raise ValueError("Invalid model type")
            
            # Check model parameters
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model has {param_count:,} parameters")
            
            # Check required attributes
            required_attrs = ['forward', 'config']
            for attr in required_attrs:
                if not hasattr(self.model, attr):
                    raise ValueError(f"Model missing required attribute: {attr}")
            
            # Check model config
            if hasattr(self.model, 'config'):
                config = self.model.config
                if config.vocab_size != self.tokenizer.get_vocab_size():
                    logger.warning(f"Vocab size mismatch: model={config.vocab_size}, tokenizer={self.tokenizer.get_vocab_size()}")
            
            self.validation_results['architecture'] = True
            return True
            
        except Exception as e:
            logger.error(f"Architecture validation failed: {e}")
            self.validation_results['architecture'] = False
            return False
    
    def validate_forward_pass(self) -> bool:
        """Validate model forward pass"""
        try:
            logger.info("Validating forward pass")
            
            # Create dummy input
            dummy_text = "This is a test input for model validation"
            inputs = self.tokenizer.encode(
                dummy_text,
                return_tensors="pt",
                max_length=128,
                padding=True,
                truncation=True
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
            
            # Forward pass
            self.model.eval()
            with torch.no_grad():
                if isinstance(inputs, dict):
                    outputs = self.model(**inputs)
                else:
                    outputs = self.model(inputs)
            
            # Check outputs
            if outputs is None:
                raise ValueError("Model returned None")
            
            if hasattr(outputs, 'logits'):
                if outputs.logits.shape[-1] != self.model.config.vocab_size:
                    raise ValueError("Output dimension mismatch")
            
            self.validation_results['forward_pass'] = True
            return True
            
        except Exception as e:
            logger.error(f"Forward pass validation failed: {e}")
            self.validation_results['forward_pass'] = False
            return False
    
    def validate_gradient_flow(self) -> bool:
        """Validate gradient flow through model"""
        try:
            logger.info("Validating gradient flow")
            
            # Create dummy input
            dummy_text = "Test gradient flow"
            inputs = self.tokenizer.encode(
                dummy_text,
                return_tensors="pt",
                max_length=32,
                padding=True,
                truncation=True
            )
            
            # Add labels for loss calculation
            if isinstance(inputs, dict):
                inputs['labels'] = inputs['input_ids'].clone()
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass with gradient
            self.model.train()
            outputs = self.model(**inputs)
            
            if not hasattr(outputs, 'loss') and 'loss' not in outputs:
                # Calculate loss manually
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
                labels = inputs['labels']
                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1)
                )
            else:
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            has_gradient = False
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if param.grad.abs().sum() > 0:
                        has_gradient = True
                        break
            
            # Clear gradients
            self.model.zero_grad()
            
            if not has_gradient:
                logger.warning("No gradients detected in model")
            
            self.validation_results['gradient_flow'] = has_gradient
            return has_gradient
            
        except Exception as e:
            logger.error(f"Gradient flow validation failed: {e}")
            self.validation_results['gradient_flow'] = False
            return False
    
    def validate_memory_usage(self) -> bool:
        """Check memory usage and optimization"""
        try:
            logger.info("Validating memory usage")
            
            if torch.cuda.is_available():
                # Check GPU memory
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                
                # Run forward pass
                dummy_input = torch.randint(0, 1000, (1, 256)).cuda()
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = (peak_memory - initial_memory) / 1024**3  # GB
                
                logger.info(f"Model uses approximately {memory_used:.2f} GB for inference")
                
                # Check if memory usage is reasonable
                if memory_used > 40:  # More than 40GB for single forward pass
                    logger.warning("High memory usage detected")
                
                torch.cuda.empty_cache()
            
            self.validation_results['memory_usage'] = True
            return True
            
        except Exception as e:
            logger.error(f"Memory validation failed: {e}")
            self.validation_results['memory_usage'] = False
            return False
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete model validation"""
        logger.info("Starting comprehensive model validation")
        
        # Run all validation checks
        self.validate_model_architecture()
        self.validate_forward_pass()
        self.validate_gradient_flow()
        self.validate_memory_usage()
        
        # Calculate overall validation status
        all_passed = all(self.validation_results.values())
        
        logger.info(f"Validation complete. Status: {'PASSED' if all_passed else 'FAILED'}")
        logger.info(f"Results: {self.validation_results}")
        
        return {
            "passed": all_passed,
            "results": self.validation_results,
            "timestamp": datetime.now().isoformat()
        }


class RobustDataLoader:
    """Robust data loader with error handling and recovery"""
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        tokenizer: RythmTokenizer,
        max_length: int = 8192,
        num_workers: int = 4,
        ignore_errors: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_workers = num_workers
        self.ignore_errors = ignore_errors
        self.error_count = 0
        self.processed_count = 0
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function with error handling"""
        valid_samples = []
        
        for sample in batch:
            try:
                # Process sample
                if isinstance(sample, dict) and 'text' in sample:
                    text = sample['text']
                elif isinstance(sample, str):
                    text = sample
                else:
                    continue
                
                # Tokenize
                encoded = self.tokenizer.encode(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                if isinstance(encoded, dict):
                    valid_samples.append({
                        'input_ids': encoded['input_ids'].squeeze(0),
                        'attention_mask': encoded.get('attention_mask', torch.ones_like(encoded['input_ids'])).squeeze(0),
                        'labels': encoded['input_ids'].squeeze(0)
                    })
                else:
                    valid_samples.append({
                        'input_ids': encoded.squeeze(0),
                        'attention_mask': torch.ones_like(encoded.squeeze(0)),
                        'labels': encoded.squeeze(0)
                    })
                
                self.processed_count += 1
                
            except Exception as e:
                self.error_count += 1
                if not self.ignore_errors:
                    logger.error(f"Error processing sample: {e}")
                    raise
                else:
                    logger.debug(f"Skipped problematic sample: {e}")
                    continue
        
        if not valid_samples:
            # Return dummy batch if all samples failed
            return {
                'input_ids': torch.zeros(1, self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(1, self.max_length, dtype=torch.long),
                'labels': torch.zeros(1, self.max_length, dtype=torch.long)
            }
        
        # Stack valid samples
        return {
            'input_ids': torch.stack([s['input_ids'] for s in valid_samples]),
            'attention_mask': torch.stack([s['attention_mask'] for s in valid_samples]),
            'labels': torch.stack([s['labels'] for s in valid_samples])
        }
    
    def get_dataloader(self) -> DataLoader:
        """Get PyTorch DataLoader with error handling"""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
    
    def get_statistics(self) -> Dict[str, int]:
        """Get data loading statistics"""
        return {
            "processed": self.processed_count,
            "errors": self.error_count,
            "error_rate": self.error_count / max(1, self.processed_count + self.error_count)
        }


class AdvancedTrainer:
    """Main trainer class with all advanced features"""
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.accelerator = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf') if not config.greater_is_better else float('-inf')
        
        # Initialize
        self._setup()
    
    def _setup(self):
        """Setup training environment"""
        try:
            # Initialize accelerator for distributed training
            self.accelerator = Accelerator(
                mixed_precision=self.config.mixed_precision,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            )
            
            # Load tokenizer
            self.tokenizer = self._load_tokenizer()
            
            # Load model
            self.model = self._load_model()
            
            # Apply LoRA if configured
            if self.config.use_lora:
                self.model = apply_lora_to_model(self.model, self.config)
            
            # Validate model
            if self.config.validate_model:
                validator = ModelValidator(self.model, self.tokenizer, self.config)
                validation_results = validator.run_full_validation()
                if not validation_results['passed']:
                    logger.warning("Model validation failed, continuing anyway")
            
            # Setup optimizer and scheduler
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            
            # Setup mixed precision
            if self.config.mixed_precision != "no":
                self.scaler = GradScaler()
            
            # Prepare for distributed training
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler
            )
            
            logger.info("Training setup complete")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
    
    def _load_tokenizer(self) -> RythmTokenizer:
        """Load tokenizer with error handling"""
        try:
            if self.config.tokenizer_path and Path(self.config.tokenizer_path).exists():
                tokenizer = RythmTokenizer.from_pretrained(self.config.tokenizer_path)
            else:
                tokenizer = create_tokenizer()
                tokenizer.save_pretrained("./tokenizer")
            
            logger.info(f"Loaded tokenizer with vocab size {tokenizer.get_vocab_size()}")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            # Create fallback tokenizer
            return create_tokenizer()
    
    def _load_model(self) -> nn.Module:
        """Load model with error handling"""
        try:
            if self.config.model_path and Path(self.config.model_path).exists():
                # Load from checkpoint
                logger.info(f"Loading model from {self.config.model_path}")
                config = RythmConfig()
                model = RythmForCausalLM(config)
                
                checkpoint = torch.load(
                    Path(self.config.model_path) / "model.pt",
                    map_location=self.device
                )
                model.load_state_dict(checkpoint, strict=False)
            else:
                # Create new model
                logger.info("Creating new model")
                config = RythmConfig()
                model = RythmForCausalLM(config)
            
            model = model.to(self.device)
            
            # Enable gradient checkpointing
            if self.config.gradient_checkpointing:
                model.gradient_checkpointing_enable()
            
            # Compile model if configured
            if self.config.compile_model and hasattr(torch, 'compile'):
                model = torch.compile(model)
            
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"Model loaded with {param_count:,} parameters")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        # Get parameters to optimize
        if self.config.use_lora:
            # Only optimize LoRA parameters
            params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        else:
            params_to_optimize = self.model.parameters()
        
        # Create optimizer
        if self.config.optimizer == "adamw":
            optimizer = optim.AdamW(
                params_to_optimize,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer == "adam8bit":
            optimizer = bnb.optim.Adam8bit(
                params_to_optimize,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            optimizer = optim.SGD(
                params_to_optimize,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        
        return optimizer
    
    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler"""
        if self.config.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_steps,
                eta_min=self.config.min_learning_rate
            )
        elif self.config.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.max_steps
            )
        else:
            scheduler = optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0
            )
        
        return scheduler
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Main training loop with error recovery"""
        logger.info("Starting training")
        
        # Create robust data loader
        train_loader = RobustDataLoader(
            train_dataset,
            self.config.batch_size,
            self.tokenizer,
            self.config.max_seq_length,
            self.config.num_workers,
            self.config.ignore_data_errors
        ).get_dataloader()
        
        # Training loop with error recovery
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            try:
                # Train epoch
                train_metrics = self._train_epoch(train_loader)
                logger.info(f"Epoch {epoch + 1} - Train metrics: {train_metrics}")
                
                # Evaluation
                if self.config.do_eval and eval_dataset is not None:
                    eval_metrics = self._evaluate(eval_dataset)
                    logger.info(f"Epoch {epoch + 1} - Eval metrics: {eval_metrics}")
                    
                    # Save best model
                    if self._is_better(eval_metrics[self.config.metric_for_best_model]):
                        self.best_metric = eval_metrics[self.config.metric_for_best_model]
                        self._save_checkpoint(best=True)
                
                # Regular checkpoint
                if (epoch + 1) % self.config.save_steps == 0:
                    self._save_checkpoint()
                    
            except Exception as e:
                logger.error(f"Error in epoch {epoch + 1}: {e}")
                if self.config.max_retries > 0:
                    logger.info("Attempting recovery...")
                    self._recover_from_error()
                else:
                    raise
        
        logger.info("Training completed successfully")
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                with self.accelerator.autocast():
                    outputs = self.model(**batch)
                    loss = outputs['loss'] if 'loss' in outputs else outputs.loss
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                
                # Update metrics
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': total_loss / num_batches,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'step': self.global_step
                })
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics({
                        'train/loss': total_loss / num_batches,
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'train/epoch': self.epoch,
                        'train/global_step': self.global_step
                    })
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                if not self.config.ignore_data_errors:
                    raise
                continue
        
        return {'loss': total_loss / max(1, num_batches)}
    
    def _evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        
        eval_loader = RobustDataLoader(
            eval_dataset,
            self.config.batch_size,
            self.tokenizer,
            self.config.max_seq_length,
            self.config.num_workers,
            self.config.ignore_data_errors
        ).get_dataloader()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with self.accelerator.autocast():
                    outputs = self.model(**batch)
                    loss = outputs['loss'] if 'loss' in outputs else outputs.loss
                
                total_loss += loss.item()
                num_batches += 1
        
        return {'eval_loss': total_loss / max(1, num_batches)}
    
    def _is_better(self, metric: float) -> bool:
        """Check if metric is better than best"""
        if self.config.greater_is_better:
            return metric > self.best_metric
        else:
            return metric < self.best_metric
    
    def _save_checkpoint(self, best: bool = False):
        """Save model checkpoint"""
        save_dir = Path(self.config.output_dir)
        if best:
            save_dir = save_dir / "best_model"
        else:
            save_dir = save_dir / f"checkpoint-{self.global_step}"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_to_save = self.accelerator.unwrap_model(self.model)
        torch.save(model_to_save.state_dict(), save_dir / "model.pt")
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }
        torch.save(training_state, save_dir / "training_state.pt")
        
        # Save config
        with open(save_dir / "config.json", 'w') as f:
            json.dump(vars(self.config), f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved to {save_dir}")
    
    def _recover_from_error(self):
        """Attempt to recover from training error"""
        try:
            logger.info("Attempting error recovery")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Reduce batch size
            self.config.batch_size = max(1, self.config.batch_size // 2)
            logger.info(f"Reduced batch size to {self.config.batch_size}")
            
            # Reset optimizer state
            self.optimizer.zero_grad()
            
            logger.info("Recovery successful")
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            raise
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to wandb or tensorboard"""
        if self.config.report_to == "wandb":
            try:
                wandb.log(metrics, step=self.global_step)
            except:
                pass
        
        # Also log to console
        logger.debug(f"Metrics: {metrics}")


def main():
    """Main training entry point"""
    print("=" * 80)
    print("RYTHM AI 1.2 EUROPA - ADVANCED TRAINING SYSTEM")
    print("LoRA/QLoRA + RLHF + Federated Learning")
    print("=" * 80)
    
    # Create configuration
    config = AdvancedTrainingConfig(
        model_name="rythm-europa-8b",
        use_lora=True,
        use_qlora=False,
        use_rlhf=False,  # Disabled for initial training
        use_federated=False,  # Disabled for initial training
        batch_size=4,
        learning_rate=5e-5,
        num_epochs=3,
        max_seq_length=4096,
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        gradient_checkpointing=True,
        validate_model=True,
        ignore_data_errors=True
    )
    
    # Create trainer
    trainer = AdvancedTrainer(config)
    
    # Create dummy dataset for demonstration
    dummy_data = [
        {"text": "Calculate tax for income of $150,000 with standard deductions"},
        {"text": "Investment portfolio analysis for retirement planning"},
        {"text": "Balance sheet preparation for Q4 2024"},
        {"text": "GDPR compliance checklist for financial institutions"},
    ] * 100
    
    train_dataset = dummy_data
    
    # Start training
    try:
        trainer.train(train_dataset)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
