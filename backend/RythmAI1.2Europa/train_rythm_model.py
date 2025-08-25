"""
Rythm AI 1.2 Europa - Training Pipeline
Advanced training system for 8B parameter financial expert model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

import os
import json
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import math
import warnings
warnings.filterwarnings("ignore")

from rythm_model_architecture import RythmForCausalLM, RythmConfig, count_parameters

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration for Rythm AI 1.2 Europa"""
    # Model configuration
    model_name: str = "rythm-europa-8b"
    model_config: Optional[RythmConfig] = None
    
    # Training hyperparameters
    batch_size: int = 4
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    min_learning_rate: float = 2e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    warmup_steps: int = 2000
    lr_scheduler_type: str = "cosine"
    num_cycles: float = 0.5
    
    # Training duration
    num_epochs: int = 3
    max_steps: int = -1
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 10
    
    # Data configuration
    max_seq_length: int = 8192
    dataset_path: str = "./financial_datasets"
    num_workers: int = 4
    
    # Optimization
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    compile_model: bool = True
    
    # Distributed training
    use_distributed: bool = torch.cuda.device_count() > 1
    local_rank: int = -1
    world_size: int = 1
    
    # Checkpointing
    output_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    save_total_limit: int = 3
    
    # Evaluation
    eval_batch_size: int = 8
    do_eval: bool = True
    eval_accumulation_steps: int = 10
    
    # Wandb logging
    use_wandb: bool = True
    wandb_project: str = "rythm-europa-training"
    wandb_run_name: Optional[str] = None
    
    # Financial expertise configuration
    financial_data_weight: float = 2.0  # Weight financial data more heavily
    use_expert_routing: bool = True
    expert_capacity_factor: float = 1.25


class FinancialDataset(Dataset):
    """Custom dataset for financial document training"""
    def __init__(self, data_path: str, max_length: int = 8192, config: TrainingConfig = None):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.config = config
        self.examples = []
        
        # Financial domain special tokens
        self.special_tokens = {
            "[TAX]": 128001,
            "[INVESTMENT]": 128002,
            "[ACCOUNTING]": 128003,
            "[REGULATION]": 128004,
            "[CALCULATION]": 128005,
            "[DOCUMENT]": 128006,
            "[TABLE]": 128007,
            "[FORMULA]": 128008,
        }
        
        # Load and prepare data
        self.load_financial_data()
        
    def load_financial_data(self):
        """Load financial training data from multiple sources"""
        logger.info(f"Loading financial data from {self.data_path}")
        
        # Create synthetic financial training data
        self.create_synthetic_financial_data()
        
        # Load tax code documents
        self.load_tax_codes()
        
        # Load financial regulations
        self.load_financial_regulations()
        
        # Load accounting standards
        self.load_accounting_standards()
        
        logger.info(f"Loaded {len(self.examples)} training examples")
    
    def create_synthetic_financial_data(self):
        """Create synthetic financial training examples"""
        # Tax calculation examples
        tax_examples = [
            {
                "context": "[TAX] Calculate income tax for an individual with annual income of $150,000",
                "response": "For an individual with $150,000 annual income:\n1. Federal Tax Calculation:\n   - First $11,000: 10% = $1,100\n   - $11,001-$44,725: 12% = $4,047\n   - $44,726-$95,375: 22% = $11,143\n   - $95,376-$150,000: 24% = $13,110\n   Total Federal Tax: $29,400\n\n2. Effective Tax Rate: 19.6%\n3. After-tax Income: $120,600\n\n[CALCULATION] Based on 2024 US federal tax brackets for single filers."
            },
            {
                "context": "[INVESTMENT] Analyze portfolio allocation for retirement planning",
                "response": "Optimal retirement portfolio allocation:\n1. Age-based allocation (60 years old):\n   - Stocks: 40% (100 - age rule)\n   - Bonds: 40%\n   - Real Estate: 10%\n   - Commodities: 5%\n   - Cash: 5%\n\n2. Risk-adjusted approach:\n   - Growth stocks: 20%\n   - Dividend stocks: 20%\n   - Government bonds: 25%\n   - Corporate bonds: 15%\n   - REITs: 10%\n   - Gold/Silver: 5%\n   - Emergency fund: 5%\n\n[INVESTMENT] Rebalance quarterly to maintain target allocation."
            },
            {
                "context": "[ACCOUNTING] Prepare balance sheet for small business",
                "response": "Balance Sheet Preparation:\n\nASSETS:\nCurrent Assets:\n- Cash: $50,000\n- Accounts Receivable: $30,000\n- Inventory: $40,000\nTotal Current: $120,000\n\nFixed Assets:\n- Equipment: $80,000\n- Less: Depreciation: ($20,000)\nNet Fixed: $60,000\n\nTOTAL ASSETS: $180,000\n\nLIABILITIES & EQUITY:\nCurrent Liabilities:\n- Accounts Payable: $25,000\n- Short-term Loans: $15,000\nTotal Current: $40,000\n\nLong-term Debt: $60,000\n\nOwner's Equity: $80,000\n\nTOTAL LIABILITIES & EQUITY: $180,000\n\n[ACCOUNTING] Balance sheet balances correctly."
            },
            {
                "context": "[REGULATION] Explain GDPR compliance for financial data",
                "response": "GDPR Compliance for Financial Services:\n\n1. Lawful Basis:\n   - Consent for marketing\n   - Contract for services\n   - Legal obligation for AML/KYC\n   - Legitimate interests for fraud prevention\n\n2. Data Subject Rights:\n   - Right to access financial records\n   - Right to rectification of errors\n   - Right to erasure (with exceptions)\n   - Right to data portability\n   - Right to object to processing\n\n3. Technical Measures:\n   - Encryption at rest and transit\n   - Pseudonymization of personal data\n   - Access controls and audit logs\n   - Regular security assessments\n\n4. Organizational Measures:\n   - Data Protection Officer appointment\n   - Privacy Impact Assessments\n   - Staff training programs\n   - Vendor management procedures\n\n[REGULATION] Non-compliance penalties up to 4% of global turnover."
            }
        ]
        
        # GST calculations for India
        gst_examples = [
            {
                "context": "[TAX] Calculate GST for inter-state supply of goods worth ₹100,000",
                "response": "GST Calculation for Inter-state Supply:\n\nGoods Value: ₹100,000\nIGST Rate: 18% (assuming standard rate)\n\nCalculation:\n- IGST = ₹100,000 × 18% = ₹18,000\n- Total Invoice Value = ₹118,000\n\nTax Credit Chain:\n- Supplier claims ITC on inputs\n- Buyer can claim ITC of ₹18,000\n- Net GST liability depends on output tax\n\n[TAX] File GSTR-1 by 11th of next month, GSTR-3B by 20th."
            }
        ]
        
        # Add examples to dataset
        for example in tax_examples + gst_examples:
            self.examples.append({
                "input": example["context"],
                "output": example["response"],
                "type": "financial_expert"
            })
    
    def load_tax_codes(self):
        """Load tax code documentation"""
        tax_codes = [
            {
                "input": "[TAX] Explain Section 80C of Indian Income Tax Act",
                "output": "Section 80C - Deductions for Investments:\n\nMaximum Limit: ₹1,50,000 per financial year\n\nEligible Investments:\n1. Life Insurance Premiums\n2. Employee Provident Fund (EPF)\n3. Public Provident Fund (PPF)\n4. National Savings Certificate (NSC)\n5. 5-year Tax Saving FD\n6. Equity Linked Savings Scheme (ELSS)\n7. Tuition fees for children\n8. Home loan principal repayment\n9. Sukanya Samriddhi Yojana\n10. National Pension System (additional ₹50,000 under 80CCD(1B))\n\n[TAX] Reduces taxable income, not tax liability directly.",
                "type": "tax_code"
            },
            {
                "input": "[TAX] Explain Section 179 depreciation in US tax code",
                "output": "Section 179 - Immediate Expense Deduction:\n\n2024 Limits:\n- Maximum Deduction: $1,220,000\n- Phase-out Threshold: $3,050,000\n- Maximum Vehicle Deduction: $28,900\n\nQualifying Property:\n1. Tangible personal property\n2. Machinery and equipment\n3. Computers and software\n4. Office furniture\n5. Qualified improvement property\n6. Certain vehicles over 6,000 lbs GVWR\n\nBenefits:\n- Immediate full deduction vs. depreciation\n- Reduces current year tax liability\n- Improves cash flow\n- Can create NOL to carry forward\n\n[TAX] Must be placed in service during tax year.",
                "type": "tax_code"
            }
        ]
        
        self.examples.extend(tax_codes)
    
    def load_financial_regulations(self):
        """Load financial regulation documentation"""
        regulations = [
            {
                "input": "[REGULATION] Explain Basel III capital requirements",
                "output": "Basel III Capital Requirements:\n\n1. Minimum Capital Ratios:\n   - Common Equity Tier 1 (CET1): 4.5%\n   - Tier 1 Capital: 6.0%\n   - Total Capital: 8.0%\n\n2. Capital Buffers:\n   - Capital Conservation Buffer: 2.5%\n   - Countercyclical Buffer: 0-2.5%\n   - G-SIB Surcharge: 1-3.5%\n\n3. Leverage Ratio:\n   - Minimum: 3%\n   - G-SIBs: Additional buffer\n\n4. Liquidity Requirements:\n   - Liquidity Coverage Ratio (LCR): ≥100%\n   - Net Stable Funding Ratio (NSFR): ≥100%\n\n[REGULATION] Full implementation by 2023, affects all international banks.",
                "type": "regulation"
            }
        ]
        
        self.examples.extend(regulations)
    
    def load_accounting_standards(self):
        """Load accounting standards documentation"""
        accounting = [
            {
                "input": "[ACCOUNTING] Explain IFRS 9 Financial Instruments",
                "output": "IFRS 9 Financial Instruments:\n\n1. Classification of Financial Assets:\n   - Amortized Cost\n   - Fair Value through OCI (FVOCI)\n   - Fair Value through P&L (FVTPL)\n\n2. Expected Credit Loss (ECL) Model:\n   - Stage 1: 12-month ECL\n   - Stage 2: Lifetime ECL (increased risk)\n   - Stage 3: Lifetime ECL (credit-impaired)\n\n3. Hedge Accounting:\n   - Fair value hedges\n   - Cash flow hedges\n   - Net investment hedges\n\n4. Key Changes from IAS 39:\n   - Forward-looking impairment model\n   - Simplified classification\n   - Improved hedge accounting\n\n[ACCOUNTING] Mandatory for annual periods beginning January 1, 2018.",
                "type": "accounting"
            }
        ]
        
        self.examples.extend(accounting)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize (simplified - in production use proper tokenizer)
        input_ids = self.simple_tokenize(example["input"], example["output"])
        
        # Pad or truncate to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        else:
            input_ids = input_ids + [0] * (self.max_length - len(input_ids))
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
            "type": example["type"]
        }
    
    def simple_tokenize(self, input_text, output_text):
        """Simple tokenization for demonstration"""
        # In production, use proper tokenizer like SentencePiece or BPE
        text = f"{input_text}\n{output_text}"
        
        # Simple character-level tokenization for demo
        tokens = []
        for char in text:
            tokens.append(ord(char) % 128000)  # Map to vocab size
        
        return tokens


class RythmTrainer:
    """Advanced trainer for Rythm AI 1.2 Europa"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize distributed training if multiple GPUs
        if config.use_distributed:
            self.setup_distributed()
        
        # Create model
        self.model = self.create_model()
        
        # Setup optimizer and scheduler
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Initialize wandb
        if config.use_wandb and self.is_main_process():
            self.setup_wandb()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
    def setup_distributed(self):
        """Setup distributed training"""
        init_process_group(backend='nccl')
        self.config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.config.world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(self.config.local_rank)
        
    def create_model(self):
        """Create and initialize model"""
        logger.info("Creating Rythm AI 1.2 Europa model...")
        
        if self.config.model_config is None:
            self.config.model_config = RythmConfig()
        
        model = RythmForCausalLM(self.config.model_config)
        
        # Count parameters
        param_count = count_parameters(model)
        logger.info(f"Model initialized with {param_count:,} parameters ({param_count/1e9:.2f}B)")
        
        # Move to device
        model = model.to(self.device)
        
        # Compile model for faster training (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile()...")
            model = torch.compile(model)
        
        # Distributed Data Parallel
        if self.config.use_distributed:
            model = DDP(model, device_ids=[self.config.local_rank])
        
        # Gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model
    
    def create_optimizer(self):
        """Create AdamW optimizer with weight decay"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.epsilon
        )
        
        return optimizer
    
    def create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.lr_scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.warmup_steps,
                T_mult=2,
                eta_min=self.config.min_learning_rate
            )
        elif self.config.lr_scheduler_type == "linear":
            scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.warmup_steps
            )
        else:
            scheduler = None
        
        return scheduler
    
    def setup_wandb(self):
        """Initialize Weights & Biases logging"""
        run_name = self.config.wandb_run_name or f"rythm-europa-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        wandb.init(
            project=self.config.wandb_project,
            name=run_name,
            config=vars(self.config)
        )
        
        # Watch model
        wandb.watch(self.model, log_freq=100)
    
    def is_main_process(self):
        """Check if this is the main process in distributed training"""
        return not self.config.use_distributed or self.config.local_rank == 0
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not self.is_main_process())
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Mixed precision training
            with autocast(enabled=self.config.use_mixed_precision):
                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels
                )
                loss = outputs['loss']
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                if self.config.use_mixed_precision:
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
            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            if self.global_step % self.config.logging_steps == 0 and self.is_main_process():
                avg_loss = epoch_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.2e}',
                    'step': self.global_step
                })
                
                if self.config.use_wandb:
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/learning_rate': lr,
                        'train/epoch': epoch,
                        'train/global_step': self.global_step
                    })
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0 and self.is_main_process():
                self.save_checkpoint()
        
        return epoch_loss / num_batches
    
    def evaluate(self, eval_loader):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", disable=not self.is_main_process()):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                with autocast(enabled=self.config.use_mixed_precision):
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=labels
                    )
                    loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if self.is_main_process():
            logger.info(f"Validation Loss: {avg_loss:.4f}")
            
            if self.config.use_wandb:
                wandb.log({
                    'eval/loss': avg_loss,
                    'eval/epoch': self.epoch,
                    'eval/global_step': self.global_step
                })
        
        return avg_loss
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), checkpoint_dir / "model.pt")
        
        # Save optimizer and scheduler
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        if self.scheduler:
            torch.save(self.scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': vars(self.config)
        }
        
        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"Checkpoint saved at {checkpoint_dir}")
        
        # Keep only last N checkpoints
        self.cleanup_checkpoints()
    
    def cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones"""
        checkpoint_dirs = sorted(Path(self.config.output_dir).glob("checkpoint-*"))
        
        if len(checkpoint_dirs) > self.config.save_total_limit:
            for checkpoint_dir in checkpoint_dirs[:-self.config.save_total_limit]:
                logger.info(f"Removing old checkpoint: {checkpoint_dir}")
                for file in checkpoint_dir.iterdir():
                    file.unlink()
                checkpoint_dir.rmdir()
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training Rythm AI 1.2 Europa...")
        logger.info(f"Training on {self.device}")
        
        # Create datasets
        train_dataset = FinancialDataset(
            self.config.dataset_path,
            max_length=self.config.max_seq_length,
            config=self.config
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            if self.config.do_eval:
                eval_loss = self.evaluate(train_loader)  # Using train_loader as eval for demo
                
                # Save best model
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.save_checkpoint()
                    logger.info(f"New best model saved with loss: {eval_loss:.4f}")
            
            # Log epoch summary
            if self.is_main_process():
                logger.info(f"Epoch {epoch + 1} Summary:")
                logger.info(f"  Train Loss: {train_loss:.4f}")
                if self.config.do_eval:
                    logger.info(f"  Eval Loss: {eval_loss:.4f}")
                logger.info(f"  Best Loss: {self.best_loss:.4f}")
        
        logger.info("\nTraining completed!")
        
        # Final save
        if self.is_main_process():
            final_checkpoint = Path(self.config.output_dir) / "final_model"
            final_checkpoint.mkdir(parents=True, exist_ok=True)
            
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            torch.save(model_to_save.state_dict(), final_checkpoint / "model.pt")
            
            # Save config
            self.config.model_config.__dict__.pop('__post_init__', None)  # Remove post_init before saving
            config_dict = vars(self.config.model_config)
            with open(final_checkpoint / "config.json", 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Final model saved at {final_checkpoint}")
        
        # Cleanup distributed training
        if self.config.use_distributed:
            destroy_process_group()
        
        if self.config.use_wandb and self.is_main_process():
            wandb.finish()


def main():
    """Main training entry point"""
    print("=" * 80)
    print("RYTHM AI 1.2 EUROPA - TRAINING SYSTEM")
    print("Training 8B Parameter Financial Expert Model")
    print("=" * 80)
    
    # Create training configuration
    config = TrainingConfig(
        model_name="rythm-europa-8b",
        batch_size=4,
        learning_rate=2e-4,
        num_epochs=3,
        max_seq_length=8192,
        output_dir="./checkpoints",
        use_wandb=False,  # Set to True if you have wandb configured
        use_mixed_precision=torch.cuda.is_available(),
        gradient_checkpointing=True,
        compile_model=False  # Set to True if using PyTorch 2.0+
    )
    
    # Check available resources
    if torch.cuda.is_available():
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("\nWarning: No GPU available, training will be slow!")
        print("For optimal performance, use a machine with NVIDIA GPUs")
    
    # Create trainer
    trainer = RythmTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
