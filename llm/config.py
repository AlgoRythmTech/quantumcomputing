"""Configuration system for the LLM training pipeline."""

from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
import yaml
import json
from pathlib import Path


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer training and usage."""
    vocab_size: int = 32000
    model_type: str = "bpe"  # or "unigram"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    character_coverage: float = 0.9995
    input_sentence_size: int = 10_000_000
    shuffle_input_sentence: bool = True
    normalization_rule_name: str = "nmt_nfkc_cf"
    
    # Training data
    input_glob: Optional[str] = None
    input_files: Optional[List[str]] = None
    output_dir: str = "artifacts/tokenizer"


@dataclass
class ModelConfig:
    """Configuration for the Transformer model architecture."""
    vocab_size: int = 32000
    max_seq_len: int = 4096
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    
    # Attention settings
    use_flash_attention: bool = True
    rope_base: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Weight tying
    tie_word_embeddings: bool = True
    
    # Gradient checkpointing
    gradient_checkpointing: bool = False
    
    # Initialization
    initializer_range: float = 0.02


@dataclass
class TrainingConfig:
    """Configuration for training procedure."""
    # Data
    dataset_paths: List[str] = field(default_factory=list)
    tokenizer_path: str = "artifacts/tokenizer"
    max_seq_len: int = 4096
    
    # Training parameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    num_epochs: int = 1
    max_steps: Optional[int] = None
    
    # Mixed precision
    use_amp: bool = True
    
    # Optimizer
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Scheduler
    scheduler: str = "cosine"  # "linear", "constant", "cosine"
    warmup_steps: int = 2000
    
    # Checkpointing
    output_dir: str = "artifacts/checkpoints"
    save_every_n_steps: int = 1000
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # Evaluation
    eval_every_n_steps: int = 500
    eval_dataset_paths: Optional[List[str]] = None
    
    # Logging
    logging_steps: int = 10
    log_level: str = "INFO"
    
    # Distributed training
    local_rank: int = -1
    
    # Miscellaneous
    seed: int = 42
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    model_path: str = "artifacts/checkpoints/best"
    tokenizer_path: str = "artifacts/tokenizer"
    
    # Test data
    test_dataset_paths: List[str] = field(default_factory=list)
    max_seq_len: int = 4096
    batch_size: int = 16
    
    # Metrics
    compute_perplexity: bool = True
    compute_bleu: bool = False
    compute_rouge: bool = False
    
    # Generation settings for BLEU/ROUGE
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    
    # Output
    output_dir: str = "artifacts/evaluation"


@dataclass
class Config:
    """Main configuration combining all components."""
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Convert nested dictionaries to dataclasses
        config = cls()
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'tokenizer' in data:
            config.tokenizer = TokenizerConfig(**data['tokenizer'])
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        if 'evaluation' in data:
            config.evaluation = EvaluationConfig(**data['evaluation'])
            
        return config
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model': self.model.__dict__,
            'tokenizer': self.tokenizer.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, indent=2, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model': self.model.__dict__,
            'tokenizer': self.tokenizer.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__
        }


def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from file (YAML or JSON)."""
    path = Path(path)
    if path.suffix in ['.yaml', '.yml']:
        return Config.from_yaml(path)
    elif path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Similar conversion as YAML but from JSON
        config = Config()
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'tokenizer' in data:
            config.tokenizer = TokenizerConfig(**data['tokenizer'])
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        if 'evaluation' in data:
            config.evaluation = EvaluationConfig(**data['evaluation'])
        return config
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")