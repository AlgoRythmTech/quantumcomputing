"""SentencePiece tokenizer implementation."""

import os
import glob
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import sentencepiece as spm
import json


class SPTokenizer:
    """SentencePiece tokenizer wrapper with special tokens."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: int = 32000,
        model_type: str = "bpe",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        # Special tokens list
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token]
        
        self.sp_model = spm.SentencePieceProcessor()
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self.sp_model = None
    
    def train(
        self,
        input_files: List[str],
        output_dir: str,
        character_coverage: float = 0.9995,
        input_sentence_size: int = 10_000_000,
        shuffle_input_sentence: bool = True,
        normalization_rule_name: str = "nmt_nfkc_cf",
        **kwargs
    ) -> None:
        """Train SentencePiece tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        
        model_prefix = os.path.join(output_dir, "tokenizer")
        
        # Prepare user-defined symbols
        user_defined_symbols = ",".join(self.special_tokens)
        
        # Training arguments
        train_args = [
            f"--input={','.join(input_files)}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={self.vocab_size}",
            f"--model_type={self.model_type}",
            f"--character_coverage={character_coverage}",
            f"--input_sentence_size={input_sentence_size}",
            f"--shuffle_input_sentence={shuffle_input_sentence}",
            f"--normalization_rule_name={normalization_rule_name}",
            f"--user_defined_symbols={user_defined_symbols}",
            f"--pad_id=0",
            f"--unk_id=1",
            f"--bos_id=2",
            f"--eos_id=3",
            "--split_by_whitespace=false",
            "--add_dummy_prefix=true",
            "--remove_extra_whitespaces=true",
            "--hard_vocab_limit=false",
        ]
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if isinstance(value, bool):
                value = str(value).lower()
            train_args.append(f"--{key}={value}")
        
        # Train the model
        spm.SentencePieceTrainer.train(" ".join(train_args))
        
        # Load the trained model
        self.load(model_prefix + ".model")
        
        # Save tokenizer config
        self._save_config(output_dir)
        
        print(f"Tokenizer trained and saved to {output_dir}")
        print(f"Vocabulary size: {self.get_vocab_size()}")
        
    def load(self, model_path: str) -> None:
        """Load pre-trained SentencePiece model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        
        # Load config if available
        config_path = os.path.join(os.path.dirname(model_path), "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.vocab_size = config.get('vocab_size', self.vocab_size)
                self.pad_token = config.get('pad_token', self.pad_token)
                self.unk_token = config.get('unk_token', self.unk_token)
                self.bos_token = config.get('bos_token', self.bos_token)
                self.eos_token = config.get('eos_token', self.eos_token)
                self.special_tokens = config.get('special_tokens', self.special_tokens)
    
    def _save_config(self, output_dir: str) -> None:
        """Save tokenizer configuration."""
        config = {
            'vocab_size': self.vocab_size,
            'model_type': self.model_type,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'special_tokens': self.special_tokens,
            'pad_token_id': self.pad_token_id,
            'unk_token_id': self.unk_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
        }
        
        config_path = os.path.join(output_dir, "tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def encode(
        self,
        text: Union[str, List[str]], 
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None
    ) -> Union[List[int], List[List[int]]]:
        """Encode text to token IDs."""
        if self.sp_model is None:
            raise ValueError("Tokenizer not loaded. Train or load a model first.")
        
        if isinstance(text, str):
            # Single string
            tokens = self.sp_model.encode(text, out_type=int)
            if add_bos:
                tokens = [self.bos_token_id] + tokens
            if add_eos:
                tokens = tokens + [self.eos_token_id]
            if max_length is not None and len(tokens) > max_length:
                tokens = tokens[:max_length]
            return tokens
        else:
            # List of strings
            result = []
            for t in text:
                tokens = self.sp_model.encode(t, out_type=int)
                if add_bos:
                    tokens = [self.bos_token_id] + tokens
                if add_eos:
                    tokens = tokens + [self.eos_token_id]
                if max_length is not None and len(tokens) > max_length:
                    tokens = tokens[:max_length]
                result.append(tokens)
            return result
    
    def decode(
        self, 
        token_ids: Union[List[int], List[List[int]]], 
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        """Decode token IDs to text."""
        if self.sp_model is None:
            raise ValueError("Tokenizer not loaded. Train or load a model first.")
        
        if isinstance(token_ids[0], int) if token_ids else True:
            # Single sequence
            if skip_special_tokens:
                # Filter out special token IDs
                special_ids = {self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id}
                token_ids = [t for t in token_ids if t not in special_ids]
            return self.sp_model.decode(token_ids)
        else:
            # List of sequences
            result = []
            for tokens in token_ids:
                if skip_special_tokens:
                    special_ids = {self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id}
                    tokens = [t for t in tokens if t not in special_ids]
                result.append(self.sp_model.decode(tokens))
            return result
    
    def encode_as_pieces(self, text: str) -> List[str]:
        """Encode text to subword pieces."""
        if self.sp_model is None:
            raise ValueError("Tokenizer not loaded.")
        return self.sp_model.encode(text, out_type=str)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.sp_model is None:
            return self.vocab_size
        return self.sp_model.get_piece_size()
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary as dict."""
        if self.sp_model is None:
            raise ValueError("Tokenizer not loaded.")
        
        vocab = {}
        for i in range(self.sp_model.get_piece_size()):
            piece = self.sp_model.id_to_piece(i)
            vocab[piece] = i
        return vocab
    
    @property
    def pad_token_id(self) -> int:
        """Pad token ID."""
        if self.sp_model is None:
            return 0  # Default
        return self.sp_model.piece_to_id(self.pad_token)
    
    @property
    def unk_token_id(self) -> int:
        """Unknown token ID."""
        if self.sp_model is None:
            return 1  # Default
        return self.sp_model.piece_to_id(self.unk_token)
    
    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID."""
        if self.sp_model is None:
            return 2  # Default
        return self.sp_model.piece_to_id(self.bos_token)
    
    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        if self.sp_model is None:
            return 3  # Default
        return self.sp_model.piece_to_id(self.eos_token)
    
    def save_pretrained(self, output_dir: str) -> None:
        """Save tokenizer to directory (compatible with transformers style)."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy model file
        import shutil
        if hasattr(self.sp_model, 'serialized_model_proto'):
            model_path = os.path.join(output_dir, "tokenizer.model")
            with open(model_path, 'wb') as f:
                f.write(self.sp_model.serialized_model_proto())
        
        # Save config
        self._save_config(output_dir)
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "SPTokenizer":
        """Load tokenizer from directory."""
        if os.path.isdir(model_path):
            # Look for tokenizer.model or tokenizer.model file
            model_file = None
            for filename in ["tokenizer.model", "tokenizer.model"]:
                candidate = os.path.join(model_path, filename)
                if os.path.exists(candidate):
                    model_file = candidate
                    break
            
            if model_file is None:
                raise FileNotFoundError(f"No tokenizer model found in {model_path}")
        else:
            model_file = model_path
        
        return cls(model_path=model_file, **kwargs)


def train_tokenizer_from_glob(
    input_glob: str,
    output_dir: str,
    vocab_size: int = 32000,
    model_type: str = "bpe",
    **kwargs
) -> SPTokenizer:
    """Train tokenizer from glob pattern."""
    input_files = glob.glob(input_glob)
    if not input_files:
        raise ValueError(f"No files found matching pattern: {input_glob}")
    
    print(f"Found {len(input_files)} files matching pattern: {input_glob}")
    
    tokenizer = SPTokenizer(vocab_size=vocab_size, model_type=model_type, **kwargs)
    tokenizer.train(input_files=input_files, output_dir=output_dir, **kwargs)
    
    return tokenizer