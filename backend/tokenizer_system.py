"""
Advanced Tokenizer System for Rythm AI 1.2 Europa
Production-ready tokenization with SentencePiece and Tiktoken
"""

import os
import json
import regex as re
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
import sentencepiece as spm
import tiktoken
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Configuration for Rythm AI tokenizer"""
    vocab_size: int = 128000
    model_type: str = "sentencepiece"  # "sentencepiece" or "tiktoken"
    model_file: str = "rythm_tokenizer.model"
    special_tokens: Dict[str, int] = field(default_factory=lambda: {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        # Financial domain special tokens
        "[TAX]": 5,
        "[INVESTMENT]": 6,
        "[ACCOUNTING]": 7,
        "[REGULATION]": 8,
        "[CALCULATION]": 9,
        "[DOCUMENT]": 10,
        "[TABLE]": 11,
        "[FORMULA]": 12,
        "[CURRENCY]": 13,
        "[DATE]": 14,
        "[AMOUNT]": 15,
        "[PERCENTAGE]": 16,
        # Multimodal tokens
        "[IMAGE]": 17,
        "[AUDIO]": 18,
        "[VIDEO]": 19,
        "[CHART]": 20,
        # System tokens
        "[INST]": 21,
        "[/INST]": 22,
        "[SYS]": 23,
        "[/SYS]": 24,
        "[USER]": 25,
        "[/USER]": 26,
        "[ASSISTANT]": 27,
        "[/ASSISTANT]": 28,
    })
    max_length: int = 32768
    padding_side: str = "right"
    truncation_side: str = "right"
    add_bos_token: bool = True
    add_eos_token: bool = True
    normalization: bool = True
    byte_fallback: bool = True
    split_special_tokens: bool = False
    clean_up_tokenization_spaces: bool = True


class RythmTokenizer:
    """Advanced tokenizer for Rythm AI with SentencePiece backend"""
    
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self.sp_model = None
        self.tiktoken_enc = None
        self.vocab = {}
        self.reverse_vocab = {}
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """Initialize the tokenizer based on configuration"""
        if self.config.model_type == "sentencepiece":
            self._initialize_sentencepiece()
        elif self.config.model_type == "tiktoken":
            self._initialize_tiktoken()
        else:
            raise ValueError(f"Unknown tokenizer type: {self.config.model_type}")
    
    def _initialize_sentencepiece(self):
        """Initialize SentencePiece tokenizer"""
        model_path = Path(self.config.model_file)
        
        if model_path.exists():
            # Load existing model
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(str(model_path))
            logger.info(f"Loaded SentencePiece model from {model_path}")
        else:
            # Train new model
            logger.info("Training new SentencePiece model...")
            self._train_sentencepiece_model()
        
        # Build vocabulary
        self._build_vocabulary()
    
    def _initialize_tiktoken(self):
        """Initialize Tiktoken tokenizer"""
        try:
            # Use cl100k_base encoding (GPT-4 encoding)
            self.tiktoken_enc = tiktoken.get_encoding("cl100k_base")
            logger.info("Initialized Tiktoken tokenizer")
        except Exception as e:
            logger.error(f"Failed to initialize Tiktoken: {e}")
            # Fallback to SentencePiece
            self.config.model_type = "sentencepiece"
            self._initialize_sentencepiece()
    
    def _train_sentencepiece_model(self):
        """Train a new SentencePiece model"""
        # Create training data
        training_data = self._create_training_data()
        training_file = "tokenizer_training_data.txt"
        
        with open(training_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(training_data))
        
        # Training arguments
        train_args = {
            'input': training_file,
            'model_prefix': 'rythm_tokenizer',
            'vocab_size': self.config.vocab_size,
            'character_coverage': 0.9995,
            'model_type': 'unigram',
            'pad_id': 0,
            'unk_id': 3,
            'bos_id': 1,
            'eos_id': 2,
            'pad_piece': '<pad>',
            'unk_piece': '<unk>',
            'bos_piece': '<s>',
            'eos_piece': '</s>',
            'normalization_rule_name': 'identity' if not self.config.normalization else 'nmt_nfkc',
            'remove_extra_whitespaces': True,
            'add_dummy_prefix': True,
            'split_digits': True,
            'byte_fallback': self.config.byte_fallback,
            'vocabulary_output_piece_score': True,
            'train_extremely_large_corpus': False,
            'seed_sentencepiece_size': 1000000,
            'shrinking_factor': 0.75,
            'max_sentence_length': 16384,
            'num_threads': os.cpu_count(),
            'num_sub_iterations': 2,
            'max_sentencepiece_length': 16,
            'split_by_unicode_script': True,
            'split_by_number': True,
            'split_by_whitespace': True,
            'treat_whitespace_as_suffix': False,
            'allow_whitespace_only_pieces': True,
            'user_defined_symbols': list(self.config.special_tokens.keys()),
        }
        
        # Train model
        spm.SentencePieceTrainer.train(**{k: v for k, v in train_args.items() if v is not None})
        
        # Load trained model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load('rythm_tokenizer.model')
        
        # Clean up training file
        os.remove(training_file)
        
        logger.info("SentencePiece model trained successfully")
    
    def _create_training_data(self) -> List[str]:
        """Create training data for tokenizer"""
        # Financial domain training data
        training_samples = [
            # Tax terminology
            "Calculate income tax deductions for fiscal year 2024",
            "Section 80C allows deductions up to ₹1,50,000",
            "Capital gains tax applies to investment profits",
            "GST rate is 18% for standard goods",
            "Form 16 contains salary and TDS details",
            
            # Investment terminology
            "Diversified portfolio with 60% equity 40% debt allocation",
            "P/E ratio indicates stock valuation metrics",
            "Compound annual growth rate CAGR calculation",
            "Systematic investment plan SIP returns",
            "Net asset value NAV of mutual funds",
            
            # Accounting terminology
            "Balance sheet shows assets liabilities and equity",
            "Profit and loss statement for Q4 2024",
            "Depreciation calculated using straight-line method",
            "Accounts receivable and payable reconciliation",
            "Cash flow statement operating investing financing",
            
            # Regulatory compliance
            "GDPR compliance for data protection",
            "Basel III capital adequacy requirements",
            "IFRS 9 financial instruments standard",
            "SOX Sarbanes-Oxley Act compliance",
            "Anti-money laundering AML regulations",
            
            # Mathematical expressions
            "ROI = (Gain - Cost) / Cost × 100%",
            "Future Value = Present Value × (1 + r)^n",
            "Tax = Income × Tax Rate - Deductions",
            "EMI = P × r × (1+r)^n / ((1+r)^n - 1)",
            
            # Multilingual samples
            "बैलेंस शीट विश्लेषण",
            "निवेश पोर्टफोलियो",
            "कर गणना",
            
            # Numbers and currencies
            "$1,234,567.89 USD",
            "€987,654.32 EUR",
            "₹50,00,000 INR",
            "£123,456.78 GBP",
            "¥1,234,567 JPY",
            
            # Dates and percentages
            "FY 2024-25 Q1 results",
            "31st March 2024",
            "15.5% annual return",
            "7.25% interest rate",
        ]
        
        # Extend with more samples
        extended_samples = []
        for sample in training_samples:
            extended_samples.append(sample)
            extended_samples.append(sample.lower())
            extended_samples.append(sample.upper())
            # Add variations
            extended_samples.append(f"[TAX] {sample}")
            extended_samples.append(f"[INVESTMENT] {sample}")
            extended_samples.append(f"[ACCOUNTING] {sample}")
        
        # Add code samples
        code_samples = [
            "def calculate_tax(income, deductions): return max(0, income - deductions) * 0.3",
            "SELECT * FROM transactions WHERE amount > 10000",
            "if profit > 0: tax = profit * tax_rate else: tax = 0",
        ]
        
        extended_samples.extend(code_samples)
        
        return extended_samples * 100  # Repeat for better coverage
    
    def _build_vocabulary(self):
        """Build vocabulary from SentencePiece model"""
        if self.sp_model:
            vocab_size = self.sp_model.get_piece_size()
            for i in range(vocab_size):
                piece = self.sp_model.id_to_piece(i)
                self.vocab[piece] = i
                self.reverse_vocab[i] = piece
            
            # Add special tokens
            for token, token_id in self.config.special_tokens.items():
                if token not in self.vocab:
                    self.vocab[token] = token_id
                    self.reverse_vocab[token_id] = token
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = False,
        return_token_type_ids: bool = False,
    ) -> Union[List[int], Dict[str, Any]]:
        """Encode text to token IDs with full compatibility"""
        if isinstance(text, str):
            texts = [text]
            single_text = True
        else:
            texts = text
            single_text = False
        
        max_length = max_length or self.config.max_length
        encoded_texts = []
        attention_masks = []
        
        for txt in texts:
            # Preprocess text
            txt = self._preprocess_text(txt)
            
            # Encode based on tokenizer type
            if self.config.model_type == "sentencepiece" and self.sp_model:
                ids = self.sp_model.encode(txt)
            elif self.config.model_type == "tiktoken" and self.tiktoken_enc:
                ids = self.tiktoken_enc.encode(txt)
            else:
                raise ValueError("No tokenizer model loaded")
            
            # Add special tokens
            if add_special_tokens:
                if self.config.add_bos_token:
                    ids = [self.config.special_tokens["<s>"]] + ids
                if self.config.add_eos_token:
                    ids = ids + [self.config.special_tokens["</s>"]]
            
            # Truncation
            if truncation and len(ids) > max_length:
                if self.config.truncation_side == "right":
                    ids = ids[:max_length]
                else:
                    ids = ids[-max_length:]
            
            # Padding
            attention_mask = [1] * len(ids)
            if padding:
                padding_length = max_length - len(ids)
                if padding_length > 0:
                    pad_id = self.config.special_tokens["<pad>"]
                    if self.config.padding_side == "right":
                        ids = ids + [pad_id] * padding_length
                        attention_mask = attention_mask + [0] * padding_length
                    else:
                        ids = [pad_id] * padding_length + ids
                        attention_mask = [0] * padding_length + attention_mask
            
            encoded_texts.append(ids)
            attention_masks.append(attention_mask)
        
        # Format output
        if single_text:
            encoded_texts = encoded_texts[0]
            attention_masks = attention_masks[0]
        
        if return_tensors == "pt":
            encoded_texts = torch.tensor(encoded_texts, dtype=torch.long)
            if return_attention_mask:
                attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        elif return_tensors == "np":
            encoded_texts = np.array(encoded_texts, dtype=np.int64)
            if return_attention_mask:
                attention_masks = np.array(attention_masks, dtype=np.int64)
        
        # Build output dictionary
        if return_attention_mask or return_token_type_ids:
            output = {"input_ids": encoded_texts}
            if return_attention_mask:
                output["attention_mask"] = attention_masks
            if return_token_type_ids:
                # For single sequence, token type IDs are all 0
                if return_tensors == "pt":
                    token_type_ids = torch.zeros_like(encoded_texts)
                elif return_tensors == "np":
                    token_type_ids = np.zeros_like(encoded_texts)
                else:
                    token_type_ids = [0] * len(encoded_texts)
                output["token_type_ids"] = token_type_ids
            return output
        
        return encoded_texts
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor, np.ndarray],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = None,
    ) -> str:
        """Decode token IDs back to text"""
        # Convert to list if tensor
        if isinstance(token_ids, (torch.Tensor, np.ndarray)):
            token_ids = token_ids.tolist()
        
        # Handle batch decoding
        if isinstance(token_ids[0], list):
            return [self.decode(ids, skip_special_tokens, clean_up_tokenization_spaces) 
                   for ids in token_ids]
        
        # Filter special tokens if requested
        if skip_special_tokens:
            special_ids = set(self.config.special_tokens.values())
            token_ids = [tid for tid in token_ids if tid not in special_ids]
        
        # Decode based on tokenizer type
        if self.config.model_type == "sentencepiece" and self.sp_model:
            text = self.sp_model.decode(token_ids)
        elif self.config.model_type == "tiktoken" and self.tiktoken_enc:
            text = self.tiktoken_enc.decode(token_ids)
        else:
            # Fallback to reverse vocab
            pieces = [self.reverse_vocab.get(tid, '<unk>') for tid in token_ids]
            text = ''.join(pieces).replace('▁', ' ')
        
        # Clean up
        clean_up = clean_up_tokenization_spaces if clean_up_tokenization_spaces is not None else self.config.clean_up_tokenization_spaces
        if clean_up:
            text = self._clean_up_tokenization(text)
        
        return text
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before tokenization"""
        # Handle special financial patterns
        text = re.sub(r'\$([0-9,]+\.?\d*)', r'[CURRENCY] $ \1', text)
        text = re.sub(r'€([0-9,]+\.?\d*)', r'[CURRENCY] € \1', text)
        text = re.sub(r'₹([0-9,]+\.?\d*)', r'[CURRENCY] ₹ \1', text)
        text = re.sub(r'£([0-9,]+\.?\d*)', r'[CURRENCY] £ \1', text)
        text = re.sub(r'¥([0-9,]+\.?\d*)', r'[CURRENCY] ¥ \1', text)
        
        # Handle percentages
        text = re.sub(r'(\d+\.?\d*)%', r'[PERCENTAGE] \1 %', text)
        
        # Handle dates
        text = re.sub(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', r'[DATE] \1', text)
        text = re.sub(r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})', r'[DATE] \1', text)
        
        return text
    
    def _clean_up_tokenization(self, text: str) -> str:
        """Clean up tokenization artifacts"""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Fix punctuation spacing
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*(["\'])', r'\1\2', text)
        # Remove space before apostrophe in contractions
        text = re.sub(r"\s+'", "'", text)
        return text.strip()
    
    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Batch encode multiple texts efficiently"""
        return self.encode(
            texts,
            add_special_tokens=True,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_attention_mask=True,
        )
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer configuration and model"""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_file = save_dir / "tokenizer_config.json"
        config_dict = {
            "vocab_size": self.config.vocab_size,
            "model_type": self.config.model_type,
            "special_tokens": self.config.special_tokens,
            "max_length": self.config.max_length,
            "padding_side": self.config.padding_side,
            "truncation_side": self.config.truncation_side,
        }
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save SentencePiece model
        if self.sp_model:
            model_file = save_dir / "tokenizer.model"
            with open(model_file, 'wb') as f:
                f.write(self.sp_model.serialized_model_proto())
        
        # Save vocabulary
        vocab_file = save_dir / "vocab.json"
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f, indent=2)
        
        logger.info(f"Tokenizer saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, load_directory: str) -> "RythmTokenizer":
        """Load tokenizer from saved directory"""
        load_dir = Path(load_directory)
        
        # Load config
        config_file = load_dir / "tokenizer_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            config = TokenizerConfig(**config_dict)
        else:
            config = TokenizerConfig()
        
        # Create tokenizer
        tokenizer = cls(config)
        
        # Load SentencePiece model
        model_file = load_dir / "tokenizer.model"
        if model_file.exists():
            tokenizer.sp_model = spm.SentencePieceProcessor()
            tokenizer.sp_model.load(str(model_file))
        
        # Load vocabulary
        vocab_file = load_dir / "vocab.json"
        if vocab_file.exists():
            with open(vocab_file, 'r') as f:
                tokenizer.vocab = json.load(f)
            tokenizer.reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        
        logger.info(f"Tokenizer loaded from {load_directory}")
        return tokenizer
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.sp_model:
            return self.sp_model.get_piece_size()
        elif self.tiktoken_enc:
            return self.tiktoken_enc.n_vocab
        else:
            return len(self.vocab)
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert tokens to IDs"""
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.config.special_tokens["<unk>"])
        return [self.vocab.get(token, self.config.special_tokens["<unk>"]) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Convert IDs to tokens"""
        if isinstance(ids, int):
            return self.reverse_vocab.get(ids, "<unk>")
        return [self.reverse_vocab.get(id, "<unk>") for id in ids]
    
    def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, List[str]]]) -> int:
        """Add special tokens to vocabulary"""
        added_tokens = 0
        for key, value in special_tokens_dict.items():
            if isinstance(value, str):
                value = [value]
            for token in value:
                if token not in self.vocab:
                    new_id = len(self.vocab)
                    self.vocab[token] = new_id
                    self.reverse_vocab[new_id] = token
                    self.config.special_tokens[token] = new_id
                    added_tokens += 1
        
        if added_tokens > 0:
            logger.info(f"Added {added_tokens} special tokens")
        
        return added_tokens


class HuggingFaceTokenizerWrapper(PreTrainedTokenizerFast):
    """Wrapper to make RythmTokenizer compatible with HuggingFace transformers"""
    
    def __init__(self, tokenizer: RythmTokenizer, **kwargs):
        self.rythm_tokenizer = tokenizer
        super().__init__(
            tokenizer_object=None,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            **kwargs
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        ids = self.rythm_tokenizer.encode(text, add_special_tokens=False)
        return self.rythm_tokenizer.convert_ids_to_tokens(ids)
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID"""
        return self.rythm_tokenizer.convert_tokens_to_ids(token)
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token"""
        return self.rythm_tokenizer.convert_ids_to_tokens(index)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary"""
        return self.rythm_tokenizer.vocab
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save vocabulary"""
        self.rythm_tokenizer.save_pretrained(save_directory)
        return (str(Path(save_directory) / "vocab.json"),)


def create_tokenizer(vocab_size: int = 128000, model_type: str = "sentencepiece") -> RythmTokenizer:
    """Create and initialize a tokenizer"""
    config = TokenizerConfig(
        vocab_size=vocab_size,
        model_type=model_type
    )
    tokenizer = RythmTokenizer(config)
    logger.info(f"Created {model_type} tokenizer with vocab size {vocab_size}")
    return tokenizer


if __name__ == "__main__":
    print("=" * 80)
    print("RYTHM AI TOKENIZER SYSTEM")
    print("Advanced tokenization for financial AI")
    print("=" * 80)
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    
    # Test tokenization
    test_texts = [
        "Calculate my tax deduction for FY 2024-25",
        "My portfolio has $50,000 in stocks and €30,000 in bonds",
        "The P/E ratio is 15.5 and dividend yield is 3.2%",
        "[TAX] Section 80C allows ₹1,50,000 deduction",
    ]
    
    print("\nTokenization Tests:")
    for text in test_texts:
        encoded = tokenizer.encode(text, return_tensors="pt")
        decoded = tokenizer.decode(encoded)
        print(f"\nOriginal: {text}")
        print(f"Encoded: {encoded[:20]}..." if len(str(encoded)) > 20 else f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
    
    # Save tokenizer
    tokenizer.save_pretrained("./tokenizer")
    print("\nTokenizer saved to ./tokenizer")
    
    # Test loading
    loaded_tokenizer = RythmTokenizer.from_pretrained("./tokenizer")
    print("Tokenizer loaded successfully")
    
    print(f"\nVocabulary size: {loaded_tokenizer.get_vocab_size()}")
    print(f"Special tokens: {len(loaded_tokenizer.config.special_tokens)}")
