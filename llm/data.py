"""Data loading and processing pipeline for language model training."""

import os
import glob
import random
from typing import List, Optional, Iterator, Dict, Any, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist

from .tokenizer import SPTokenizer


class TextLineDataset(Dataset):
    """Dataset that reads text files line by line."""
    
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: SPTokenizer,
        max_seq_len: int = 4096,
        min_seq_len: int = 10,
        pack_sequences: bool = True,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        """
        Args:
            file_paths: List of text files to read
            tokenizer: Tokenizer to use for encoding
            max_seq_len: Maximum sequence length
            min_seq_len: Minimum sequence length (filter shorter sequences)
            pack_sequences: Whether to pack multiple sequences into one
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
        """
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.pack_sequences = pack_sequences
        self.add_bos = add_bos
        self.add_eos = add_eos
        
        # Load and prepare data
        self.examples = self._load_examples()
        
        print(f"Loaded {len(self.examples)} examples from {len(file_paths)} files")
    
    def _load_examples(self) -> List[str]:
        """Load text examples from files."""
        examples = []
        
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if len(line) >= self.min_seq_len:
                            examples.append(line)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        text = self.examples[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            add_bos=self.add_bos,
            add_eos=self.add_eos,
            max_length=self.max_seq_len
        )
        
        # Pad to max length
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(tokens))
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # Labels are the same as input_ids for causal language modeling
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class PackedTextDataset(Dataset):
    """Dataset that packs multiple short sequences into longer ones."""
    
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: SPTokenizer,
        max_seq_len: int = 4096,
        min_seq_len: int = 10,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.add_bos = add_bos
        self.add_eos = add_eos
        
        # Load and pack examples
        self.packed_examples = self._load_and_pack_examples()
        
        print(f"Packed into {len(self.packed_examples)} examples of max length {max_seq_len}")
    
    def _load_and_pack_examples(self) -> List[List[int]]:
        """Load text and pack into fixed-length sequences."""
        all_tokens = []
        
        # First, collect all tokens
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    
                # Tokenize entire file
                tokens = self.tokenizer.encode(text, add_bos=False, add_eos=False)
                all_tokens.extend(tokens)
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        # Pack tokens into sequences of max_seq_len
        packed_examples = []
        current_seq = []
        
        for token in all_tokens:
            current_seq.append(token)
            
            if len(current_seq) >= self.max_seq_len:
                # Add BOS/EOS if requested
                if self.add_bos:
                    current_seq = [self.tokenizer.bos_token_id] + current_seq[:-1]
                if self.add_eos:
                    current_seq = current_seq[:-1] + [self.tokenizer.eos_token_id]
                
                packed_examples.append(current_seq[:self.max_seq_len])
                current_seq = []
        
        # Handle remaining tokens
        if len(current_seq) >= self.min_seq_len:
            # Pad to max_seq_len
            if self.add_bos:
                current_seq = [self.tokenizer.bos_token_id] + current_seq
            if self.add_eos:
                current_seq = current_seq + [self.tokenizer.eos_token_id]
            
            # Pad or truncate
            if len(current_seq) < self.max_seq_len:
                current_seq = current_seq + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(current_seq))
            else:
                current_seq = current_seq[:self.max_seq_len]
            
            packed_examples.append(current_seq)
        
        return packed_examples
    
    def __len__(self) -> int:
        return len(self.packed_examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single packed example."""
        tokens = self.packed_examples[idx]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # Labels are the same as input_ids for causal language modeling
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class StreamingTextDataset:
    """Streaming dataset for very large text corpora."""
    
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: SPTokenizer,
        max_seq_len: int = 4096,
        min_seq_len: int = 10,
        pack_sequences: bool = True,
        add_bos: bool = True,
        add_eos: bool = True,
        shuffle_files: bool = True,
        buffer_size: int = 10000,
    ):
        self.file_paths = file_paths if not shuffle_files else random.sample(file_paths, len(file_paths))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.pack_sequences = pack_sequences
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.buffer_size = buffer_size
        
        print(f"Streaming from {len(file_paths)} files")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over examples."""
        buffer = []
        current_tokens = []
        
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if len(line) < self.min_seq_len:
                            continue
                        
                        if self.pack_sequences:
                            # Pack multiple lines into sequences
                            tokens = self.tokenizer.encode(line, add_bos=False, add_eos=False)
                            current_tokens.extend(tokens)
                            
                            while len(current_tokens) >= self.max_seq_len:
                                seq = current_tokens[:self.max_seq_len]
                                current_tokens = current_tokens[self.max_seq_len:]
                                
                                if self.add_bos:
                                    seq = [self.tokenizer.bos_token_id] + seq[:-1]
                                if self.add_eos:
                                    seq = seq[:-1] + [self.tokenizer.eos_token_id]
                                
                                example = self._create_example(seq)
                                buffer.append(example)
                                
                                if len(buffer) >= self.buffer_size:
                                    random.shuffle(buffer)
                                    yield from buffer
                                    buffer = []
                        else:
                            # Use individual lines
                            tokens = self.tokenizer.encode(
                                line,
                                add_bos=self.add_bos,
                                add_eos=self.add_eos,
                                max_length=self.max_seq_len
                            )
                            
                            if len(tokens) < self.max_seq_len:
                                tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(tokens))
                            
                            example = self._create_example(tokens)
                            buffer.append(example)
                            
                            if len(buffer) >= self.buffer_size:
                                random.shuffle(buffer)
                                yield from buffer
                                buffer = []
            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        # Yield remaining examples
        if buffer:
            random.shuffle(buffer)
            yield from buffer
        
        # Handle remaining packed tokens
        if self.pack_sequences and len(current_tokens) >= self.min_seq_len:
            if len(current_tokens) < self.max_seq_len:
                current_tokens = current_tokens + [self.tokenizer.pad_token_id] * (
                    self.max_seq_len - len(current_tokens)
                )
            
            if self.add_bos:
                current_tokens = [self.tokenizer.bos_token_id] + current_tokens[:-1]
            if self.add_eos:
                current_tokens = current_tokens[:-1] + [self.tokenizer.eos_token_id]
            
            example = self._create_example(current_tokens[:self.max_seq_len])
            yield example
    
    def _create_example(self, tokens: List[int]) -> Dict[str, torch.Tensor]:
        """Create a training example from tokens."""
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def get_dataset_files(dataset_paths: List[str], extensions: List[str] = None) -> List[str]:
    """Get all files from dataset paths.
    
    Args:
        dataset_paths: List of file paths, directories, or glob patterns
        extensions: File extensions to include (e.g., ['.txt', '.json'])
        
    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = ['.txt', '.json', '.jsonl']
    
    all_files = []
    
    for path in dataset_paths:
        path = Path(path)
        
        if path.is_file():
            all_files.append(str(path))
        elif path.is_dir():
            # Recursively find files with specified extensions
            for ext in extensions:
                pattern = f"**/*{ext}"
                files = list(path.glob(pattern))
                all_files.extend([str(f) for f in files])
        else:
            # Treat as glob pattern
            files = glob.glob(str(path))
            all_files.extend(files)
    
    # Remove duplicates and sort
    all_files = sorted(list(set(all_files)))
    
    print(f"Found {len(all_files)} files")
    return all_files


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
) -> DataLoader:
    """Create a DataLoader with optional distributed support."""
    
    sampler = None
    if distributed:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Distributed training is not available")
        
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=True  # Drop last incomplete batch for distributed training
        )
        shuffle = False  # Sampler handles shuffling
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # For stable training
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


class DataModule:
    """Data module for managing train/eval datasets."""
    
    def __init__(
        self,
        tokenizer: SPTokenizer,
        train_dataset_paths: List[str],
        eval_dataset_paths: Optional[List[str]] = None,
        max_seq_len: int = 4096,
        min_seq_len: int = 10,
        pack_sequences: bool = True,
        use_streaming: bool = False,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.tokenizer = tokenizer
        self.train_dataset_paths = train_dataset_paths
        self.eval_dataset_paths = eval_dataset_paths or []
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.pack_sequences = pack_sequences
        self.use_streaming = use_streaming
        self.add_bos = add_bos
        self.add_eos = add_eos
        
        # Get file lists
        self.train_files = get_dataset_files(train_dataset_paths)
        self.eval_files = get_dataset_files(eval_dataset_paths) if eval_dataset_paths else []
        
        print(f"Training files: {len(self.train_files)}")
        print(f"Evaluation files: {len(self.eval_files)}")
    
    def get_train_dataset(self) -> Union[Dataset, StreamingTextDataset]:
        """Get training dataset."""
        if self.use_streaming:
            return StreamingTextDataset(
                file_paths=self.train_files,
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
                min_seq_len=self.min_seq_len,
                pack_sequences=self.pack_sequences,
                add_bos=self.add_bos,
                add_eos=self.add_eos,
            )
        elif self.pack_sequences:
            return PackedTextDataset(
                file_paths=self.train_files,
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
                min_seq_len=self.min_seq_len,
                add_bos=self.add_bos,
                add_eos=self.add_eos,
            )
        else:
            return TextLineDataset(
                file_paths=self.train_files,
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
                min_seq_len=self.min_seq_len,
                pack_sequences=False,
                add_bos=self.add_bos,
                add_eos=self.add_eos,
            )
    
    def get_eval_dataset(self) -> Optional[Dataset]:
        """Get evaluation dataset."""
        if not self.eval_files:
            return None
        
        if self.pack_sequences:
            return PackedTextDataset(
                file_paths=self.eval_files,
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
                min_seq_len=self.min_seq_len,
                add_bos=self.add_bos,
                add_eos=self.add_eos,
            )
        else:
            return TextLineDataset(
                file_paths=self.eval_files,
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
                min_seq_len=self.min_seq_len,
                pack_sequences=False,
                add_bos=self.add_bos,
                add_eos=self.add_eos,
            )
    
    def get_train_dataloader(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        distributed: bool = False,
    ) -> DataLoader:
        """Get training dataloader."""
        dataset = self.get_train_dataset()
        
        if self.use_streaming:
            # Streaming dataset doesn't support standard DataLoader
            # You would need to implement custom batching logic
            raise NotImplementedError("Streaming datasets need custom batching")
        
        return create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            distributed=distributed,
        )
    
    def get_eval_dataloader(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        distributed: bool = False,
    ) -> Optional[DataLoader]:
        """Get evaluation dataloader."""
        dataset = self.get_eval_dataset()
        if dataset is None:
            return None
        
        return create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            distributed=distributed,
        )