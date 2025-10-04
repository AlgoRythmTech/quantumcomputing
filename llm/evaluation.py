"""Evaluation metrics for language models."""

import os
import math
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import GPTModel
from .tokenizer import SPTokenizer
from .data import DataModule


def compute_perplexity(
    model: GPTModel,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs['loss']
            
            # Count non-padded tokens
            attention_mask = batch['attention_mask']
            num_tokens = attention_mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity


def compute_bleu_score(
    predictions: List[str],
    references: List[str],
    max_order: int = 4
) -> Dict[str, float]:
    """Compute BLEU score using sacrebleu."""
    try:
        import sacrebleu
    except ImportError:
        raise ImportError("sacrebleu is required for BLEU score computation. Install with: pip install sacrebleu")
    
    # Convert single references to list format if needed
    if isinstance(references[0], str):
        references = [[ref] for ref in references]
    
    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(predictions, references)
    
    # Compute individual n-gram scores
    scores = {
        'bleu': bleu.score,
        'bleu_1': 0.0,
        'bleu_2': 0.0,
        'bleu_3': 0.0,
        'bleu_4': 0.0,
        'bp': bleu.bp,
        'ratio': bleu.ratio,
        'hyp_len': bleu.sys_len,
        'ref_len': bleu.ref_len,
    }
    
    # Compute individual n-gram BLEU scores
    for n in range(1, min(max_order + 1, 5)):
        bleu_n = sacrebleu.corpus_bleu(
            predictions, 
            references,
            max_ngram_order=n
        )
        scores[f'bleu_{n}'] = bleu_n.score
    
    return scores


def compute_rouge_score(
    predictions: List[str],
    references: List[str],
    rouge_types: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE score using rouge-score."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("rouge-score is required for ROUGE computation. Install with: pip install rouge-score")
    
    if rouge_types is None:
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
    
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    scores = {rouge_type: {'precision': [], 'recall': [], 'fmeasure': []} 
              for rouge_type in rouge_types}
    
    for pred, ref in zip(predictions, references):
        rouge_scores = scorer.score(ref, pred)
        
        for rouge_type in rouge_types:
            scores[rouge_type]['precision'].append(rouge_scores[rouge_type].precision)
            scores[rouge_type]['recall'].append(rouge_scores[rouge_type].recall)
            scores[rouge_type]['fmeasure'].append(rouge_scores[rouge_type].fmeasure)
    
    # Compute averages
    avg_scores = {}
    for rouge_type in rouge_types:
        avg_scores[rouge_type] = {
            'precision': sum(scores[rouge_type]['precision']) / len(scores[rouge_type]['precision']),
            'recall': sum(scores[rouge_type]['recall']) / len(scores[rouge_type]['recall']),
            'fmeasure': sum(scores[rouge_type]['fmeasure']) / len(scores[rouge_type]['fmeasure']),
        }
    
    return avg_scores


def generate_texts(
    model: GPTModel,
    tokenizer: SPTokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    do_sample: bool = True,
    device: torch.device = torch.device('cpu')
) -> List[str]:
    """Generate texts from prompts."""
    model.eval()
    generated_texts = []
    
    with torch.no_grad():
        for prompt in prompts:
            # Encode prompt
            input_ids = tokenizer.encode(prompt, add_bos=True)
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            
            # Generate
            generated_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode only the generated part
            prompt_length = input_ids.size(1)
            generated_part = generated_ids[0, prompt_length:].tolist()
            generated_text = tokenizer.decode(generated_part, skip_special_tokens=True)
            
            generated_texts.append(generated_text)
    
    return generated_texts


class Evaluator:
    """Model evaluator with multiple metrics."""
    
    def __init__(
        self,
        model: GPTModel,
        tokenizer: SPTokenizer,
        device: torch.device = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def evaluate_perplexity(self, dataloader: DataLoader) -> float:
        """Evaluate perplexity."""
        return compute_perplexity(self.model, dataloader, self.device)
    
    def evaluate_generation(
        self,
        prompts: List[str],
        references: List[str],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        compute_bleu: bool = True,
        compute_rouge: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate text generation quality."""
        # Generate texts
        generated_texts = generate_texts(
            self.model,
            self.tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            device=self.device
        )
        
        results = {
            'generated_texts': generated_texts,
            'references': references,
            'prompts': prompts
        }
        
        # Compute BLEU score
        if compute_bleu:
            try:
                bleu_scores = compute_bleu_score(generated_texts, references)
                results['bleu'] = bleu_scores
            except Exception as e:
                print(f"Error computing BLEU: {e}")
                results['bleu'] = None
        
        # Compute ROUGE score
        if compute_rouge:
            try:
                rouge_scores = compute_rouge_score(generated_texts, references)
                results['rouge'] = rouge_scores
            except Exception as e:
                print(f"Error computing ROUGE: {e}")
                results['rouge'] = None
        
        return results
    
    def evaluate_full(
        self,
        dataloader: DataLoader,
        prompts: Optional[List[str]] = None,
        references: Optional[List[str]] = None,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """Run full evaluation with all metrics."""
        results = {}
        
        # Compute perplexity
        print("Computing perplexity...")
        results['perplexity'] = self.evaluate_perplexity(dataloader)
        
        # Compute generation metrics if prompts and references provided
        if prompts and references:
            print("Evaluating text generation...")
            generation_results = self.evaluate_generation(
                prompts, references, **generation_kwargs
            )
            results.update(generation_results)
        
        return results


def load_test_data_from_file(file_path: str) -> Tuple[List[str], List[str]]:
    """Load test prompts and references from a JSON file.
    
    Expected format:
    [
        {"prompt": "...", "reference": "..."},
        {"prompt": "...", "reference": "..."},
        ...
    ]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = [item['prompt'] for item in data]
    references = [item['reference'] for item in data]
    
    return prompts, references


def evaluate_model(
    model_path: str,
    tokenizer_path: str,
    test_dataset_paths: List[str],
    batch_size: int = 16,
    max_seq_len: int = 4096,
    output_dir: str = "artifacts/evaluation",
    generation_prompts_file: Optional[str] = None,
    compute_perplexity: bool = True,
    compute_bleu: bool = False,
    compute_rouge: bool = False,
    **generation_kwargs
) -> Dict[str, Any]:
    """Evaluate a trained model."""
    # Load model and tokenizer
    print(f"Loading model from {model_path}")
    tokenizer = SPTokenizer.from_pretrained(tokenizer_path)
    
    # Load model checkpoint
    if os.path.isdir(model_path):
        checkpoint_file = os.path.join(model_path, 'pytorch_model.bin')
    else:
        checkpoint_file = model_path
    
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model_config = checkpoint.get('config', {}).get('model', {})
    
    # Create model
    model = GPTModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=model_config.get('d_model', 768),
        n_layers=model_config.get('n_layers', 12),
        n_heads=model_config.get('n_heads', 12),
        d_ff=model_config.get('d_ff', 3072),
        max_seq_len=model_config.get('max_seq_len', 4096),
        dropout=model_config.get('dropout', 0.1),
        layer_norm_eps=model_config.get('layer_norm_eps', 1e-6),
        use_flash_attention=model_config.get('use_flash_attention', True),
        rope_base=model_config.get('rope_base', 10000.0),
        tie_word_embeddings=model_config.get('tie_word_embeddings', True),
        gradient_checkpointing=False,  # Disable for evaluation
        initializer_range=model_config.get('initializer_range', 0.02),
        pad_token_id=tokenizer.pad_token_id,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = Evaluator(model, tokenizer)
    
    results = {}
    
    # Evaluate perplexity on test datasets
    if compute_perplexity and test_dataset_paths:
        data_module = DataModule(
            tokenizer=tokenizer,
            train_dataset_paths=[],  # No training data needed
            eval_dataset_paths=test_dataset_paths,
            max_seq_len=max_seq_len,
            pack_sequences=True,
        )
        
        test_dataloader = data_module.get_eval_dataloader(
            batch_size=batch_size,
            num_workers=0,  # Simpler for evaluation
            pin_memory=False,
            distributed=False,
        )
        
        if test_dataloader is not None:
            perplexity = evaluator.evaluate_perplexity(test_dataloader)
            results['perplexity'] = perplexity
            print(f"Perplexity: {perplexity:.4f}")
    
    # Evaluate generation metrics
    if (compute_bleu or compute_rouge) and generation_prompts_file:
        if os.path.exists(generation_prompts_file):
            prompts, references = load_test_data_from_file(generation_prompts_file)
            
            generation_results = evaluator.evaluate_generation(
                prompts=prompts,
                references=references,
                compute_bleu=compute_bleu,
                compute_rouge=compute_rouge,
                **generation_kwargs
            )
            
            results.update(generation_results)
            
            if 'bleu' in results and results['bleu']:
                print(f"BLEU score: {results['bleu']['bleu']:.4f}")
            
            if 'rouge' in results and results['rouge']:
                for rouge_type, scores in results['rouge'].items():
                    print(f"{rouge_type.upper()} F1: {scores['fmeasure']:.4f}")
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        # Convert any non-serializable objects to strings
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                serializable_results[k] = v
            else:
                serializable_results[k] = str(v)
        
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation results saved to {output_dir / 'evaluation_results.json'}")
    
    return results


def run_benchmark(
    model_path: str,
    tokenizer_path: str,
    benchmark_name: str = "custom",
    output_dir: str = "artifacts/benchmark"
) -> Dict[str, Any]:
    """Run standardized benchmarks."""
    # This is a placeholder for more comprehensive benchmarking
    # You could integrate with common benchmarks like:
    # - GLUE/SuperGLUE
    # - HellaSwag
    # - ARC
    # - etc.
    
    print(f"Running {benchmark_name} benchmark...")
    
    # For now, just run basic evaluation
    results = {
        'benchmark_name': benchmark_name,
        'model_path': model_path,
        'status': 'completed'
    }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f'{benchmark_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results