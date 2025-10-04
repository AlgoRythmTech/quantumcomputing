"""Decoder-only Transformer model with RoPE and modern attention."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency tensor
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('freqs', freqs)
        
        # Precompute positional encodings
        t = torch.arange(max_seq_len).float()
        freqs_outer = torch.outer(t, freqs)
        
        # Create complex exponential: e^(i * freq * pos)
        freqs_complex = torch.polar(torch.ones_like(freqs_outer), freqs_outer)
        self.register_buffer('freqs_complex', freqs_complex)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, dim)
            seq_len: Sequence length (uses x.size(-2) if None)
            
        Returns:
            Rotated tensor of same shape as input
        """
        if seq_len is None:
            seq_len = x.size(-2)
        
        # Reshape x to complex representation
        # x shape: (..., seq_len, dim) -> (..., seq_len, dim//2)
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_reshaped)
        
        # Get frequencies for current sequence length
        freqs = self.freqs_complex[:seq_len].to(x.device)
        
        # Apply rotation
        x_rotated = x_complex * freqs
        
        # Convert back to real representation
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.view(*x.shape)
        
        return x_out


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional FlashAttention support."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
        max_seq_len: int = 8192,
        rope_base: float = 10000.0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_flash_attention = use_flash_attention
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE
        self.rope = RoPEEmbedding(self.head_dim, max_seq_len, rope_base)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Attention mask of shape (batch_size, seq_len, seq_len)
            is_causal: Whether to use causal masking
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        k = self.k_proj(x)  # (batch_size, seq_len, d_model)
        v = self.v_proj(x)  # (batch_size, seq_len, d_model)
        
        # Reshape to multi-head format
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Apply RoPE to queries and keys
        q = self.rope(q)
        k = self.rope(k)
        
        # Transpose for attention computation: (batch_size, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.use_flash_attention:
            # Use PyTorch 2.0+ scaled_dot_product_attention (FlashAttention when available)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal and attention_mask is None
            )
        else:
            # Manual attention computation
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
            elif is_causal:
                # Create causal mask
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final output projection
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network with SwiGLU activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Gate projection for SwiGLU
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU activation."""
        # SwiGLU: swish(W1(x)) * W3(x)
        gate = F.silu(self.w1(x))  # SiLU/Swish activation
        x = gate * self.w3(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm formula: x / sqrt(mean(x^2) + eps) * weight
        norm = x.norm(dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return x / (norm + self.eps) * self.weight


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
        use_flash_attention: bool = True,
        max_seq_len: int = 8192,
        rope_base: float = 10000.0
    ):
        super().__init__()
        
        # Pre-norm architecture (like GPT-2, Llama)
        self.input_layernorm = RMSNorm(d_model, eps=layer_norm_eps)
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
            max_seq_len=max_seq_len,
            rope_base=rope_base
        )
        
        self.post_attention_layernorm = RMSNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """Forward pass of transformer block."""
        # Pre-norm attention
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x, attention_mask, is_causal)
        x = residual + x
        
        # Pre-norm feed-forward
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.feed_forward(x)
        x = residual + x
        
        return x


class GPTModel(nn.Module):
    """Decoder-only transformer language model (GPT-style)."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
        use_flash_attention: bool = True,
        rope_base: float = 10000.0,
        tie_word_embeddings: bool = True,
        gradient_checkpointing: bool = False,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        self.pad_token_id = pad_token_id
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                use_flash_attention=use_flash_attention,
                max_seq_len=max_seq_len,
                rope_base=rope_base
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(d_model, eps=layer_norm_eps)
        
        # Language modeling head
        if tie_word_embeddings:
            # Share weights with token embeddings
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(lambda module: self._init_weights(module, initializer_range))
    
    def _init_weights(self, module, initializer_range: float):
        """Initialize weights following Gemma-like initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> dict:
        """Forward pass of the model.
        
        Args:
            input_ids: Token ids of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            labels: Labels for language modeling loss (batch_size, seq_len)
            return_dict: Whether to return a dict or tuple
            
        Returns:
            Dictionary containing logits and loss (if labels provided)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        # Create causal attention mask if not provided
        if attention_mask is not None:
            # Convert padding mask to attention mask
            # attention_mask: 1 for tokens to attend to, 0 for padding
            extended_attention_mask = attention_mask[:, None, None, :].expand(
                batch_size, 1, seq_len, seq_len
            )
            # Create causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
            extended_attention_mask = extended_attention_mask * causal_mask
        else:
            extended_attention_mask = None
        
        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    extended_attention_mask,
                    True,  # is_causal
                    use_reentrant=False
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    is_causal=extended_attention_mask is None
                )
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Language modeling head
        if self.tie_word_embeddings:
            # Use tied weights (transpose of embedding matrix)
            logits = F.linear(hidden_states, self.token_embedding.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss computation
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if return_dict:
            return {
                'logits': logits,
                'loss': loss,
                'hidden_states': hidden_states,
            }
        else:
            return (logits, loss) if loss is not None else (logits,)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text using the model."""
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        
        self.eval()
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(generated, return_dict=True)
                next_token_logits = outputs['logits'][:, -1, :]  # Last token logits
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample or take argmax
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for EOS token
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return generated
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """Get number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params