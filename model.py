import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Single decoder block: masked self-attention + MLP."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual connection.
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out

        # Pre-norm feed-forward with residual connection.
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    """
    Minimal decoder-only transformer for next-token prediction.
    Input: token ids [batch, seq_len]
    Output: logits [batch, seq_len, vocab_size]
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Causal mask so position t can only attend to <= t.
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}.")

        pos = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(pos)[None, :, :]
        x = self.dropout(x)

        attn_mask = self.causal_mask[:seq_len, :seq_len]
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
