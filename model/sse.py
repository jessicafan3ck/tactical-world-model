"""
Spatial Set Encoder (SSE)
=========================
Set Transformer that encodes an unordered set of 22 player freeze-frame
positions into a fixed-size latent tactical representation z.

Ported from the "Beyond Actions" analytics project with three upgrades
for the World Model:
  1. Larger capacity: d_model=128, z_dim=256
  2. AttentionSSE subclass returns per-head attention weights for
     visualization and the causal intervention engine
  3. TeamEncoder: aggregates possession-level z embeddings into a
     stable team fingerprint for conditioning the flow matching model

Permutation invariance guarantee: CLS-token aggregation + TransformerEncoder
are invariant to the ordering of the 22 player tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSetEncoder(nn.Module):
    """
    Encodes an unordered player set into a latent tactical state z.

    Input per player: [x_norm, y_norm, is_teammate, is_actor]
      x_norm = x / 120,  y_norm = y / 80

    Args:
        input_dim  : per-player feature dimension (default 4)
        d_model    : transformer hidden dimension
        nhead      : number of attention heads
        num_layers : transformer encoder depth
        z_dim      : output embedding dimension
        dropout    : attention + FFN dropout
    """

    def __init__(self,
                 input_dim:  int   = 4,
                 d_model:    int   = 128,
                 nhead:      int   = 4,
                 num_layers: int   = 3,
                 z_dim:      int   = 256,
                 dropout:    float = 0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.z_proj = nn.Sequential(
            nn.Linear(d_model, z_dim),
            nn.LayerNorm(z_dim),
        )

    def forward(self,
                positions: torch.Tensor,
                mask:      torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions : (B, P, input_dim)
            mask      : (B, P) bool — True where padded

        Returns:
            z : (B, z_dim)
        """
        B = positions.shape[0]
        h = self.input_proj(positions)

        cls       = self.cls_token.expand(B, -1, -1)
        h         = torch.cat([cls, h], dim=1)
        cls_mask  = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)

        h = self.transformer(h, src_key_padding_mask=full_mask)
        return self.z_proj(h[:, 0, :])


class AttentionSSE(SpatialSetEncoder):
    """
    SpatialSetEncoder variant that also returns per-head attention weights
    from the first transformer layer.  Used for attention visualization
    and the causal intervention engine.
    """

    def forward_with_attention(self,
                                positions: torch.Tensor,
                                mask:      torch.Tensor
                                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z    : (B, z_dim)
            attn : (B, nhead, 1+P, 1+P)  — attention from layer 0
        """
        B = positions.shape[0]
        h = self.input_proj(positions)

        cls       = self.cls_token.expand(B, -1, -1)
        h         = torch.cat([cls, h], dim=1)
        cls_mask  = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)

        # Run first layer manually to capture attention weights
        layer0 = self.transformer.layers[0]
        h_norm = layer0.norm1(h)
        attn_out, attn_weights = layer0.self_attn(
            h_norm, h_norm, h_norm,
            key_padding_mask=full_mask,
            need_weights=True,
            average_attn_weights=False,   # per-head
        )
        h = h + layer0.dropout1(attn_out)
        h = h + layer0.dropout2(layer0.linear2(
            layer0.dropout(layer0.activation(layer0.linear1(layer0.norm2(h))))
        ))

        # Remaining layers
        for layer in self.transformer.layers[1:]:
            h = layer(h, src_key_padding_mask=full_mask)

        z = self.z_proj(h[:, 0, :])
        return z, attn_weights   # attn_weights: (B, nhead, seq, seq)


class TacticalPredictor(nn.Module):
    """
    Full prediction model: player set → z → possession outcome logits.

    Context (entry_state, phase) concatenated to z before the head so the
    encoder learns pure spatial structure independent of game context.
    """

    def __init__(self,
                 input_dim:   int   = 4,
                 d_model:     int   = 128,
                 nhead:       int   = 4,
                 num_layers:  int   = 3,
                 z_dim:       int   = 256,
                 context_dim: int   = 3,
                 n_targets:   int   = 3,
                 dropout:     float = 0.1):
        super().__init__()

        self.encoder = SpatialSetEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            z_dim=z_dim,
            dropout=dropout,
        )

        head_in = z_dim + context_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_in // 2, n_targets),
        )

    def forward(self,
                positions: torch.Tensor,
                mask:      torch.Tensor,
                context:   torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z      : (B, z_dim)
            logits : (B, n_targets)
        """
        z      = self.encoder(positions, mask)
        logits = self.head(torch.cat([z, context], dim=-1))
        return z, logits


class TeamEncoder(nn.Module):
    """
    Aggregates possession-level z embeddings into a stable team fingerprint.

    During training, run SSE over all possessions for a team and mean-pool
    the resulting z vectors. The TeamEncoder optionally learns a light
    refinement MLP on top of the mean-pooled z.

    Used as the conditioning signal for the flow matching generator.
    """

    def __init__(self, z_dim: int = 256, fingerprint_dim: int = 256):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Linear(z_dim, fingerprint_dim),
            nn.GELU(),
            nn.LayerNorm(fingerprint_dim),
        )

    def forward(self, z_stack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_stack : (N_possessions, z_dim)  — all z's for one team

        Returns:
            fingerprint : (fingerprint_dim,)
        """
        mean_z = z_stack.mean(dim=0)
        return self.refine(mean_z)


# ── Losses ────────────────────────────────────────────────────────────────────

def masked_bce_loss(logits: torch.Tensor,
                    targets: torch.Tensor) -> torch.Tensor:
    """BCE loss ignoring NaN targets (start-zone filtered)."""
    valid = ~torch.isnan(targets)
    if valid.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)
    return F.binary_cross_entropy_with_logits(
        logits[valid], targets[valid], reduction="mean"
    )


# ── Factories ─────────────────────────────────────────────────────────────────

def build_predictor(z_dim: int = 256) -> TacticalPredictor:
    return TacticalPredictor(
        input_dim=4, d_model=128, nhead=4, num_layers=3,
        z_dim=z_dim, context_dim=3, n_targets=3, dropout=0.1,
    )


def build_attention_encoder(z_dim: int = 256) -> AttentionSSE:
    return AttentionSSE(
        input_dim=4, d_model=128, nhead=4, num_layers=3,
        z_dim=z_dim, dropout=0.1,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = build_predictor()
    print(f"TacticalPredictor parameters: {count_parameters(model):,}")

    B, P = 4, 23
    pos  = torch.randn(B, P, 4)
    mask = torch.zeros(B, P, dtype=torch.bool)
    mask[:, 20:] = True
    ctx  = torch.randn(B, 3)

    z, logits = model(pos, mask, ctx)
    print(f"z: {z.shape}  logits: {logits.shape}")

    model.eval()
    with torch.no_grad():
        z1, _ = model(pos, mask, ctx)
        idx   = torch.randperm(P)
        z2, _ = model(pos[:, idx], mask[:, idx], ctx)
    print(f"Permutation invariance max diff: {(z1 - z2).abs().max():.2e}")
