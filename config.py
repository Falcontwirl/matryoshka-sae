"""All the knobs for the Matryoshka SAE run in one place."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- model / activations ---
    model_name: str = "gpt2"  # GPT-2 small, 124M, d_model=768
    layer: int = 6  # transformer block index (0-indexed, 12 blocks total) -> residual stream post-block
    ctx_len: int = 128

    # --- data ---
    # Streamed, effectively unbounded, same source distribution as the paper (The Pile).
    # For a fast local/dry-run smoke test, swap to "NeelNanda/pile-10k" (only 10k docs).
    dataset_name: str = "monology/pile-uncopyrighted"
    dataset_text_field: str = "text"

    # --- SAE architecture (MatryoshkaBatchTopK) ---
    activation_dim: int = 768  # GPT-2 small residual stream width
    dict_size: int = 6144  # 8x expansion
    # Cumulative nested widths: [192, 576, 1344, 2880, 6144]
    # (mirrors the paper's geometric-doubling-plus-remainder pattern, scaled down ~10x
    # from their real 65536-wide, 5-level Gemma-2-2B run)
    group_fractions: list = field(
        default_factory=lambda: [192 / 6144, 384 / 6144, 768 / 6144, 1536 / 6144, 3264 / 6144]
    )
    k: int = 32  # ~0.5% density at dict_size=6144, in line with the paper's k in [20, 320]

    # --- training ---
    target_tokens: int = 500_000_000  # ceiling to aim for; wall-clock cutoff below is the real limit
    batch_size: int = 2048  # matches the paper's SAE-step batch size
    lr: float = 3e-4  # matches the paper
    warmup_steps: int = 1000
    auxk_alpha: float = 1 / 32
    seed: int = 0

    # --- activation buffer (controls LM-forward-pass batching, not the SAE step) ---
    n_ctxs: int = 5_000  # buffer holds n_ctxs * ctx_len activations (~640k) before needing a refill
    refresh_batch_size: int = 256  # texts per GPT-2 forward pass when refilling the buffer

    # --- safety / infra ---
    max_train_hours: float = 4.5  # hard wall-clock cap; leaves ~30min headroom under a 5h budget
    checkpoint_every_minutes: float = 15.0
    log_every: int = 50
    save_dir: str = "./checkpoints/gpt2_small_layer6_matryoshka"
    device: str = "cuda"
