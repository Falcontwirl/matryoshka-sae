"""Quick sanity check for a trained Matryoshka SAE checkpoint.

Reports, per nested prefix level: fraction of variance explained by
reconstructing with only that many latents, plus overall L0 and dead-feature
count. This is what tells you whether the hierarchy actually formed (variance
explained should rise steeply on the smallest prefix, then rise more slowly
per level added).

Usage:
    python eval.py checkpoints/gpt2_small_layer6_matryoshka/ae_final.pt
"""

import argparse

import torch as t
from nnsight import LanguageModel

from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE

from config import Config
from data import stream_text


def main(checkpoint_path: str, n_batches: int):
    cfg = Config()
    device = cfg.device

    ae = MatryoshkaBatchTopKSAE.from_pretrained(checkpoint_path, device=device)
    ae.eval()
    print(
        f"Loaded SAE: activation_dim={ae.activation_dim}, dict_size={ae.dict_size}, "
        f"k={ae.k.item()}, group_sizes={ae.group_sizes.tolist()}"
    )

    model_dtype = t.bfloat16 if "cuda" in device else t.float32
    model = LanguageModel(cfg.model_name, device_map=device, dispatch=True, torch_dtype=model_dtype)
    submodule = model.transformer.h[cfg.layer]

    # Different seed than training so this isn't evaluated on the exact same shuffle order.
    text_gen = stream_text(cfg.dataset_name, cfg.dataset_text_field, seed=cfg.seed + 1)
    buffer = ActivationBuffer(
        data=text_gen,
        model=model,
        submodule=submodule,
        d_submodule=cfg.activation_dim,
        io="out",
        n_ctxs=200,
        ctx_len=cfg.ctx_len,
        refresh_batch_size=cfg.refresh_batch_size,
        out_batch_size=cfg.batch_size,
        device=device,
    )

    group_bounds = [0] + list(t.cumsum(ae.group_sizes, dim=0).tolist())
    n_levels = ae.active_groups

    l0_total = 0.0
    var_explained_by_level = [0.0] * n_levels

    with t.no_grad():
        for _, act in zip(range(n_batches), buffer):
            f, _active_F, _ = ae.encode(act, return_active=True, use_threshold=False)
            l0_total += (f != 0).float().sum(dim=-1).mean().item()

            x_hat = t.zeros_like(act) + ae.b_dec
            total_var = t.var(act.float(), dim=0).sum()
            for lvl in range(n_levels):
                lo, hi = group_bounds[lvl], group_bounds[lvl + 1]
                x_hat = x_hat + f[:, lo:hi] @ ae.W_dec[lo:hi]
                resid_var = t.var((act - x_hat).float(), dim=0).sum()
                var_explained_by_level[lvl] += (1 - resid_var / total_var).item()

    print(f"\nOver {n_batches} batches of {cfg.batch_size} tokens each:")
    print(f"  mean L0 (active latents/token): {l0_total / n_batches:.1f}")
    print("  fraction of variance explained, by nested prefix width:")
    for lvl in range(n_levels):
        width = group_bounds[lvl + 1]
        print(f"    first {width:>6} latents: {var_explained_by_level[lvl] / n_batches:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to an ae_*.pt checkpoint file")
    parser.add_argument("--n_batches", type=int, default=20)
    args = parser.parse_args()
    main(args.checkpoint, args.n_batches)
