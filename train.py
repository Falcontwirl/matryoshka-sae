"""Train a Matryoshka BatchTopK SAE on GPT-2 small residual-stream activations.

Uses the real dictionary_learning MatryoshkaBatchTopKTrainer / ActivationBuffer
(not a reimplementation), with an added hard wall-clock cutoff so a cloud GPU
rental never runs past a fixed budget regardless of throughput.

Usage:
    python train.py                  # full run per config.py
    python train.py --smoke_test     # ~3 min dry run to validate the pipeline
"""

import argparse
import signal
import time
from pathlib import Path

import torch as t
from nnsight import LanguageModel

from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKTrainer

from config import Config
from data import stream_text

_STOP = False


def _handle_stop(signum, frame):
    global _STOP
    print(f"\nReceived signal {signum}; will stop after this step and save a checkpoint.")
    _STOP = True


def save_checkpoint(trainer, save_dir: Path, step: int, tokens_seen: int, tag: str):
    ae_path = save_dir / f"ae_{tag}.pt"
    t.save({k: v.cpu() for k, v in trainer.ae.state_dict().items()}, ae_path)
    t.save(
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "optimizer": trainer.optimizer.state_dict(),
            "config": trainer.config,
        },
        save_dir / f"trainer_state_{tag}.pt",
    )
    print(f"  [checkpoint] step {step} ({tokens_seen:,} tokens) -> {ae_path}")


def main(cfg: Config):
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    t.manual_seed(cfg.seed)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = "cuda" in cfg.device
    model_dtype = t.bfloat16 if use_cuda else t.float32

    print(f"Loading {cfg.model_name} via nnsight (dtype={model_dtype})...")
    model = LanguageModel(cfg.model_name, device_map=cfg.device, dispatch=True, torch_dtype=model_dtype)
    submodule = model.transformer.h[cfg.layer]

    text_gen = stream_text(cfg.dataset_name, cfg.dataset_text_field, seed=cfg.seed)

    print(f"Setting up activation buffer (dataset={cfg.dataset_name})...")
    buffer = ActivationBuffer(
        data=text_gen,
        model=model,
        submodule=submodule,
        d_submodule=cfg.activation_dim,
        io="out",
        n_ctxs=cfg.n_ctxs,
        ctx_len=cfg.ctx_len,
        refresh_batch_size=cfg.refresh_batch_size,
        out_batch_size=cfg.batch_size,
        device=cfg.device,
    )

    est_steps = max(cfg.target_tokens // cfg.batch_size, cfg.warmup_steps + 1)

    trainer = MatryoshkaBatchTopKTrainer(
        steps=est_steps,
        activation_dim=cfg.activation_dim,
        dict_size=cfg.dict_size,
        k=cfg.k,
        layer=cfg.layer,
        lm_name=cfg.model_name,
        group_fractions=cfg.group_fractions,
        lr=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        auxk_alpha=cfg.auxk_alpha,
        seed=cfg.seed,
        device=cfg.device,
        wandb_name="matryoshka_gpt2_small",
        submodule_name=f"blocks.{cfg.layer}.resid_post",
    )
    print("Trainer config:", trainer.config)

    start_time = time.time()
    deadline = start_time + cfg.max_train_hours * 3600
    last_checkpoint_time = start_time
    tokens_seen = 0
    step = 0

    autocast_ctx = t.autocast(device_type="cuda", dtype=t.bfloat16) if use_cuda else None

    print(
        f"Training for up to {cfg.max_train_hours}h (hard cutoff) or "
        f"{cfg.target_tokens:,} tokens, whichever comes first."
    )

    try:
        for act in buffer:
            now = time.time()
            if now >= deadline:
                print(f"Hit {cfg.max_train_hours}h wall-clock cap at step {step}. Stopping.")
                break
            if tokens_seen >= cfg.target_tokens:
                print(f"Reached target of {cfg.target_tokens:,} tokens at step {step}. Stopping.")
                break
            if _STOP:
                print("Stop signal received. Stopping.")
                break

            if autocast_ctx is not None:
                with autocast_ctx:
                    trainer.update(step, act)
            else:
                trainer.update(step, act)

            tokens_seen += act.shape[0]

            if step % cfg.log_every == 0:
                elapsed = now - start_time
                with t.no_grad():
                    logs = trainer.loss(act.to(cfg.device), step=step, logging=True)
                l0 = (logs.f != 0).float().sum(dim=-1).mean().item()
                msg = (
                    f"step {step:>7} | {tokens_seen / 1e6:8.2f}M tok | {elapsed / 60:6.1f} min | "
                    f"loss {logs.losses['loss']:.4f} | l2 {logs.losses['l2_loss']:.4f} | "
                    f"auxk {logs.losses['auxk_loss']:.4f} | L0 {l0:.1f} | dead {trainer.dead_features}"
                )
                if elapsed > 0 and step > 0:
                    tok_per_sec = tokens_seen / elapsed
                    eta_tokens = tokens_seen + tok_per_sec * max(deadline - now, 0)
                    msg += f" | ~{tok_per_sec:,.0f} tok/s | projected total @ deadline ~{eta_tokens / 1e6:,.0f}M tok"
                print(msg)

            if now - last_checkpoint_time >= cfg.checkpoint_every_minutes * 60:
                save_checkpoint(trainer, save_dir, step, tokens_seen, tag="latest")
                last_checkpoint_time = now

            step += 1
    finally:
        save_checkpoint(trainer, save_dir, step, tokens_seen, tag="final")
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="~3 minute dry run (tiny token budget + time cap) to validate the pipeline before committing GPU hours.",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.smoke_test:
        cfg.target_tokens = 200_000
        cfg.max_train_hours = 0.05  # 3 minutes
        cfg.n_ctxs = 500
        cfg.log_every = 5
        cfg.checkpoint_every_minutes = 1.0
        cfg.save_dir = "./checkpoints/smoke_test"
        print("Running in --smoke_test mode: tiny budget, just validating the pipeline runs end to end.")

    main(cfg)
