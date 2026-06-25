
"""
trainer.py
----------
All training and evaluation loop functions for the three-stage FER pipeline.

  train_stage1        – expression-encoder pre-training via MINE
  train_stage2        – identity-encoder training with adversarial loss
  train_stage3        – FER classifier fine-tuning on frozen expression embeds
  evaluate_classifier – accuracy on a val DataLoader
  compute_mig         – Mutual Information Gap metric
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import (
    ExpressionEncoder,
    IdentityEncoder,
    FERClassifier,
    Discriminator,
    Stage1Loss,
    Stage2MILoss,
    StableMINE,
    PairStatisticsNetwork,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _shuffle(x: torch.Tensor) -> torch.Tensor:
    """Return x with rows permuted randomly (in-place safe)."""
    idx = torch.randperm(x.size(0), device=x.device)
    return x[idx]


def _discriminator_loss(disc: Discriminator, e: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    i_shuffled = _shuffle(i)
    out_pos = disc(e, i)
    out_neg = disc(e, i_shuffled)
    return (
        F.binary_cross_entropy_with_logits(out_pos, torch.zeros_like(out_pos))
        + F.binary_cross_entropy_with_logits(out_neg, torch.ones_like(out_neg))
    )


def _encoder_adversarial_loss(disc: Discriminator, e: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    """Generator-style loss: fool the discriminator into labelling joint pairs as real."""
    fake_logits = disc(e, i)
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


@torch.no_grad()
def _discriminator_accuracy(disc: Discriminator, e: torch.Tensor, i: torch.Tensor) -> float:
    i_shuffled = _shuffle(i)
    fake_correct = torch.sigmoid(disc(e, i)) < 0.5
    real_correct = torch.sigmoid(disc(e, i_shuffled)) >= 0.5
    n = fake_correct.numel() + real_correct.numel()
    return (fake_correct.float().sum() + real_correct.float().sum()).item() / n


# ---------------------------------------------------------------------------
# Stage 1 – Expression encoder pre-training
# ---------------------------------------------------------------------------

def train_stage1(
    expr_encoder: ExpressionEncoder,
    stage1_criterion: Stage1Loss,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    device: torch.device,
    save_path: str | None = None,
) -> list[dict[str, float]]:
    """
    Train the expression encoder using the MINE-based Stage-1 objective.

    Returns
    -------
    history : list of per-epoch metric dicts
    """
    history = []

    for epoch in range(num_epochs):
        expr_encoder.train()
        stage1_criterion.train()

        epoch_loss = epoch_global_mi = epoch_local_mi = epoch_l1 = 0.0

        for M, N, _ in train_loader:
            M, N = M.to(device), N.to(device)

            optimizer.zero_grad(set_to_none=True)

            E_M, pooled_M, fmap_M = expr_encoder(M)
            E_N, pooled_N, fmap_N = expr_encoder(N)

            loss, global_mi, local_mi, l1 = stage1_criterion(
                pooled_M, pooled_N, fmap_M, fmap_N, E_M, E_N
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_global_mi += global_mi.item()
            epoch_local_mi += local_mi.item()
            epoch_l1 += l1.item()

        n = len(train_loader)
        metrics = {
            "epoch": epoch + 1,
            "loss": epoch_loss / n,
            "global_mi": epoch_global_mi / n,
            "local_mi": epoch_local_mi / n,
            "l1": epoch_l1 / n,
        }
        history.append(metrics)

        print(
            f"[Stage1] Epoch {metrics['epoch']:03d} | "
            f"Loss={metrics['loss']:.4f} | "
            f"GMI={metrics['global_mi']:.4f} | "
            f"LMI={metrics['local_mi']:.4f} | "
            f"L1={metrics['l1']:.4f}"
        )

    if save_path:
        torch.save(expr_encoder.state_dict(), save_path)
        print(f"[Stage1] Encoder saved → {save_path}")

    return history


# ---------------------------------------------------------------------------
# Stage 2 – Identity encoder training with adversarial disentanglement
# ---------------------------------------------------------------------------

def train_stage2(
    expr_encoder: ExpressionEncoder,
    identity_enc: IdentityEncoder,
    disc: Discriminator,
    mi_loss_fn: Stage2MILoss,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    device: torch.device,
    adv_weight: float = 0.025,
    save_path_enc: str | None = None,
    save_path_disc: str | None = None,
) -> list[dict[str, float]]:
    """
    Train the identity encoder while keeping the expression encoder frozen.

    Returns
    -------
    history : list of per-epoch metric dicts
    """
    # Freeze expression encoder
    for p in expr_encoder.parameters():
        p.requires_grad = False
    expr_encoder.eval()

    history = []

    for epoch in range(num_epochs):
        identity_enc.train()
        mi_loss_fn.train()

        mi_sum = acc_sum = 0.0
        steps = 0

        for M, N, _ in train_loader:
            M, N = M.to(device), N.to(device)

            with torch.no_grad():
                EM, _, _ = expr_encoder(M)
                EN, _, _ = expr_encoder(N)

            IM, pIM, fIM = identity_enc(M)
            IN, pIN, fIN = identity_enc(N)

            # -- Discriminator update --
            d_loss = (
                _discriminator_loss(disc, EM, IM.detach())
                + _discriminator_loss(disc, EN, IN.detach())
            )
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # -- Generator (identity encoder + MINE) update --
            TM = torch.cat([EM, IM], dim=1)
            TN = torch.cat([EN, IN], dim=1)

            mi_loss, g_mi, l_mi = mi_loss_fn(pIM, pIN, fIM, fIN, TM, TN)
            fool = (
                _encoder_adversarial_loss(disc, EM, IM)
                + _encoder_adversarial_loss(disc, EN, IN)
            )
            g_loss = mi_loss + adv_weight * fool

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            with torch.no_grad():
                acc_val = (
                    _discriminator_accuracy(disc, EM, IM)
                    + _discriminator_accuracy(disc, EN, IN)
                ) / 2

            mi_sum += (g_mi + l_mi).item()
            acc_sum += acc_val
            steps += 1

        metrics = {
            "epoch": epoch + 1,
            "mi": mi_sum / steps,
            "disc_acc": acc_sum / steps,
        }
        history.append(metrics)
        print(
            f"[Stage2] Epoch {metrics['epoch']:03d} | "
            f"MI={metrics['mi']:.4f} | "
            f"disc_acc={metrics['disc_acc']:.3f}"
        )

    if save_path_enc:
        torch.save(identity_enc.state_dict(), save_path_enc)
        print(f"[Stage2] Identity encoder saved → {save_path_enc}")
    if save_path_disc:
        torch.save(disc.state_dict(), save_path_disc)
        print(f"[Stage2] Discriminator saved → {save_path_disc}")

    return history


# ---------------------------------------------------------------------------
# Stage 3 – FER classifier
# ---------------------------------------------------------------------------

def train_stage3(
    expr_encoder: ExpressionEncoder,
    classifier: FERClassifier,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    device: torch.device,
    save_path: str | None = None,
) -> list[dict[str, float]]:
    """
    Fine-tune the FER classifier on top of a frozen expression encoder.

    Returns
    -------
    history : list of per-epoch metric dicts
    """
    # Expression encoder stays frozen
    for p in expr_encoder.parameters():
        p.requires_grad = False
    expr_encoder.eval()

    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(num_epochs):
        classifier.train()
        epoch_loss = correct = total = 0

        for M, _, label in train_loader:
            M, label = M.to(device), label.to(device)

            with torch.no_grad():
                E_M, _, _ = expr_encoder(M)

            logits = classifier(E_M)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (logits.argmax(dim=1) == label).sum().item()
            total += label.size(0)

        train_acc = correct / total
        val_acc = evaluate_classifier(expr_encoder, classifier, val_loader, device)

        metrics = {
            "epoch": epoch + 1,
            "loss": epoch_loss / len(train_loader),
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(metrics)
        print(
            f"[Stage3] Epoch {metrics['epoch']:03d} | "
            f"Loss={metrics['loss']:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    if save_path:
        torch.save(classifier.state_dict(), save_path)
        print(f"[Stage3] Classifier saved → {save_path}")

    return history


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_classifier(
    expr_encoder: ExpressionEncoder,
    classifier: FERClassifier,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Return top-1 accuracy on the validation set."""
    expr_encoder.eval()
    classifier.eval()

    correct = total = 0
    for M, _, label in val_loader:
        M, label = M.to(device), label.to(device)
        E_M, _, _ = expr_encoder(M)
        correct += (classifier(E_M).argmax(dim=1) == label).sum().item()
        total += label.size(0)

    return correct / total


def compute_mig(
    e_m: torch.Tensor,
    e_n: torch.Tensor,
    i_m: torch.Tensor,
    device: torch.device,
    train_steps: int = 300,
    lr: float = 1e-3,
) -> dict[str, float]:
    """
    Compute the Mutual Information Gap (MIG) between expression and identity.

    Parameters
    ----------
    e_m, e_n : expression embeddings for two paired views
    i_m      : identity embeddings for the first view
    """
    mine_ee = StableMINE(PairStatisticsNetwork(64)).to(device)
    mine_ei = StableMINE(PairStatisticsNetwork(64)).to(device)
    opt_ee = torch.optim.Adam(mine_ee.parameters(), lr=lr)
    opt_ei = torch.optim.Adam(mine_ei.parameters(), lr=lr)

    n = e_m.size(0)
    batch_size = min(64, n)

    def _train_one(mine_net, opt, x, y) -> float:
        best = -float("inf")
        for _ in range(train_steps):
            idx = torch.randperm(n, device=device)[:batch_size]
            loss, mi = mine_net(x[idx], y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            best = max(best, mi.item())
        return best

    i_ee = _train_one(mine_ee, opt_ee, e_m, e_n)
    i_ei = _train_one(mine_ei, opt_ei, e_m, i_m)

    return {"I_E_E": i_ee, "I_E_I": i_ei, "MIG": i_ee - i_ei}