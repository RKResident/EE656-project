"""
main.py
-------
Entry point that wires together the three training stages for the
AffectNet DICE-FER pipeline.

Usage
-----
    python main.py

Adjust the CONFIG dict below or pass CLI args to change paths / hyper-params.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from datasets import build_dataloaders
from models import (
    ExpressionEncoder,
    IdentityEncoder,
    FERClassifier,
    Discriminator,
    GlobalStatisticsNetwork,
    LocalStatisticsNetwork,
    GlobalStatisticsNetworkID,
    LocalStatisticsNetworkID,
    Stage1Loss,
    Stage2MILoss,
    StableMINE,
    LocalMINE,
)
from trainer import (
    train_stage1,
    train_stage2,
    train_stage3,
    evaluate_classifier,
    compute_mig,
)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    "train_path": "/path/to/AffectNet/Train",
    "test_path":  "/path/to/AffectNet/Test",
    "save_dir":   "./checkpoints",

    # ── Data ───────────────────────────────────────────────────────────────
    "img_size":    112,
    "batch_size":  32,
    "num_workers": 4,

    # ── Stage 1 ────────────────────────────────────────────────────────────
    "stage1_epochs": 40,
    "stage1_lr":     1e-4,
    "stage1_delta":  0.1,        # L1 regularisation weight

    # ── Stage 2 ────────────────────────────────────────────────────────────
    "stage2_epochs":     30,
    "stage2_lr_gen":     1e-4,
    "stage2_lr_disc":    1e-4,
    "stage2_adv_weight": 0.025,

    # ── Stage 3 ────────────────────────────────────────────────────────────
    "stage3_epochs": 30,
    "stage3_lr":     1e-4,

    # ── MIG evaluation ─────────────────────────────────────────────────────
    "mig_train_steps": 300,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_save_path(save_dir: str, filename: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, filename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    print("\n── Loading data ─────────────────────────────────────────────────")
    train_loader, val_loader, train_imagefolder = build_dataloaders(
        train_path=cfg["train_path"],
        test_path=cfg["test_path"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        img_size=cfg["img_size"],
    )
    num_classes = len(train_imagefolder.classes)
    print(f"Classes ({num_classes}): {train_imagefolder.classes}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Stage 1 – Expression encoder ────────────────────────────────────────
    print("\n── Stage 1: Expression encoder ──────────────────────────────────")
    expr_encoder = ExpressionEncoder().to(device)

    global_stats_net = GlobalStatisticsNetwork().to(device)
    local_stats_net  = LocalStatisticsNetwork().to(device)
    global_mine      = StableMINE(global_stats_net).to(device)
    local_mine       = LocalMINE(local_stats_net).to(device)
    stage1_criterion = Stage1Loss(global_mine, local_mine, delta=cfg["stage1_delta"])

    optimizer_stage1 = torch.optim.Adam(
        list(expr_encoder.parameters())
        + list(global_stats_net.parameters())
        + list(local_stats_net.parameters()),
        lr=cfg["stage1_lr"],
    )

    train_stage1(
        expr_encoder=expr_encoder,
        stage1_criterion=stage1_criterion,
        optimizer=optimizer_stage1,
        train_loader=train_loader,
        num_epochs=cfg["stage1_epochs"],
        device=device,
        save_path=_make_save_path(cfg["save_dir"], "expr_encoder_stage1.pt"),
    )

    # ── Stage 2 – Identity encoder ──────────────────────────────────────────
    print("\n── Stage 2: Identity encoder ────────────────────────────────────")
    identity_enc = IdentityEncoder().to(device)
    disc         = Discriminator().to(device)

    global_mine_id = StableMINE(GlobalStatisticsNetworkID()).to(device)
    local_mine_id  = LocalMINE(LocalStatisticsNetworkID()).to(device)
    mi_loss_fn     = Stage2MILoss(global_mine_id, local_mine_id).to(device)

    optimizer_g = torch.optim.Adam(
        list(identity_enc.parameters())
        + list(global_mine_id.parameters())
        + list(local_mine_id.parameters()),
        lr=cfg["stage2_lr_gen"],
    )
    optimizer_d = torch.optim.Adam(disc.parameters(), lr=cfg["stage2_lr_disc"])

    train_stage2(
        expr_encoder=expr_encoder,
        identity_enc=identity_enc,
        disc=disc,
        mi_loss_fn=mi_loss_fn,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        train_loader=train_loader,
        num_epochs=cfg["stage2_epochs"],
        device=device,
        adv_weight=cfg["stage2_adv_weight"],
        save_path_enc=_make_save_path(cfg["save_dir"], "identity_encoder_stage2.pt"),
        save_path_disc=_make_save_path(cfg["save_dir"], "discriminator_stage2.pt"),
    )

    # ── MIG evaluation ──────────────────────────────────────────────────────
    print("\n── MIG evaluation ───────────────────────────────────────────────")
    expr_encoder.eval()
    identity_enc.eval()

    eval_M_list, eval_N_list = [], []
    for M, N, _ in val_loader:
        eval_M_list.append(M)
        eval_N_list.append(N)
    eval_M = torch.cat(eval_M_list).to(device)
    eval_N = torch.cat(eval_N_list).to(device)

    with torch.no_grad():
        eval_E_M, _, _ = expr_encoder(eval_M)
        eval_E_N, _, _ = expr_encoder(eval_N)
        eval_I_M, _, _ = identity_enc(eval_M)

    mig_scores = compute_mig(
        eval_E_M, eval_E_N, eval_I_M,
        device=device,
        train_steps=cfg["mig_train_steps"],
    )
    print(f"MIG results: {mig_scores}")

    # ── Stage 3 – FER classifier ────────────────────────────────────────────
    print("\n── Stage 3: FER classifier ──────────────────────────────────────")
    classifier = FERClassifier(num_classes=num_classes).to(device)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg["stage3_lr"])

    train_stage3(
        expr_encoder=expr_encoder,
        classifier=classifier,
        optimizer=classifier_optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg["stage3_epochs"],
        device=device,
        save_path=_make_save_path(cfg["save_dir"], "classifier_stage3.pt"),
    )

    # ── Final evaluation ────────────────────────────────────────────────────
    final_acc = evaluate_classifier(expr_encoder, classifier, val_loader, device)
    print(f"\nFinal validation accuracy: {final_acc:.4f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> dict:
    parser = argparse.ArgumentParser(description="AffectNet DICE-FER training")
    parser.add_argument("--train-path",      default=CONFIG["train_path"])
    parser.add_argument("--test-path",       default=CONFIG["test_path"])
    parser.add_argument("--save-dir",        default=CONFIG["save_dir"])
    parser.add_argument("--batch-size",      type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--num-workers",     type=int,   default=CONFIG["num_workers"])
    parser.add_argument("--stage1-epochs",   type=int,   default=CONFIG["stage1_epochs"])
    parser.add_argument("--stage2-epochs",   type=int,   default=CONFIG["stage2_epochs"])
    parser.add_argument("--stage3-epochs",   type=int,   default=CONFIG["stage3_epochs"])
    args = parser.parse_args()

    cfg = dict(CONFIG)  # start from defaults
    cfg.update({
        "train_path":    args.train_path,
        "test_path":     args.test_path,
        "save_dir":      args.save_dir,
        "batch_size":    args.batch_size,
        "num_workers":   args.num_workers,
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "stage3_epochs": args.stage3_epochs,
    })
    return cfg


if __name__ == "__main__":
    main(_parse_args())