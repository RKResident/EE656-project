"""
models.py
---------
All neural-network architectures used in the three-stage FER pipeline:

  Stage 1 – Expression encoder (MINE-based disentanglement)
  Stage 2 – Identity encoder + adversarial discriminator
  Stage 3 – Linear FER classifier

  Supporting modules: statistics networks, MINE objectives, loss wrappers.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class ExpressionEncoder(nn.Module):
    """
    ResNet-18 backbone that produces:
      z       – 64-d expression embedding (projection head output)
      pooled  – 512-d global-pooled feature
      fmap    – spatial feature map from the final ResNet block
    """

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.feature_extractor = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(512, 64)

    def forward(self, x):
        fmap = self.feature_extractor(x)
        pooled = self.avgpool(fmap).flatten(1)
        z = self.projection(pooled)
        return z, pooled, fmap


class IdentityEncoder(nn.Module):
    """
    Identical architecture to ExpressionEncoder but trained to capture
    identity-specific (rather than expression-specific) information.
    """

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1")
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.feature_extractor = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(512, 64)

    def forward(self, x):
        fmap = self.feature_extractor(x)
        pooled = self.avgpool(fmap).flatten(1)
        z = self.projection(pooled)
        return z, pooled, fmap


# ---------------------------------------------------------------------------
# Statistics networks (used inside MINE estimators)
# ---------------------------------------------------------------------------

class GlobalStatisticsNetwork(nn.Module):
    """
    Stage 1: maps (512-d pooled feature, 64-d expression code) → scalar.
    Input dim = 512 + 64 = 576.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(576, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 1),
        )

    def forward(self, pooled_feat, z):
        return self.net(torch.cat([pooled_feat, z], dim=1))


class LocalStatisticsNetwork(nn.Module):
    """
    Stage 1: maps (spatial feature map, 64-d expression code) → score map.
    Input channels = 512 + 64 = 576.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(576, 512, kernel_size=1), nn.ELU(),
            nn.Conv2d(512, 1, kernel_size=1),
        )

    def forward(self, fmap, z):
        B, C, H, W = fmap.shape
        z_exp = z.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        return self.net(torch.cat([fmap, z_exp], dim=1))


class GlobalStatisticsNetworkID(nn.Module):
    """
    Stage 2: maps (512-d pooled feature, 128-d combined token) → scalar.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512 + 128, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 1),
        )

    def forward(self, pooled_feat, t):
        return self.net(torch.cat([pooled_feat, t], dim=1))


class LocalStatisticsNetworkID(nn.Module):
    """
    Stage 2: maps (spatial feature map, 128-d combined token) → score map.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(512 + 128, 512, kernel_size=1), nn.ELU(),
            nn.Conv2d(512, 1, kernel_size=1),
        )

    def forward(self, fmap, t):
        B, C, H, W = fmap.shape
        t_exp = t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        return self.net(torch.cat([fmap, t_exp], dim=1))


class PairStatisticsNetwork(nn.Module):
    """Generic statistics network for two dim-d embeddings (used in MIG eval)."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, a, b):
        return self.net(torch.cat([a, b], dim=1))


# ---------------------------------------------------------------------------
# MINE estimators
# ---------------------------------------------------------------------------

class StableMINE(nn.Module):
    """
    Mutual-information estimator (MINE) with log-sum-exp stabilisation.
    Wraps any global (vector × vector → scalar) statistics network.
    """

    def __init__(self, stats_network: nn.Module):
        super().__init__()
        self.stats_network = stats_network

    def forward(self, image_features, latent_codes):
        N = image_features.size(0)

        joint_scores = self.stats_network(image_features, latent_codes)
        joint_mean = joint_scores.mean()

        # All N×N marginal pairs
        x_exp = image_features.unsqueeze(1).expand(N, N, image_features.size(1))
        z_exp = latent_codes.unsqueeze(0).expand(N, N, latent_codes.size(1))

        all_scores = self.stats_network(
            x_exp.reshape(N * N, image_features.size(1)),
            z_exp.reshape(N * N, latent_codes.size(1)),
        ).squeeze()

        log_mean_exp = torch.logsumexp(all_scores, dim=0) - math.log(N * N)
        mi_estimate = joint_mean - log_mean_exp

        return -mi_estimate, mi_estimate.detach()


class LocalMINE(nn.Module):
    """
    Mutual-information estimator for spatial (feature-map × vector) pairs.
    Wraps any local (fmap × vector → score map) statistics network.
    """

    def __init__(self, stats_network: nn.Module):
        super().__init__()
        self.stats_network = stats_network

    def forward(self, fmap, latent_codes):
        B, C, H, W = fmap.shape

        joint_scores = self.stats_network(fmap, latent_codes).sum(dim=[2, 3])
        joint_mean = joint_scores.mean()

        # All B×B marginal pairs
        fmap_exp = fmap.unsqueeze(1).expand(B, B, C, H, W)
        z_exp = latent_codes.unsqueeze(0).expand(B, B, latent_codes.size(1))

        all_scores = self.stats_network(
            fmap_exp.reshape(B * B, C, H, W),
            z_exp.reshape(B * B, latent_codes.size(1)),
        ).sum(dim=[2, 3]).squeeze()

        log_mean_exp = torch.logsumexp(all_scores, dim=0) - math.log(B * B)
        mi_estimate = joint_mean - log_mean_exp

        return -mi_estimate, mi_estimate.detach()


# ---------------------------------------------------------------------------
# Loss modules
# ---------------------------------------------------------------------------

class Stage1Loss(nn.Module):
    """
    Combined Stage-1 objective:
      L = GlobalMINE + LocalMINE + delta * L1(E_M, E_N)
    """

    def __init__(self, global_mine: StableMINE, local_mine: LocalMINE, delta: float = 0.1):
        super().__init__()
        self.global_mine = global_mine
        self.local_mine = local_mine
        self.delta = delta
        self.l1_loss = nn.L1Loss()

    def forward(self, pooled_M, pooled_N, fmap_M, fmap_N, E_M, E_N):
        global_loss_M, global_mi_M = self.global_mine(pooled_M, E_N)
        global_loss_N, global_mi_N = self.global_mine(pooled_N, E_M)
        local_loss_M, local_mi_M = self.local_mine(fmap_M, E_N)
        local_loss_N, local_mi_N = self.local_mine(fmap_N, E_M)

        global_loss = 0.5 * (global_loss_M + global_loss_N)
        local_loss = 1.0 * (local_loss_M + local_loss_N)
        l1 = self.l1_loss(E_M, E_N)
        total_loss = global_loss + local_loss + self.delta * l1

        global_mi = 0.5 * (global_mi_M + global_mi_N)
        local_mi = 1.0 * (local_mi_M + local_mi_N)
        return total_loss, global_mi, local_mi, l1


class Stage2MILoss(nn.Module):
    """
    Stage-2 MI objective maximising I(image features; combined token T).
    """

    def __init__(self, global_mine_id: StableMINE, local_mine_id: LocalMINE):
        super().__init__()
        self.global_mine_id = global_mine_id
        self.local_mine_id = local_mine_id

    def forward(self, pooled_M, pooled_N, fmap_M, fmap_N, T_M, T_N):
        global_loss_M, global_mi_M = self.global_mine_id(pooled_M, T_M)
        global_loss_N, global_mi_N = self.global_mine_id(pooled_N, T_N)
        local_loss_M, local_mi_M = self.local_mine_id(fmap_M, T_M)
        local_loss_N, local_mi_N = self.local_mine_id(fmap_N, T_N)

        global_loss = 0.5 * (global_loss_M + global_loss_N)
        local_loss = 0.5 * (local_loss_M + local_loss_N)
        total_loss = global_loss + local_loss

        global_mi = 0.5 * (global_mi_M + global_mi_N)
        local_mi = 0.5 * (local_mi_M + local_mi_N)
        return total_loss, global_mi, local_mi


# ---------------------------------------------------------------------------
# Discriminator (Stage 2 adversarial head)
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """Distinguishes (expression, identity) joint pairs from shuffled pairs."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64 + 64, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, e, i):
        return self.net(torch.cat([e, i], dim=1))


# ---------------------------------------------------------------------------
# Classifier (Stage 3)
# ---------------------------------------------------------------------------

class FERClassifier(nn.Module):
    """Lightweight MLP that classifies expression embeddings into emotions."""

    def __init__(self, embedding_dim: int = 64, num_classes: int = 7,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, e):
        return self.net(e)