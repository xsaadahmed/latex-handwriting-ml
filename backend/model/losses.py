from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch import nn


GanLossType = Literal["bce", "mse"]


@dataclass
class Pix2PixLoss:
    """
    Loss bundle for Pix2Pix.

    Components:
    - Adversarial loss: BCEWithLogits or MSE (LSGAN-style)
    - L1 reconstruction loss
    - Generator objective: adv + lambda_L1 * L1
    """

    lambda_L1: float = 100.0
    gan_loss: GanLossType = "bce"

    def __post_init__(self) -> None:
        if self.gan_loss == "bce":
            # Discriminator returns logits; use BCEWithLogits for numerical stability.
            self._adv_criterion: nn.Module = nn.BCEWithLogitsLoss()
        elif self.gan_loss == "mse":
            self._adv_criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported gan_loss={self.gan_loss!r}. Use 'bce' or 'mse'.")

        self._l1_criterion = nn.L1Loss()

    def generator_loss(
        self,
        fake_output: torch.Tensor,
        real_output: torch.Tensor,
        fake_predictions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute generator loss.

        Parameters
        ----------
        fake_output:
            Generator output image (B, 1, H, W), typically in [-1, 1].
        real_output:
            Ground-truth target image (B, 1, H, W), typically in [-1, 1].
        fake_predictions:
            Discriminator patch logits/predictions for (source, fake_output).

        Returns
        -------
        total, adv, l1
            Combined generator loss and its components.
        """
        ones = torch.ones_like(fake_predictions)

        adv = self._adv_criterion(fake_predictions, ones)
        l1 = self._l1_criterion(fake_output, real_output)
        total = adv + (self.lambda_L1 * l1)
        return total, adv, l1

    def discriminator_loss(
        self,
        real_predictions: torch.Tensor,
        fake_predictions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute discriminator loss.

        Parameters
        ----------
        real_predictions:
            Discriminator patch logits/predictions for (source, real_output).
        fake_predictions:
            Discriminator patch logits/predictions for (source, fake_output.detach()).

        Returns
        -------
        total, real_loss, fake_loss
            Combined discriminator loss and its components.
        """
        ones = torch.ones_like(real_predictions)
        zeros = torch.zeros_like(fake_predictions)

        real_loss = self._adv_criterion(real_predictions, ones)
        fake_loss = self._adv_criterion(fake_predictions, zeros)
        total = 0.5 * (real_loss + fake_loss)
        return total, real_loss, fake_loss

