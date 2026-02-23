"""Prithvi v2 model module for GEO-Bench classification."""

from typing import List, Optional

import torch
import torch.nn as nn
from geobench.task import TaskSpecifications
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

from geobench_exp.torch_toolbox.model import GeoBenchBaseModule


class GeoBenchPrithviClassifier(GeoBenchBaseModule):
    """GEO-Bench Lightning module wrapping the Prithvi-EO v2 ViT backbone.

    Loads a pretrained Prithvi-EO encoder via terratorch, pools the CLS token,
    and attaches a linear classification head.  Supports any Prithvi v2 variant
    registered in ``terratorch.models.backbones.prithvi_vit.prithvi_cfgs``.

    Band mapping: by default the model is configured for the 6 HLS bands used
    during Prithvi pre-training (Blue, Green, Red, NIR-Narrow, SWIR-1, SWIR-2).
    Pass ``model_bands`` in the YAML config to override for any other channel
    selection (uses integer band indices as fallback).
    """

    def __init__(
        self,
        task_specs: TaskSpecifications,
        in_channels: int,
        prithvi_variant: str = "prithvi_eo_v2_100",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> None:
        """Initialise GeoBenchPrithviClassifier.

        Args:
            task_specs: GEO-Bench task specification object.
            in_channels: number of input spectral channels (set automatically
                from ``band_names`` in the data config).
            prithvi_variant: name of the Prithvi variant to load, e.g.
                ``"prithvi_eo_v2_100"``, ``"prithvi_eo_v2_300"``.
            pretrained: whether to download and load pretrained HuggingFace
                weights for the backbone.
            freeze_backbone: freeze all backbone parameters and only train the
                classification head (linear probing mode).
            optimizer: partial optimizer callable (passed via Hydra config).
            lr_scheduler: optional partial LR scheduler callable.
        """
        self.save_hyperparameters(ignore=["task_specs"])
        super().__init__(task_specs, in_channels, freeze_backbone, optimizer, lr_scheduler)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def configure_the_model(self) -> None:
        """Build the Prithvi ViT backbone and attach a classification head."""
        from terratorch.datasets import HLSBands
        from terratorch.models.backbones.prithvi_vit import _create_prithvi

        in_channels: int = self.hparams["in_channels"]

        # Default: map to the 6 HLS bands used during Prithvi pre-training.
        # If a different number of channels is requested we fall back to
        # integer band indices (no weight matching from the patch embed).
        _default_bands: dict[int, List] = {
            6: [
                HLSBands.BLUE,
                HLSBands.GREEN,
                HLSBands.RED,
                HLSBands.NIR_NARROW,
                HLSBands.SWIR_1,
                HLSBands.SWIR_2,
            ],
            3: [HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED],
        }
        model_bands = _default_bands.get(in_channels, list(range(in_channels)))

        self.backbone = _create_prithvi(
            variant=self.hparams["prithvi_variant"],
            pretrained=self.hparams["pretrained"],
            model_bands=model_bands,
            num_frames=1,
            encoder_only=True,
        )

        embed_dim: int = self.backbone.embed_dim
        n_classes: int = self.task_specs.label_type.n_classes
        self.head = nn.Linear(embed_dim, n_classes)

        if self.hparams["freeze_backbone"]:
            for param in self.backbone.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder + head.

        Args:
            x: input tensor of shape ``(B, C, H, W)``.

        Returns:
            logits of shape ``(B, n_classes)``.
        """
        # forward_features returns a list of per-block outputs, each (B, 1+N, D).
        # We use the CLS token (position 0) from the final block.
        features: list = self.backbone.forward_features(x)
        cls_token = features[-1][:, 0]  # (B, embed_dim)
        return self.head(cls_token)
