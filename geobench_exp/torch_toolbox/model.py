"""Model."""

import os
import time
from typing import Callable, Dict, List, Optional, Union

import lightning
import numpy as np
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn.functional as F
import torchmetrics
from geobench.dataset import SegmentationClasses
from geobench.label import Classification, MultiLabelClassification
from geobench.task import TaskSpecifications
from lightning import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torchgeo.models import get_weight
from torchgeo.trainers import utils
from torchvision.models._api import WeightsEnum




class GeoBenchBaseModule(LightningModule):
    """GeoBench Base Lightning Module."""

    def __init__(
        self,
        task_specs: TaskSpecifications,
        in_channels,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> None:
        """Initialize a new ClassificationTask instance.

        Args:
            task_specs: an object describing the task to be performed
            model: Name of the `timm
                <https://huggingface.co/docs/timm/reference/models>`__ model to use.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False
                or None for random weights, or the path to a saved model state dict.
            in_channels: Number of input channels to model.
            freeze_backbone: Freeze the backbone network to linear probe
                the classifier head.
            optimizer: Optimizer to use for training
            lr_scheduler: Learning rate scheduler to use for training
        """
        super().__init__()
        self.task_specs = task_specs

        self.loss_fn = train_loss_generator(task_specs)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_metrics = eval_metrics_generator(task_specs)
        self.eval_metrics = eval_metrics_generator(task_specs)
        self.test_metrics = eval_metrics_generator(task_specs)

        self.configure_the_model()

        # Fixed val sample for WandB triplet tracking (set once on first val pass).
        self._fixed_val_input: Optional[Tensor] = None
        self._fixed_val_target: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor (N, C, H, W)

        Returns:
            Tensor (N, num_classes)
        """
        return self.model(x)

    def configure_the_model(self) -> None:
        """Initialize the model."""
        raise NotImplementedError("Necessary to define a model.")

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx=0) -> Dict[str, Tensor]:  # type: ignore
        """Define steps taken during training mode.

        Args:
            batch: input batch
            batch_idx: index of batch

        Returns:
            training step outputs
        """
        inputs, target = batch["input"], batch["label"]
        output = self(inputs)
        loss_train = self.loss_fn(output, target)
        self.train_metrics(output, target)
        self.log("train_loss", loss_train, logger=True)
        self.log("current_time", time.time(), logger=True)

        return loss_train

    def on_train_epoch_end(self, *arg, **kwargs) -> None:  # type: ignore
        """Define actions after a training epoch.

        Args:
            outputs: outputs from :meth:`__training_step`
        """
        self.log_dict({f"train_{k}": v.mean() for k, v in self.train_metrics.compute().items()}, logger=True)
        self.train_metrics.reset()

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int):
        """Define steps taken during validation mode.

        Args:
            batch: input batch
            batch_idx: index of batch
            dataloader_idx: index of dataloader

        Returns:
            validation step outputs
        """
        self.prefix = ["val", "test"][dataloader_idx]
        inputs, target = batch["input"], batch["label"]
        output = self(inputs)
        loss = self.loss_fn(output, target)
        self.log(f"{self.prefix}_loss", loss)
        if self.prefix == "val":
            self.eval_metrics(output, target)
            # Cache the very first sample once — reused every epoch for the WandB triplet.
            if self._fixed_val_input is None and batch_idx == 0:
                self._fixed_val_input = inputs[0:1].detach().cpu()
                self._fixed_val_target = target[0:1].detach().cpu()
        else:
            self.test_metrics(output, target)

        return loss

    def on_validation_epoch_end(self):
        """Define actions after a validation epoch."""

        # eval metrics
        eval_metrics = self.eval_metrics.compute()
        self.log_dict({f"val_{k}": v.mean() for k, v in eval_metrics.items()}, logger=True)
        self.eval_metrics.reset()

        # test metrics
        test_metrics = self.test_metrics.compute()
        self.log_dict({f"test_{k}": v.mean() for k, v in test_metrics.items()}, logger=True)
        self.test_metrics.reset()

        self._maybe_log_wandb_images()

    # ------------------------------------------------------------------
    # WandB image triplet visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def _to_rgb_uint8(tensor: Tensor) -> "np.ndarray":
        """Convert a (C, H, W) float tensor to an (H, W, 3) uint8 RGB array.

        Uses the first three channels as R, G, B and applies a per-image
        percentile stretch (2nd–98th) so the image looks sensible regardless
        of normalisation.
        """
        img = tensor[:3].float().numpy()  # (3, H, W)
        out = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)
        for i in range(3):
            ch = img[i]
            lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
            if hi > lo:
                ch = (ch - lo) / (hi - lo)
            else:
                ch = np.zeros_like(ch)
            out[:, :, i] = (np.clip(ch, 0, 1) * 255).astype(np.uint8)
        return out

    @staticmethod
    def _colorise_mask(mask: Tensor, n_classes: int) -> "np.ndarray":
        """Map an integer class mask (H, W) to an (H, W, 3) uint8 colour image."""
        palette = np.array(
            [
                [int((i * 67 + 41) % 256), int((i * 113 + 97) % 256), int((i * 151 + 53) % 256)]
                for i in range(n_classes)
            ],
            dtype=np.uint8,
        )
        return palette[mask.numpy().astype(int)]

    @staticmethod
    def _draw_label_panel(H: int, W: int, colour: "np.ndarray", label: str) -> "np.ndarray":
        """Render a solid-colour block of size (H, W, 3) with *label* centred over it."""
        from PIL import Image, ImageDraw, ImageFont

        panel = np.full((H, W, 3), colour, dtype=np.uint8)
        img = Image.fromarray(panel)
        draw = ImageDraw.Draw(img)

        # Choose a font size that fits within the panel width.
        font: ImageFont.ImageFont
        font_size = max(12, W // 8)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

        # Measure text and wrap long labels at word boundaries; honour explicit newlines.
        paragraphs = label.split("\n")
        lines: List[str] = []
        for para in paragraphs:
            words = para.split()
            current = ""
            for word in words:
                test = (current + " " + word).strip()
                bbox = draw.textbbox((0, 0), test, font=font)
                if bbox[2] - bbox[0] > W - 8 and current:
                    lines.append(current)
                    current = word
                else:
                    current = test
            if current:
                lines.append(current)

        # Compute total text block height and pick a contrasting text colour.
        line_h = draw.textbbox((0, 0), "A", font=font)[3] + 2
        total_h = line_h * len(lines)
        y0 = (H - total_h) // 2
        brightness = int(colour[0]) * 299 + int(colour[1]) * 587 + int(colour[2]) * 114
        text_colour = (0, 0, 0) if brightness > 128_000 else (255, 255, 255)

        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            x = (W - (bbox[2] - bbox[0])) // 2
            draw.text((x, y0 + i * line_h), line, fill=text_colour, font=font)

        return np.array(img)

    def _maybe_log_wandb_images(self) -> None:
        """Log a single fixed-sample triplet (image | GT | pred) to WandB each epoch.

        The same validation sample is reused every epoch so the model's evolution
        on that sample can be tracked over time.  No-ops when WandB is not active.
        """
        try:
            import wandb
        except ImportError:
            return

        wandb_logger: Optional[WandbLogger] = None
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
                break
        if wandb_logger is None or self._fixed_val_input is None:
            return

        is_segmentation = isinstance(self.task_specs.label_type, SegmentationClasses)
        n_classes = self.task_specs.label_type.n_classes
        class_names = getattr(self.task_specs.label_type, "class_names", None)

        # Re-run inference on the fixed sample with no gradient.
        with torch.no_grad():
            logits = self(self._fixed_val_input.to(self.device))  # (1, n_classes, ...)
        pred = logits.argmax(dim=1).squeeze(0).cpu()  # scalar or (H, W)

        inp = self._fixed_val_input[0]   # (C, H, W)
        gt = self._fixed_val_target[0]   # scalar tensor or (H, W)

        rgb = self._to_rgb_uint8(inp)    # (H, W, 3)
        H, W = rgb.shape[:2]

        if is_segmentation:
            gt_panel = self._colorise_mask(gt, n_classes)       # (H, W, 3)
            pred_panel = self._colorise_mask(pred, n_classes)   # (H, W, 3)
            caption = f"epoch {self.current_epoch}  |  image · GT · pred"
        else:
            gt_idx = int(gt.item()) if gt.ndim == 0 else int(gt.argmax().item())
            pred_idx = int(pred.item())
            if class_names and gt_idx < len(class_names):
                gt_label = class_names[gt_idx]
                pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
            else:
                gt_label, pred_label = str(gt_idx), str(pred_idx)
            # Render GT and pred as colour blocks with the class label overlaid.
            palette = np.array(
                [[int((i * 67 + 41) % 256), int((i * 113 + 97) % 256), int((i * 151 + 53) % 256)]
                 for i in range(n_classes)],
                dtype=np.uint8,
            )
            gt_panel = self._draw_label_panel(H, W, palette[gt_idx % n_classes], f"GT\n{gt_label}")
            pred_panel = self._draw_label_panel(H, W, palette[pred_idx % n_classes], f"Pred\n{pred_label}")
            correct = "\u2713" if gt_idx == pred_idx else "\u2717"
            caption = f"epoch {self.current_epoch}  {correct}  GT: {gt_label} | Pred: {pred_label}"

        # Single horizontally-stacked panel: image | GT | pred
        panel = np.concatenate([rgb, gt_panel, pred_panel], axis=1)
        wandb_logger.experiment.log(
            {"val_triplet": wandb.Image(panel, caption=caption), "epoch": self.current_epoch},
            step=wandb_logger.experiment.step,
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Define steps taken during test mode.

        Args:
            batch: input batch
            batch_idx: index of batch

        Returns:
            test step outputs
        """
        inputs, target = batch["input"], batch["label"]
        output = self(inputs)
        loss = self.loss_fn(output, target)
        self.log("test_loss", loss)
        self.test_metrics(output, target)
        return loss

    def on_test_epoch_end(self, *arg, **kwargs):
        """Define actions after a test epoch.

        Args:
            outputs: outputs from :meth:`__test_step`
        """
        test_metrics = self.test_metrics.compute()
        self.log_dict({f"test_{k}": v.mean() for k, v in test_metrics.items()}, logger=True)
        self.test_metrics.reset()

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}


class GeoBenchClassifier(GeoBenchBaseModule):
    def __init__(
        self,
        task_specs: TaskSpecifications,
        model: str,
        in_channels: int,
        weights: Union[WeightsEnum, str, bool, None] = None,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> None:
        self.save_hyperparameters(ignore=["loss_fn", "task_specs"])
        self.hparams["model"] = model
        self.weights = weights
        super().__init__(task_specs, in_channels, freeze_backbone, optimizer, lr_scheduler)

    def configure_the_model(self) -> None:
        """Configure classification model."""
        # Create model
        self.model = timm.create_model(
            self.hparams["model"],
            num_classes=self.task_specs.label_type.n_classes,
            in_chans=self.hparams["in_channels"],
            pretrained=self.weights is True,
        )

        # Load weights
        if self.weights and self.weights is not True:
            if isinstance(self.weights, WeightsEnum):
                state_dict = self.weights.get_state_dict(progress=True)
            elif os.path.exists(self.weights):
                _, state_dict = utils.extract_backbone(self.weights)
            else:
                state_dict = get_weight(self.weights).get_state_dict(progress=True)

            utils.load_state_dict(self.model, state_dict)

        # Freeze backbone and unfreeze classifier head
        if self.hparams["freeze_backbone"]:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True


class GeoBenchSegmentation(GeoBenchBaseModule):
    def __init__(
        self,
        task_specs: TaskSpecifications,
        encoder_type: str,
        decoder_type: str,
        in_channels: int,
        encoder_weights: Union[WeightsEnum, str, bool, None] = None,
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
    ) -> None:
        self.save_hyperparameters(ignore=["loss_fn", "task_specs"])

        super().__init__(task_specs, in_channels, freeze_backbone, optimizer, lr_scheduler)

    def configure_the_model(self) -> None:
        """Configure segmentation model."""
        # Load segmentation backbone from py-segmentation-models
        self.model = getattr(smp, self.hparams["decoder_type"])(
            encoder_name=self.hparams["encoder_type"],
            encoder_weights=self.hparams["encoder_weights"],
            in_channels=self.hparams["in_channels"],
            classes=self.task_specs.label_type.n_classes,
        )  # model output channels (number of cl


def eval_metrics_generator(task_specs: TaskSpecifications) -> List[torchmetrics.MetricCollection]:
    """Return the appropriate eval function depending on the task_specs.

    Args:
        task_specs: an object describing the task to be performed
        hyperparams: dictionary containing hyperparameters of the experiment

    Returns:
        metric collection used during evaluation
    """
    metrics: List[torchmetrics.MetricCollection] = {  # type: ignore
        Classification: torchmetrics.MetricCollection(
            {"Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=task_specs.label_type.n_classes)}
        ),
        SegmentationClasses: torchmetrics.MetricCollection(
            {
                "Jaccard": torchmetrics.JaccardIndex(task="multiclass", num_classes=task_specs.label_type.n_classes),
                "FBeta": torchmetrics.FBetaScore(
                    task="multiclass",
                    num_classes=task_specs.label_type.n_classes,
                    beta=2.0,
                    multidim_average="samplewise",
                ),
            }
        ),
        MultiLabelClassification: torchmetrics.MetricCollection(
            {"F1Score": torchmetrics.F1Score(task="multilabel", num_labels=task_specs.label_type.n_classes)}
        ),
    }[task_specs.label_type.__class__]

    return metrics


def _balanced_binary_cross_entropy_with_logits(outputs: Tensor, targets: Tensor) -> Tensor:
    """Compute balance binary cross entropy for multi-label classification.

    Args:
        outputs: model outputs
        targets: targets to compute binary cross entropy on
    """
    classes = targets.shape[-1]
    outputs = outputs.view(-1, classes)
    targets = targets.view(-1, classes).float()
    loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    loss = loss[targets == 0].mean() + loss[targets == 1].mean()
    return loss


def train_loss_generator(task_specs: TaskSpecifications) -> Callable[[Tensor], Tensor]:
    """Return the appropriate loss function depending on the task_specs.

    Args:
        task_specs: an object describing the task to be performed

    Returns:
        available loss functions for training
    """
    loss = {
        Classification: F.cross_entropy,
        MultiLabelClassification: _balanced_binary_cross_entropy_with_logits,
        SegmentationClasses: F.cross_entropy,
    }[task_specs.label_type.__class__]

    return loss  # type: ignore
