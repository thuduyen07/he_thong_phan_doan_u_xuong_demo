from __future__ import annotations

import threading

import numpy as np
from PIL import Image
from pathlib import Path
import torch.nn.functional as F

from backend.inference.contracts import PredictionArtifacts
from backend.inference.device import resolve_inference_device
from backend.inference.image_ops import blend_rgb, gray_to_rgb
from backend.inference.predictors.base import SegmentationPredictor
from backend.inference.runtime_paths import RESOURCES_DIR, ensure_runtime_paths

ensure_runtime_paths()

from runtime_src.common.io_utils import load_yaml  # noqa: E402
from runtime_src.seg.models import (  # noqa: E402
    build_segmentation_model,
    extract_logits,
    get_num_classes,
    get_model_type,
)
import torch  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PIL_RESAMPLING = getattr(Image, "Resampling", Image)
PIL_RESAMPLE_ALIASES = {
    "nearest": PIL_RESAMPLING.NEAREST,
    "bilinear": PIL_RESAMPLING.BILINEAR,
    "bicubic": PIL_RESAMPLING.BICUBIC,
}


class TorchSegmentationPredictor(SegmentationPredictor):
    def __init__(self, spec):
        super().__init__(spec)
        self.cfg = load_yaml(str(spec.config_path))
        if spec.config_overrides:
            self.cfg.update(spec.config_overrides)
        self._resolve_model_paths()
        self.device = resolve_inference_device(str(self.cfg.get("device", "cpu")))
        self.img_size = int(self.cfg["img_size"])
        self.num_classes = get_num_classes(self.cfg)
        self.model_type = get_model_type(self.cfg)
        self._resolve_inference_cfg()
        self._model = None
        self._model_lock = threading.Lock()

    def _resolve_model_paths(self) -> None:
        model_cfg = dict(self.cfg.get("model", {}))
        local_model_path = model_cfg.get("local_model_path", self.cfg.get("local_model_path"))
        if not local_model_path:
            return

        local_model_path = Path(str(local_model_path))
        if local_model_path.is_absolute():
            resolved = local_model_path
        else:
            candidate_roots = [
                Path(self.spec.config_path).resolve().parent,
                RESOURCES_DIR,
                RESOURCES_DIR / "pretrained",
            ]
            resolved = None
            for root in candidate_roots:
                candidate = (root / local_model_path).resolve()
                if candidate.exists():
                    resolved = candidate
                    break
            if resolved is None:
                resolved = (RESOURCES_DIR / local_model_path).resolve()
        model_cfg["local_model_path"] = str(resolved)
        self.cfg["model"] = model_cfg

    def _resolve_inference_cfg(self) -> None:
        if bool(dict(self.cfg.get("patch_forward", {})).get("enabled", False)):
            raise ValueError("The packaged web adapter does not support research patch-forward inference yet.")
        data_cfg = dict(self.cfg.get("data", {}))
        image_size_mode = str(data_cfg.get("image_size_mode", "fixed")).strip().lower()
        if image_size_mode not in {"fixed", "resize"}:
            raise ValueError("Live inference requires the same fixed-resize input mode as validation.")
        interpolation = str(data_cfg.get("resize_interpolation", "nearest")).strip().lower()
        if interpolation not in PIL_RESAMPLE_ALIASES:
            raise ValueError(f"Unsupported resize interpolation: {interpolation}")
        self.resize_interpolation = PIL_RESAMPLE_ALIASES[interpolation]

        normalization_cfg = dict(self.cfg.get("normalization", {}))
        self.normalization_enabled = bool(normalization_cfg.get("enabled", True))
        self.normalization_mean = torch.tensor(
            normalization_cfg.get("mean", IMAGENET_MEAN), dtype=torch.float32
        ).view(3, 1, 1)
        self.normalization_std = torch.tensor(
            normalization_cfg.get("std", IMAGENET_STD), dtype=torch.float32
        ).view(3, 1, 1)
        if torch.any(self.normalization_std <= 0):
            raise ValueError("Normalization standard deviations must be positive.")
        self.binary_threshold = float(self.cfg.get("binary_threshold", 0.5))
        self.threshold_source = "config.binary_threshold" if "binary_threshold" in self.cfg else "research_default"
        self.uncertainty_threshold = float(dict(self.cfg.get("uncertainty_weighting", {})).get("analysis_threshold", 0.5))
        self.boundary_enabled = bool(dict(self.cfg.get("boundary", {})).get("enabled", False))
        self.boundary_radius = int(dict(self.cfg.get("boundary", {})).get("radius", 1))
        if bool(dict(self.cfg.get("conformal", {})).get("enabled", False)):
            raise ValueError("Live CRC inference requires a packaged research CRC adapter and matching state artifact.")

    def _load_model(self):
        # Mirrors research ``load_model_weights`` student-checkpoint precedence.
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            checkpoint = torch.load(self.spec.checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = (
                checkpoint.get("student_state_dict")
                if isinstance(checkpoint, dict) and "student_state_dict" in checkpoint
                else checkpoint.get("model") if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            )
            inferred_num_classes = self._infer_num_classes_from_state_dict(state_dict)
            if inferred_num_classes is not None and inferred_num_classes != self.num_classes:
                self.num_classes = inferred_num_classes
                self.cfg["num_classes"] = inferred_num_classes
                model_cfg = dict(self.cfg.get("model", {}))
                model_cfg["num_classes"] = inferred_num_classes
                self.cfg["model"] = model_cfg

            model, _ = build_segmentation_model(self.cfg)
            model = model.to(self.device)
            if self.model_type == "segformer":
                state_dict = self._adapt_state_dict_to_model(state_dict, model.state_dict())

            model_state_dict = model.state_dict()
            compatible_state_dict = {
                key: value
                for key, value in state_dict.items()
                if key in model_state_dict and tuple(model_state_dict[key].shape) == tuple(value.shape)
            }
            if not compatible_state_dict:
                raise RuntimeError("Checkpoint did not contain compatible model weights.")
            model.load_state_dict(compatible_state_dict, strict=False)
            model.eval()
            self._model = model
            return model

    @staticmethod
    def _infer_num_classes_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int | None:
        candidate_keys = (
            "decode_head.classifier.weight",
            "classifier.weight",
            "head.weight",
            "outc.weight",
        )
        for key in candidate_keys:
            tensor = state_dict.get(key)
            if tensor is not None and getattr(tensor, "ndim", 0) >= 1:
                return int(tensor.shape[0])
        return None

    @staticmethod
    def _adapt_state_dict_to_model(
        state_dict: dict[str, torch.Tensor],
        model_state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        checkpoint_uses_legacy = any(key.startswith("segformer.stages.") for key in state_dict)
        model_uses_legacy = any(key.startswith("segformer.stages.") for key in model_state_dict)

        if checkpoint_uses_legacy == model_uses_legacy:
            return state_dict
        if checkpoint_uses_legacy and not model_uses_legacy:
            return TorchSegmentationPredictor._remap_legacy_to_encoder(state_dict)
        return TorchSegmentationPredictor._remap_encoder_to_legacy(state_dict)

    @staticmethod
    def _remap_legacy_to_encoder(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        replacements = [
            ("segformer.stages.", "segformer.encoder.block_stage_placeholder."),
            (".patch_embeddings.", ".patch_embeddings_stage_placeholder."),
            (".blocks.", ".block_stage_placeholder."),
            (".layernorm_before.", ".layer_norm_1."),
            (".layernorm_after.", ".layer_norm_2."),
            (".attention.q_proj.", ".attention.self.query."),
            (".attention.k_proj.", ".attention.self.key."),
            (".attention.v_proj.", ".attention.self.value."),
            (".attention.o_proj.", ".attention.output.dense."),
            (".attention.sequence_reduction.sequence_reduction.", ".attention.self.sr."),
            (".attention.sequence_reduction.layer_norm.", ".attention.self.layer_norm."),
            (".mlp.fc1.", ".mlp.dense1."),
            (".mlp.fc2.", ".mlp.dense2."),
            (".linear_projections.", ".linear_c."),
        ]

        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            for old, new in replacements:
                new_key = new_key.replace(old, new)

            if new_key.startswith("segformer.encoder.block_stage_placeholder."):
                new_key = new_key.replace("segformer.encoder.block_stage_placeholder.", "segformer.encoder.", 1)

            if ".patch_embeddings_stage_placeholder." in new_key:
                prefix, suffix = new_key.split(".patch_embeddings_stage_placeholder.", 1)
                stage_idx = prefix.rsplit(".", 1)[-1]
                prefix_root = prefix[: -(len(stage_idx) + 1)]
                new_key = f"{prefix_root}.patch_embeddings.{stage_idx}.{suffix}"

            if ".block_stage_placeholder." in new_key:
                prefix, suffix = new_key.split(".block_stage_placeholder.", 1)
                stage_idx = prefix.rsplit(".", 1)[-1]
                prefix_root = prefix[: -(len(stage_idx) + 1)]
                new_key = f"{prefix_root}.block.{stage_idx}.{suffix}"

            if "segformer.encoder.patch_embeddings.layer_norm." in new_key:
                prefix, suffix = new_key.split("segformer.encoder.patch_embeddings.layer_norm.", 1)
                stage_idx, remainder = suffix.split(".", 1)
                new_key = f"{prefix}segformer.encoder.patch_embeddings.{stage_idx}.layer_norm.{remainder}"

            if "attention.layer_norm.self." in new_key:
                new_key = new_key.replace("attention.layer_norm.self.", "attention.self.layer_norm.")

            if new_key.startswith("segformer.encoder.") and ".layer_norm." in new_key and "patch_embeddings." not in new_key:
                prefix, suffix = new_key.split(".layer_norm.", 1)
                tail = prefix.rsplit(".", 1)[-1]
                if tail.isdigit():
                    prefix_root = prefix[: -(len(tail) + 1)]
                    new_key = f"{prefix_root}.layer_norm.{tail}.{suffix}"

            remapped[new_key] = value
        return remapped

    @staticmethod
    def _remap_encoder_to_legacy(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key

            if new_key.startswith("segformer.encoder.patch_embeddings."):
                prefix, suffix = new_key.split("segformer.encoder.patch_embeddings.", 1)
                stage_idx, remainder = suffix.split(".", 1)
                new_key = f"{prefix}segformer.stages.{stage_idx}.patch_embeddings.{remainder}"
            elif new_key.startswith("segformer.encoder.block."):
                prefix, suffix = new_key.split("segformer.encoder.block.", 1)
                stage_idx, remainder = suffix.split(".", 1)
                block_idx, remainder = remainder.split(".", 1)
                new_key = f"{prefix}segformer.stages.{stage_idx}.blocks.{block_idx}.{remainder}"
            elif new_key.startswith("segformer.encoder.layer_norm."):
                prefix, suffix = new_key.split("segformer.encoder.layer_norm.", 1)
                stage_idx, remainder = suffix.split(".", 1)
                new_key = f"{prefix}segformer.stages.{stage_idx}.layer_norm.{remainder}"

            replacements = [
                (".layer_norm_1.", ".layernorm_before."),
                (".layer_norm_2.", ".layernorm_after."),
                (".attention.self.query.", ".attention.q_proj."),
                (".attention.self.key.", ".attention.k_proj."),
                (".attention.self.value.", ".attention.v_proj."),
                (".attention.output.dense.", ".attention.o_proj."),
                (".attention.self.sr.", ".attention.sequence_reduction.sequence_reduction."),
                (".attention.self.layer_norm.", ".attention.sequence_reduction.layer_norm."),
                (".mlp.dense1.", ".mlp.fc1."),
                (".mlp.dense2.", ".mlp.fc2."),
                (".linear_c.", ".linear_projections."),
            ]
            for old, new in replacements:
                new_key = new_key.replace(old, new)

            remapped[new_key] = value
        return remapped

    def _prepare_validation_tensor(self, image: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
        """Mirror research ``SegDataset.__getitem__`` for fixed-size validation inference."""
        original_gray = np.asarray(image.convert("L"), dtype=np.uint8)
        resized = Image.fromarray(original_gray, mode="L").resize(
            (self.img_size, self.img_size), resample=self.resize_interpolation
        )
        values = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(values).unsqueeze(0).repeat(3, 1, 1)
        if self.normalization_enabled:
            tensor = (tensor - self.normalization_mean) / self.normalization_std
        return original_gray, tensor.unsqueeze(0).to(self.device)

    @staticmethod
    def _compute_binary_entropy(probability: torch.Tensor) -> torch.Tensor:
        clamped = probability.clamp(min=1e-6, max=1.0 - 1e-6)
        entropy = -(clamped * torch.log(clamped) + (1.0 - clamped) * torch.log(1.0 - clamped))
        return torch.where(
            (probability <= 0.0) | (probability >= 1.0),
            torch.zeros_like(entropy),
            entropy,
        ) / np.log(2.0)

    @staticmethod
    def _build_predicted_boundary(mask: torch.Tensor, *, radius: int) -> torch.Tensor:
        if not mask.any():
            return torch.zeros_like(mask, dtype=torch.bool)
        safe_radius = max(1, int(radius))
        binary_mask = mask.float().unsqueeze(1)
        eroded = 1.0 - F.max_pool2d(
            1.0 - binary_mask,
            kernel_size=(2 * safe_radius) + 1,
            stride=1,
            padding=safe_radius,
        )
        return torch.logical_xor(mask, eroded[:, 0] > 0.5)

    def _predict_research_semantics(
        self, tensor: torch.Tensor
    ) -> tuple[np.ndarray, dict[str, object], dict[str, np.ndarray]]:
        # Mirrors evaluation's logits resize, pseudo-mask, and entropy semantics.
        model = self._load_model()
        with torch.inference_mode():
            logits = extract_logits(model, tensor, self.model_type)
            if logits.shape[-2:] != tensor.shape[-2:]:
                logits = F.interpolate(logits, size=tensor.shape[-2:], mode="bilinear", align_corners=False).contiguous()
            if self.num_classes != 1:
                raise ValueError("The packaged research adapter currently supports binary SegFormer outputs only.")
            probability = torch.sigmoid(logits[:, :1].contiguous())
            mask = (probability[:, 0] >= self.binary_threshold).to(dtype=torch.uint8)
            entropy = self._compute_binary_entropy(probability[:, 0])
            boundary_entropy = None
            boundary_pixel_count = None
            if self.boundary_enabled:
                boundary = self._build_predicted_boundary(mask > 0, radius=self.boundary_radius)
                boundary_pixel_count = int(boundary.sum().item())
                boundary_entropy = (entropy * boundary.float()).sum() / boundary.float().sum().clamp_min(1e-6)

            mask_map = mask[0].detach().cpu().numpy().astype(np.uint8)
            probability_map = probability[0, 0].detach().cpu().numpy().astype(np.float32)
            entropy_map = entropy[0].detach().cpu().numpy().astype(np.float32)
            summary = {
                "available": True,
                "global_entropy": float(entropy.mean().item()),
                "boundary_entropy": None if boundary_entropy is None else float(boundary_entropy.item()),
                "boundary_available": self.boundary_enabled,
                "boundary_pixel_count": boundary_pixel_count,
                "uncertain_pixel_ratio": float((entropy >= self.uncertainty_threshold).float().mean().item()),
                "predicted_tumor_ratio": float(mask.float().mean().item()),
                "mean_tumor_probability": float(probability.mean().item()),
            }
        return mask_map, summary, {"predictive_entropy": entropy_map, "tumor_probability": probability_map}

    @staticmethod
    def _resize_mask_for_display(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
        return np.asarray(
            Image.fromarray(mask.astype(np.uint8), mode="L").resize(
                (shape_hw[1], shape_hw[0]), resample=PIL_RESAMPLING.NEAREST
            ),
            dtype=np.uint8,
        )

    @staticmethod
    def _resize_map_for_display(values: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
        return np.asarray(
            Image.fromarray(values.astype(np.float32), mode="F").resize(
                (shape_hw[1], shape_hw[0]), resample=PIL_RESAMPLING.BILINEAR
            ),
            dtype=np.float32,
        )

    def _class_labels(self) -> dict[str, str]:
        if self.num_classes <= 1:
            return {"0": "background", "1": "tumor"}
        if self.num_classes == 2:
            return {"0": "background", "1": "tumor"}
        return {"0": "background", "1": "lesion", "2": "tumor"}

    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored[mask == 1] = (255, 191, 0)
        if self.num_classes > 2:
            colored[mask == 2] = (255, 64, 64)
        else:
            colored[mask == 1] = (255, 64, 64)
        return colored

    def predict(self, image: Image.Image) -> PredictionArtifacts:
        original_gray, tensor = self._prepare_validation_tensor(image)
        evaluation_mask, uncertainty_summary, uncertainty_maps = self._predict_research_semantics(tensor)
        pred_mask = self._resize_mask_for_display(evaluation_mask, original_gray.shape)

        original_rgb = gray_to_rgb(original_gray)
        color_mask = self._colorize_mask(pred_mask)
        overlay = blend_rgb(original_rgb, color_mask, alpha=0.35)

        extra_metadata = {
            "foreground_pixels": int((pred_mask > 0).sum()),
            "foreground_fraction": float((pred_mask > 0).mean()),
            "class_labels": self._class_labels(),
            "model_type": self.model_type,
            "threshold": self.binary_threshold,
            "threshold_source": self.threshold_source,
            "evaluation_shape_hw": [self.img_size, self.img_size],
            "uncertainty_summary": uncertainty_summary,
            "conformal_summary": None,
        }
        auxiliary_maps = {
            name: self._resize_map_for_display(values, original_gray.shape)
            for name, values in uncertainty_maps.items()
        }
        return PredictionArtifacts(
            mask=pred_mask,
            overlay=overlay,
            original=original_rgb,
            model_id=self.spec.model_id,
            checkpoint_path=self.spec.checkpoint_path,
            note=self.spec.note,
            extra_metadata=extra_metadata,
            auxiliary_maps=auxiliary_maps,
        )
