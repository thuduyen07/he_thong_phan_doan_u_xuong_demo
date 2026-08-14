from __future__ import annotations

import json
import threading

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from backend.inference.contracts import PredictionArtifacts
from backend.inference.device import resolve_inference_device
from backend.inference.predictors.base import SegmentationPredictor
from backend.inference.runtime_paths import RESOURCES_DIR, ensure_runtime_paths

ensure_runtime_paths()

from runtime_src.common.io_utils import load_yaml  # noqa: E402
from runtime_src.common.uncertainty import (  # noqa: E402
    build_binary_conformal_region_maps,
    binary_boundary_band,
    binary_probabilities_from_logits,
    sanitize_float,
    summarize_binary_conformal_regions,
    conformal_quantile_threshold,
    masked_fraction_below,
    masked_mean,
    masked_percentile,
    max_softmax_confidence,
    one_minus_margin,
    predictive_entropy,
    softmax_probabilities,
)
from runtime_src.seg.models import (  # noqa: E402
    build_segmentation_model,
    extract_logits,
    get_model_type,
)


class TorchSegmentationPredictor(SegmentationPredictor):
    def __init__(self, spec):
        super().__init__(spec)
        self.cfg = load_yaml(str(spec.config_path))
        if spec.config_overrides:
            self.cfg.update(spec.config_overrides)
        self._resolve_model_paths()
        self.device = resolve_inference_device(str(self.cfg.get("device", "cpu")))
        self.img_size = int(self.cfg["img_size"])
        self.num_classes = int(self.cfg["num_classes"])
        self.model_type = get_model_type(self.cfg)
        self._conformal_profile = self._load_conformal_profile()
        self._model = None
        self._model_lock = threading.Lock()

    def _resolve_model_paths(self) -> None:
        local_model_path = self.cfg.get("local_model_path")
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
        self.cfg["local_model_path"] = str(resolved)

    def _load_model(self):
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            checkpoint = torch.load(self.spec.checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            inferred_num_classes = self._infer_num_classes_from_state_dict(state_dict)
            if inferred_num_classes is not None and inferred_num_classes != self.num_classes:
                self.num_classes = inferred_num_classes
                self.cfg["num_classes"] = inferred_num_classes

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

    def _prepare_image(self, image: Image.Image):
        gray = np.array(image.convert("L"))
        metadata = {
            "original_shape_hw": [int(gray.shape[0]), int(gray.shape[1])],
            "processed_shape_hw": [int(gray.shape[0]), int(gray.shape[1])],
        }
        return gray, gray, metadata

    def _tumor_class_id(self) -> int:
        return 1 if self.num_classes <= 2 else self.num_classes - 1

    def _conformal_candidate_paths(self) -> list[Path]:
        return [
            RESOURCES_DIR / "conformal" / f"{self.spec.model_id}.json",
            self.spec.checkpoint_path.parent / "conformal_calibration.json",
            self.spec.checkpoint_path.parent / "conformal" / "conformal_calibration.json",
        ]

    def _load_conformal_profile(self) -> dict[str, object] | None:
        for candidate in self._conformal_candidate_paths():
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            payload["artifact_path"] = str(candidate)
            return payload
        return None

    def _compute_uncertainty_outputs(
        self,
        *,
        probs: np.ndarray,
        pred_mask: np.ndarray,
        boundary_width: int = 2,
        confidence_threshold: float = 0.75,
    ) -> tuple[dict[str, float | int | bool | None], dict[str, np.ndarray]]:
        tumor_class_id = self._tumor_class_id()
        pred_tumor = pred_mask == tumor_class_id
        boundary_mask = binary_boundary_band(pred_tumor, width=boundary_width)
        entropy_map = predictive_entropy(probs, class_axis=0, normalize=True)
        confidence_map = max_softmax_confidence(probs, class_axis=0)
        one_minus_msp_map = np.clip(1.0 - confidence_map, a_min=0.0, a_max=1.0)
        one_minus_margin_map = one_minus_margin(probs, class_axis=0)
        tumor_probability_map = np.clip(probs[tumor_class_id], a_min=0.0, a_max=1.0)

        summary = {
            "available": True,
            "tumor_class_id": int(tumor_class_id),
            "predicted_tumor_pixels": int(pred_tumor.sum()),
            "predicted_tumor_present": bool(pred_tumor.any()),
            "mean_predictive_entropy_pred_tumor": sanitize_float(masked_mean(entropy_map, pred_tumor)),
            "p90_predictive_entropy_pred_tumor": sanitize_float(
                masked_percentile(entropy_map, pred_tumor, 90.0)
            ),
            "mean_predictive_entropy_boundary_tumor": sanitize_float(
                masked_mean(entropy_map, boundary_mask)
            ),
            "mean_one_minus_msp_pred_tumor": sanitize_float(masked_mean(one_minus_msp_map, pred_tumor)),
            "mean_one_minus_margin_pred_tumor": sanitize_float(
                masked_mean(one_minus_margin_map, pred_tumor)
            ),
            "low_tumor_probability_fraction_pred_tumor": sanitize_float(
                masked_fraction_below(tumor_probability_map, pred_tumor, confidence_threshold)
            ),
            "mean_tumor_probability_pred_tumor": sanitize_float(masked_mean(tumor_probability_map, pred_tumor)),
        }
        maps = {
            "predictive_entropy": entropy_map.astype(np.float32),
            "one_minus_msp": one_minus_msp_map.astype(np.float32),
            "tumor_probability": tumor_probability_map.astype(np.float32),
        }
        return summary, maps

    def _compute_conformal_outputs(
        self,
        *,
        probs: np.ndarray,
    ) -> tuple[dict[str, object] | None, dict[str, np.ndarray]]:
        profile = self._conformal_profile
        if not profile:
            return None, {}

        tumor_class_id = self._tumor_class_id()
        alpha = float(profile.get("alpha", 0.1))
        lambda_threshold = profile.get("lambda_threshold")
        if lambda_threshold is None:
            probability_floor = profile.get("probability_floor")
            if probability_floor is not None:
                lambda_threshold = 1.0 - float(probability_floor)
        if lambda_threshold is None:
            calibration_scores = profile.get("calibration_scores")
            if calibration_scores:
                lambda_threshold = conformal_quantile_threshold(
                    np.asarray(calibration_scores, dtype=np.float64),
                    alpha=alpha,
                )
        if lambda_threshold is None:
            return {
                "available": False,
                "reason": "missing_lambda_threshold",
                "artifact_path": str(profile.get("artifact_path", "")),
            }, {}

        region_maps = build_binary_conformal_region_maps(
            probs,
            lambda_threshold=float(lambda_threshold),
            tumor_class_id=tumor_class_id,
            class_axis=0,
        )
        summary = summarize_binary_conformal_regions(
            region_maps,
            alpha=alpha,
            lambda_threshold=float(lambda_threshold),
            calibration_size=profile.get("calibration_size"),
            metadata={
                "method": profile.get("method", "split_conformal_prediction_sets"),
                "score_name": profile.get("score_name", "one_minus_true_class_probability"),
                "source_split": profile.get("source_split", "val"),
                "artifact_path": profile.get("artifact_path", ""),
                "empirical_coverage": profile.get("empirical_coverage"),
                "coverage_scope": profile.get("coverage_scope", "pixelwise"),
            },
        )
        maps = {
            "conformal_confident_tumor": region_maps["sure_tumor"].astype(np.float32),
            "conformal_outer_tumor": region_maps["outer_tumor"].astype(np.float32),
            "conformal_uncertain": region_maps["uncertain"].astype(np.float32),
        }
        return summary, maps

    def _predict_mask(
        self,
        processed: np.ndarray,
    ) -> tuple[
        np.ndarray,
        dict[str, float | int | bool | None],
        dict[str, np.ndarray],
        dict[str, object] | None,
        dict[str, np.ndarray],
    ]:
        resized = cv2.resize(processed, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(self.device)
        model = self._load_model()
        with torch.no_grad():
            logits = extract_logits(model, x, self.model_type)
            logits = torch.nn.functional.interpolate(
                logits,
                size=(processed.shape[0], processed.shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            logits_np = logits.squeeze(0).detach().cpu().numpy()
            if logits.shape[1] == 1:
                probs = binary_probabilities_from_logits(logits_np)
                pred = (probs[1] > 0.5).astype(np.uint8)
            else:
                probs = softmax_probabilities(logits_np, class_axis=0)
                pred = np.argmax(probs, axis=0).astype(np.uint8)
        uncertainty_summary, uncertainty_maps = self._compute_uncertainty_outputs(probs=probs, pred_mask=pred)
        conformal_summary, conformal_maps = self._compute_conformal_outputs(probs=probs)
        return pred, uncertainty_summary, uncertainty_maps, conformal_summary, conformal_maps

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
        original_gray, processed, prep_metadata = self._prepare_image(image)
        pred_mask, uncertainty_summary, uncertainty_maps, conformal_summary, conformal_maps = self._predict_mask(processed)

        original_rgb = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2RGB)
        color_mask = self._colorize_mask(pred_mask)
        overlay = cv2.addWeighted(original_rgb, 0.65, color_mask, 0.35, 0.0)

        extra_metadata = {
            **prep_metadata,
            "foreground_pixels": int((pred_mask > 0).sum()),
            "class_labels": self._class_labels(),
            "model_type": self.model_type,
            "uncertainty_summary": uncertainty_summary,
            "conformal_summary": conformal_summary,
        }
        auxiliary_maps = {
            **uncertainty_maps,
            **conformal_maps,
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
