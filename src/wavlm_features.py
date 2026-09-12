import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WavLMModel

from audio_preprocessing import (
    AudioConfig,
    augment_audio,
    audio_config_dict,
    load_audio,
)


MODEL_ID = "microsoft/wavlm-base"
FEATURE_VERSION = "wavlm-stats-last4-v2"
DEFAULT_LAYERS = (-1, -2, -3, -4)


class WavLMFeatureExtractor:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: Optional[str] = None,
        layers: Iterable[int] = DEFAULT_LAYERS,
        audio_config: Optional[AudioConfig] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = tuple(layers)
        self.audio_config = audio_config or AudioConfig()
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def extract_from_path(
        self,
        audio_path: str | Path,
        *,
        augment: bool = False,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, Dict[str, object]]:
        audio = load_audio(audio_path, self.audio_config)
        samples = audio.samples

        if augment:
            rng = np.random.default_rng(seed)
            samples = augment_audio(samples, audio.sample_rate, rng)

        if not audio.valid:
            raise ValueError(audio.warning or "invalid audio")

        embedding = self.extract_from_samples(samples, audio.sample_rate)
        info = {
            "duration_seconds": audio.duration_seconds,
            "rms": audio.rms,
            "peak_before_normalization": audio.peak,
            "was_trimmed": audio.was_trimmed,
            "augmented": augment,
        }
        return embedding, info

    def extract_from_samples(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        inputs = self.processor(
            samples,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        selected = [outputs.hidden_states[layer] for layer in self.layers]
        pooled = [self._stats_pool(hidden, attention_mask) for hidden in selected]
        embedding = torch.cat(pooled, dim=1).squeeze(0).cpu().numpy()
        return embedding.astype(np.float32, copy=False)

    @staticmethod
    def _stats_pool(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            mean = hidden_states.mean(dim=1)
            std = hidden_states.std(dim=1, unbiased=False)
            return torch.cat([mean, std], dim=1)

        mask = attention_mask.float().unsqueeze(1)
        mask = F.interpolate(mask, size=hidden_states.shape[1], mode="nearest").squeeze(1)
        mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean = (hidden_states * mask).sum(dim=1) / denom
        variance = (((hidden_states - mean.unsqueeze(1)) * mask) ** 2).sum(dim=1) / denom
        std = torch.sqrt(variance.clamp_min(1e-12))
        return torch.cat([mean, std], dim=1)

    def config_dict(self) -> Dict[str, object]:
        return {
            "feature_version": FEATURE_VERSION,
            "model_id": self.model_id,
            "layers": list(self.layers),
            "pooling": "attention-mask-aware mean+std statistics pooling",
            "audio": audio_config_dict(self.audio_config),
        }


def cache_key(audio_path: str | Path, config: Dict[str, object], variant: str = "base") -> str:
    path = Path(audio_path)
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "variant": variant,
        "config": config,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cached_embedding(
    extractor: WavLMFeatureExtractor,
    audio_path: str | Path,
    cache_dir: str | Path,
    *,
    variant: str = "base",
    augment: bool = False,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, Dict[str, object], bool]:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    key = cache_key(audio_path, extractor.config_dict(), variant=variant)
    embedding_path = cache_root / f"{key}.npy"
    metadata_path = cache_root / f"{key}.json"

    if embedding_path.exists() and metadata_path.exists():
        embedding = np.load(embedding_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return embedding, metadata, True

    embedding, info = extractor.extract_from_path(audio_path, augment=augment, seed=seed)
    metadata = {
        "source_path": str(Path(audio_path)),
        "variant": variant,
        "feature_config": extractor.config_dict(),
        "audio_info": info,
    }
    np.save(embedding_path, embedding)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return embedding, metadata, False


def supported_audio_files(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    extensions = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions)
