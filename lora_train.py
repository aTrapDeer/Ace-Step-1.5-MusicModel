"""
ACE-Step 1.5 LoRA Training Engine

Handles dataset building, VAE encoding, and flow-matching LoRA training
of the DiT decoder. Designed to work with the existing AceStepHandler.
"""

import os
import sys
import json
import math
import time
import random
import hashlib
import argparse
import tempfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import numpy as np
from loguru import logger
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}


@dataclass
class TrackEntry:
    """One audio file + its metadata."""

    audio_path: str
    caption: str = ""
    lyrics: str = ""
    bpm: Optional[int] = None
    keyscale: str = ""
    timesignature: str = "4/4"
    vocal_language: str = "en"
    duration: Optional[float] = None  # seconds (measured at scan time)


def _load_track_entry(audio_path: Path) -> TrackEntry:
    """Load one track + optional sidecar metadata."""
    sidecar = audio_path.with_suffix(".json")
    meta: Dict[str, Any] = {}
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Bad sidecar {sidecar}: {exc}")

    try:
        info = torchaudio.info(str(audio_path))
        duration = info.num_frames / info.sample_rate
    except Exception:
        duration = meta.get("duration")

    return TrackEntry(
        audio_path=str(audio_path),
        caption=meta.get("caption", ""),
        lyrics=meta.get("lyrics", ""),
        bpm=meta.get("bpm"),
        keyscale=meta.get("keyscale", ""),
        timesignature=meta.get("timesignature", "4/4"),
        vocal_language=meta.get("vocal_language", "en"),
        duration=duration,
    )


def scan_dataset_folder(folder: str) -> List[TrackEntry]:
    """Scan *folder* for audio files and optional JSON sidecars.

    For every ``track.wav`` found, if ``track.json`` exists next to it the
    metadata fields are loaded from the sidecar.  Missing sidecars are fine –
    the entry will have empty metadata that can be filled later.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    entries: List[TrackEntry] = []
    for audio_path in sorted(folder.rglob("*")):
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        entries.append(_load_track_entry(audio_path))

    logger.info(f"Scanned {len(entries)} audio files in {folder}")
    return entries


def scan_uploaded_files(file_paths: List[str]) -> List[TrackEntry]:
    """Build entries from dropped/uploaded files."""
    entries: List[TrackEntry] = []
    for path in file_paths:
        p = Path(path)
        if not p.exists():
            continue
        if p.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        entries.append(_load_track_entry(p))
    logger.info(f"Loaded {len(entries)} uploaded audio files")
    return entries


# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------


@dataclass
class LoRATrainConfig:
    """All tuneable knobs for a LoRA run."""

    # LoRA architecture
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Optimiser
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = "adamw_8bit"  # "adamw" | "adamw_8bit"
    max_grad_norm: float = 1.0

    # Schedule
    warmup_ratio: float = 0.03
    scheduler: str = "constant_with_warmup"

    # Training loop
    num_epochs: int = 50
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    save_every_n_epochs: int = 10
    log_every_n_steps: int = 5

    # Flow matching
    shift: float = 3.0  # timestep shift factor

    # Audio pre-processing
    max_duration_sec: float = 240.0  # clamp audio to this length
    sample_rate: int = 48000

    # Paths
    output_dir: str = "lora_output"
    resume_from: Optional[str] = None

    # Device
    device: str = "auto"
    dtype: str = "bf16"  # "bf16" | "fp16" | "fp32"
    mixed_precision: bool = True


# ---------------------------------------------------------------------------
# Core trainer
# ---------------------------------------------------------------------------


class LoRATrainer:
    """Thin training loop that wraps the existing AceStepHandler."""

    def __init__(self, handler, config: LoRATrainConfig):
        """
        Args:
            handler: Initialised ``AceStepHandler`` (model, vae, text_encoder loaded).
            config:  Training hyper-parameters.
        """
        self.handler = handler
        self.cfg = config

        self.device = handler.device
        self.dtype = handler.dtype

        # Will be set during prepare()
        self.peft_model = None
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.current_epoch = 0

        # Loss history for UI
        self.loss_history: List[Dict[str, Any]] = []
        self._stop_requested = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_lora_target_modules(model, requested_targets: Optional[List[str]]) -> List[str]:
        """Resolve LoRA target module suffixes against the actual decoder module names."""
        linear_module_names = [
            name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
        ]

        def _exists_as_suffix(target: str) -> bool:
            return any(name.endswith(target) for name in linear_module_names)

        requested_targets = requested_targets or []
        resolved = [target for target in requested_targets if _exists_as_suffix(target)]
        if resolved:
            return resolved

        fallback_groups = [
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            ["to_q", "to_k", "to_v", "to_out.0"],
            ["query", "key", "value", "out_proj"],
            ["wq", "wk", "wv", "wo"],
            ["qkv", "proj_out"],
        ]
        for group in fallback_groups:
            group_resolved = [target for target in group if _exists_as_suffix(target)]
            if len(group_resolved) >= 2:
                return group_resolved

        sample = ", ".join(linear_module_names[:30])
        raise ValueError(
            "Could not find LoRA target modules in decoder. "
            f"Requested={requested_targets}. "
            f"Sample linear modules: {sample}"
        )

    def prepare(self):
        """Attach LoRA adapters to the decoder and build the optimiser."""
        import copy
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        # Keep a backup of the plain base decoder so load/unload logic remains valid.
        if self.handler._base_decoder is None:
            self.handler._base_decoder = copy.deepcopy(self.handler.model.decoder)
        else:
            self.handler.model.decoder = copy.deepcopy(self.handler._base_decoder)
            self.handler.model.decoder = self.handler.model.decoder.to(self.device).to(self.dtype)
            self.handler.model.decoder.eval()

        resume_adapter = None
        if self.cfg.resume_from:
            adapter_cfg = os.path.join(self.cfg.resume_from, "adapter_config.json")
            if os.path.isfile(adapter_cfg):
                resume_adapter = self.cfg.resume_from

        if resume_adapter:
            logger.info(f"Loading existing LoRA adapter for resume: {resume_adapter}")
            self.peft_model = PeftModel.from_pretrained(
                self.handler.model.decoder,
                resume_adapter,
                is_trainable=True,
            )
        else:
            resolved_targets = self._resolve_lora_target_modules(
                self.handler.model.decoder,
                self.cfg.lora_target_modules,
            )
            logger.info(f"Using LoRA target modules: {resolved_targets}")
            peft_cfg = LoraConfig(
                r=self.cfg.lora_rank,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=resolved_targets,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.peft_model = get_peft_model(self.handler.model.decoder, peft_cfg)

        self.peft_model.print_trainable_parameters()
        self.handler.model.decoder = self.peft_model
        self.handler.model.decoder.to(self.device).to(self.dtype)
        self.handler.model.decoder.train()
        self.handler.lora_loaded = True
        self.handler.use_lora = True

        # Build optimiser (only LoRA params are trainable)
        trainable_params = [p for p in self.peft_model.parameters() if p.requires_grad]
        if self.cfg.optimizer == "adamw_8bit":
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=self.cfg.learning_rate,
                    weight_decay=self.cfg.weight_decay,
                )
            except ImportError:
                logger.warning("bitsandbytes not found – falling back to standard AdamW")
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.cfg.learning_rate,
                    weight_decay=self.cfg.weight_decay,
                )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
            )

        # Resume checkpoint state (after model/adapter restore).
        if self.cfg.resume_from and os.path.isfile(
            os.path.join(self.cfg.resume_from, "training_state.pt")
        ):
            state = torch.load(
                os.path.join(self.cfg.resume_from, "training_state.pt"),
                weights_only=False,
            )
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception as exc:
                logger.warning(f"Could not restore optimizer state, continuing fresh optimizer: {exc}")
            self.global_step = int(state.get("global_step", 0))
            # Saved epoch is completed epoch index; continue from next epoch.
            self.current_epoch = int(state.get("epoch", -1)) + 1
            loss_path = os.path.join(self.cfg.resume_from, "loss_history.json")
            if os.path.isfile(loss_path):
                try:
                    with open(loss_path, "r", encoding="utf-8") as f:
                        self.loss_history = json.load(f)
                except Exception:
                    pass
            logger.info(
                f"Resumed from {self.cfg.resume_from} "
                f"(epoch {self.current_epoch}, step {self.global_step})"
            )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load audio, resample to 48 kHz stereo, clamp to max_duration."""
        try:
            wav, sr = torchaudio.load(path)
        except Exception as torchaudio_exc:
            # torchaudio on some Space images requires torchcodec for decode.
            # Fallback to soundfile so training can proceed without torchcodec.
            try:
                audio_np, sr = sf.read(path, dtype="float32", always_2d=True)
                wav = torch.from_numpy(audio_np.T)
            except Exception as sf_exc:
                raise RuntimeError(
                    f"Failed to decode audio '{path}' with torchaudio ({torchaudio_exc}) "
                    f"and soundfile ({sf_exc})."
                ) from sf_exc

        # Resample if needed
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)

        # Convert mono → stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        # Clamp length
        max_samples = int(self.cfg.max_duration_sec * self.cfg.sample_rate)
        if wav.shape[1] > max_samples:
            wav = wav[:, :max_samples]

        return wav  # [2, T]

    def _encode_audio(self, wav: torch.Tensor) -> torch.Tensor:
        """Encode raw waveform → VAE latent on device."""
        with torch.no_grad():
            latent = self.handler._encode_audio_to_latents(wav)
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
        latent = latent.to(self.dtype)
        return latent

    def _build_text_embeddings(self, caption: str, lyrics: str):
        """Compute text & lyric embeddings using the text encoder."""
        tokenizer = self.handler.text_tokenizer
        text_encoder = self.handler.text_encoder

        # Caption embedding
        text_tokens = tokenizer(
            caption or "",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            text_hidden = text_encoder(
                input_ids=text_tokens["input_ids"]
            ).last_hidden_state.to(self.dtype)
            text_mask = text_tokens["attention_mask"].to(self.dtype)

        # Lyric embedding (token-level via embed_tokens)
        lyric_tokens = tokenizer(
            lyrics or "",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            lyric_hidden = text_encoder.embed_tokens(
                lyric_tokens["input_ids"]
            ).to(self.dtype)
            lyric_mask = lyric_tokens["attention_mask"].to(self.dtype)

        return text_hidden, text_mask, lyric_hidden, lyric_mask

    # ------------------------------------------------------------------
    # Flow matching loss
    # ------------------------------------------------------------------

    def _flow_matching_loss(
        self,
        x1: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rectified-flow MSE loss for one sample.

        Notation follows ACE-Step convention:
          x0 = noise,  x1 = clean latent
          xt = t * x0 + (1 - t) * x1
          target velocity  = x0 - x1
        """
        bsz = x1.shape[0]

        # Sample random timestep per element
        t = torch.rand(bsz, device=self.device, dtype=self.dtype)

        # Apply timestep shift: t_shifted = shift * t / (1 + (shift - 1) * t)
        if self.cfg.shift != 1.0:
            t = self.cfg.shift * t / (1.0 + (self.cfg.shift - 1.0) * t)

        t = t.clamp(1e-5, 1.0 - 1e-5)

        # Noise
        x0 = torch.randn_like(x1)

        # Interpolate
        t_expand = t.view(bsz, 1, 1)
        xt = t_expand * x0 + (1.0 - t_expand) * x1

        # Target velocity
        velocity_target = x0 - x1

        # Attention mask
        attention_mask = torch.ones(
            bsz, x1.shape[1], device=self.device, dtype=self.dtype
        )

        # Forward through decoder
        decoder_out = self.handler.model.decoder(
            hidden_states=xt,
            timestep=t,
            timestep_r=t,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            use_cache=False,
            output_attentions=False,
        )

        velocity_pred = decoder_out[0]  # first element is the predicted output
        loss = F.mse_loss(velocity_pred, velocity_target)
        return loss

    @staticmethod
    def _pad_and_stack(tensors: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
        """Pad variable-length tensors on dimension 0 and stack as batch."""
        normalized = []
        for t in tensors:
            if t.dim() >= 2 and t.shape[0] == 1:
                normalized.append(t.squeeze(0))
            else:
                normalized.append(t)

        max_len = max(t.shape[0] for t in normalized)
        template = normalized[0]
        out_shape = (len(normalized), max_len, *template.shape[1:])
        out = template.new_full(out_shape, pad_value)
        for i, t in enumerate(normalized):
            out[i, : t.shape[0]] = t
        return out

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def request_stop(self):
        """Ask the training loop to stop after the current step."""
        self._stop_requested = True

    def train(
        self,
        entries: List[TrackEntry],
        progress_callback=None,
    ) -> str:
        """Run the full LoRA training.

        Args:
            entries: List of scanned TrackEntry objects.
            progress_callback: ``fn(step, total_steps, loss, epoch)`` for UI updates.

        Returns:
            Status message.
        """
        self._stop_requested = False
        self.loss_history.clear()
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        if not entries:
            return "No training data provided."

        num_entries = len(entries)
        total_steps = (
            math.ceil(num_entries / self.cfg.batch_size)
            * self.cfg.num_epochs
        )

        # ---- Pre-encode all audio & text (fits in CPU RAM) ----
        logger.info("Pre-encoding dataset through VAE & text encoder ...")
        dataset: List[Dict[str, Any]] = []

        # Freeze VAE and text encoder (they are not trained)
        self.handler.vae.eval()
        self.handler.text_encoder.eval()

        # Reuse silence reference latent (same as handler's internal fallback path).
        ref_latent = self.handler.silence_latent[:, :750, :].to(self.device).to(self.dtype)
        ref_order_mask = torch.zeros(1, device=self.device, dtype=torch.long)

        for idx, entry in enumerate(tqdm(entries, desc="Encoding dataset")):
            try:
                wav = self._load_audio(entry.audio_path)
                latent = self._encode_audio(wav)
                text_h, text_m, lyric_h, lyric_m = self._build_text_embeddings(
                    entry.caption, entry.lyrics
                )

                # Prepare condition using the model's own prepare_condition
                with torch.no_grad():
                    enc_hs, enc_mask, ctx_lat = self.handler.model.prepare_condition(
                        text_hidden_states=text_h,
                        text_attention_mask=text_m,
                        lyric_hidden_states=lyric_h,
                        lyric_attention_mask=lyric_m,
                        refer_audio_acoustic_hidden_states_packed=ref_latent,
                        refer_audio_order_mask=ref_order_mask,
                        hidden_states=latent,
                        attention_mask=torch.ones(
                            1, latent.shape[1],
                            device=self.device, dtype=self.dtype,
                        ),
                        silence_latent=self.handler.silence_latent,
                        src_latents=latent,
                        chunk_masks=torch.ones_like(latent),
                        is_covers=[False],
                    )

                dataset.append(
                    {
                        "latent": latent.cpu(),
                        "enc_hs": enc_hs.cpu(),
                        "enc_mask": enc_mask.cpu(),
                        "ctx_lat": ctx_lat.cpu(),
                        "name": Path(entry.audio_path).stem,
                    }
                )
            except Exception as exc:
                logger.warning(f"Skipping {entry.audio_path}: {exc}")

        if not dataset:
            return "All tracks failed to encode. Check audio files."

        logger.info(f"Encoded {len(dataset)}/{num_entries} tracks.")

        # ---- Warmup scheduler ----
        total_optim_steps = math.ceil(
            total_steps / self.cfg.gradient_accumulation_steps
        )
        warmup_steps = int(total_optim_steps * self.cfg.warmup_ratio)

        if self.cfg.scheduler in {"constant_with_warmup", "linear", "cosine"}:
            try:
                from transformers import get_scheduler
                self.scheduler = get_scheduler(
                    name=self.cfg.scheduler,
                    optimizer=self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_optim_steps,
                )
            except Exception as exc:
                logger.warning(f"Could not create scheduler '{self.cfg.scheduler}', disabling scheduler: {exc}")
                self.scheduler = None
        else:
            self.scheduler = None

        # ---- Training loop ----
        logger.info(
            f"Starting LoRA training: {self.cfg.num_epochs} epochs, "
            f"{len(dataset)} samples, {total_optim_steps} optimiser steps"
        )

        self.peft_model.train()
        accum_loss = 0.0
        step_in_accum = 0

        for epoch in range(self.current_epoch, self.cfg.num_epochs):
            if self._stop_requested:
                break

            self.current_epoch = epoch
            indices = list(range(len(dataset)))
            random.shuffle(indices)

            epoch_loss = 0.0
            epoch_steps = 0

            for i in range(0, len(indices), self.cfg.batch_size):
                if self._stop_requested:
                    break

                batch_indices = indices[i : i + self.cfg.batch_size]
                batch_items = [dataset[j] for j in batch_indices]

                # Move batch to device
                latents = self._pad_and_stack([it["latent"] for it in batch_items]).to(self.device, self.dtype)
                enc_hs = self._pad_and_stack([it["enc_hs"] for it in batch_items]).to(self.device, self.dtype)
                enc_mask = self._pad_and_stack([it["enc_mask"] for it in batch_items], pad_value=0.0).to(self.device)
                if enc_mask.dtype != self.dtype:
                    enc_mask = enc_mask.to(self.dtype)
                ctx_lat = self._pad_and_stack([it["ctx_lat"] for it in batch_items]).to(self.device, self.dtype)

                # Forward + loss
                loss = self._flow_matching_loss(latents, enc_hs, enc_mask, ctx_lat)
                loss = loss / self.cfg.gradient_accumulation_steps
                loss.backward()

                accum_loss += loss.item()
                step_in_accum += 1

                if step_in_accum >= self.cfg.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(
                        self.peft_model.parameters(), self.cfg.max_grad_norm
                    )
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    avg_loss = accum_loss
                    accum_loss = 0.0
                    step_in_accum = 0

                    self.loss_history.append(
                        {
                            "step": self.global_step,
                            "epoch": epoch,
                            "loss": avg_loss,
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }
                    )

                    if self.global_step % self.cfg.log_every_n_steps == 0:
                        logger.info(
                            f"Epoch {epoch+1}/{self.cfg.num_epochs}  "
                            f"Step {self.global_step}/{total_optim_steps}  "
                            f"Loss {avg_loss:.6f}  "
                            f"LR {self.optimizer.param_groups[0]['lr']:.2e}"
                        )

                    if progress_callback:
                        progress_callback(
                            self.global_step, total_optim_steps, avg_loss, epoch
                        )

                epoch_loss += loss.item() * self.cfg.gradient_accumulation_steps
                epoch_steps += 1

            # Flush remaining micro-batches when len(dataset) is not divisible by grad accumulation.
            if step_in_accum > 0:
                torch.nn.utils.clip_grad_norm_(self.peft_model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                avg_loss = accum_loss
                accum_loss = 0.0
                step_in_accum = 0
                self.loss_history.append(
                    {
                        "step": self.global_step,
                        "epoch": epoch,
                        "loss": avg_loss,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # End of epoch – checkpoint?
            if (
                (epoch + 1) % self.cfg.save_every_n_epochs == 0
                or epoch == self.cfg.num_epochs - 1
                or self._stop_requested
            ):
                self._save_checkpoint(epoch)

            if epoch_steps > 0:
                avg_epoch_loss = epoch_loss / epoch_steps
                logger.info(
                    f"Epoch {epoch+1} complete – avg loss {avg_epoch_loss:.6f}"
                )

        # Final save
        final_dir = self._save_checkpoint(self.current_epoch, final=True)
        status = (
            "Training stopped early." if self._stop_requested else "Training complete!"
        )
        return f"{status} Adapter saved to {final_dir}"

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, final: bool = False) -> str:
        tag = "final" if final else f"epoch-{epoch+1}"
        save_dir = os.path.join(self.cfg.output_dir, tag)
        os.makedirs(save_dir, exist_ok=True)

        # Save PEFT adapter
        self.peft_model.save_pretrained(save_dir)

        # Save training state
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "epoch": epoch,
            },
            os.path.join(save_dir, "training_state.pt"),
        )

        # Save loss curve
        loss_path = os.path.join(save_dir, "loss_history.json")
        with open(loss_path, "w") as f:
            json.dump(self.loss_history, f)

        # Save config
        cfg_path = os.path.join(save_dir, "train_config.json")
        with open(cfg_path, "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)

        logger.info(f"Checkpoint saved → {save_dir}")
        return save_dir

    # ------------------------------------------------------------------
    # Adapter listing
    # ------------------------------------------------------------------

    @staticmethod
    def list_adapters(output_dir: str = "lora_output") -> List[str]:
        """Return adapter directories inside *output_dir* (recursive)."""
        results = []
        root = Path(output_dir)
        if not root.is_dir():
            return results
        for cfg in sorted(root.rglob("adapter_config.json")):
            d = cfg.parent
            if d.is_dir():
                results.append(str(d))
        return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ACE-Step 1.5 LoRA trainer (CLI)")

    # Dataset
    parser.add_argument("--dataset-dir", type=str, default="", help="Local dataset folder path")
    parser.add_argument("--dataset-repo", type=str, default="", help="HF dataset repo id (optional)")
    parser.add_argument("--dataset-revision", type=str, default="main", help="HF dataset revision")
    parser.add_argument("--dataset-subdir", type=str, default="", help="Subdirectory inside downloaded dataset")

    # Model init
    parser.add_argument("--model-config", type=str, default="acestep-v15-base", help="DiT config name")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "xpu", "cpu"])
    parser.add_argument("--offload-to-cpu", action="store_true")
    parser.add_argument("--offload-dit-to-cpu", action="store_true")
    parser.add_argument("--prefer-source", type=str, default="huggingface", choices=["huggingface", "modelscope"])

    # Train config
    parser.add_argument("--output-dir", type=str, default="lora_output")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--max-duration-sec", type=float, default=240.0)

    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adamw_8bit", choices=["adamw", "adamw_8bit"])
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--scheduler", type=str, default="constant_with_warmup", choices=["constant_with_warmup", "linear", "cosine"])
    parser.add_argument("--shift", type=float, default=3.0)

    # Optional upload
    parser.add_argument("--upload-repo", type=str, default="", help="HF model repo to upload final adapter")
    parser.add_argument("--upload-path", type=str, default="", help="Path inside upload repo (optional)")
    parser.add_argument("--upload-private", action="store_true")
    parser.add_argument("--hf-token-env", type=str, default="HF_TOKEN", help="Environment variable name for HF token")

    return parser


def _resolve_dataset_dir(args) -> str:
    if args.dataset_dir:
        return args.dataset_dir

    if not args.dataset_repo:
        raise ValueError("Provide --dataset-dir or --dataset-repo.")

    from huggingface_hub import snapshot_download

    token = os.getenv(args.hf_token_env)
    temp_root = tempfile.mkdtemp(prefix="acestep_lora_dataset_")
    local_dir = os.path.join(temp_root, "dataset")
    logger.info(f"Downloading dataset repo {args.dataset_repo}@{args.dataset_revision} to {local_dir}")
    snapshot_download(
        repo_id=args.dataset_repo,
        repo_type="dataset",
        revision=args.dataset_revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token,
    )
    if args.dataset_subdir:
        sub = os.path.join(local_dir, args.dataset_subdir)
        if not os.path.isdir(sub):
            raise FileNotFoundError(f"Dataset subdir not found: {sub}")
        return sub
    return local_dir


def _upload_adapter_if_requested(args, final_dir: str):
    if not args.upload_repo:
        return

    from huggingface_hub import HfApi

    token = os.getenv(args.hf_token_env)
    if not token:
        raise RuntimeError(
            f"{args.hf_token_env} is not set. Needed for upload to {args.upload_repo}."
        )

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.upload_repo,
        repo_type="model",
        exist_ok=True,
        private=bool(args.upload_private),
    )

    path_in_repo = args.upload_path.strip().strip("/") if args.upload_path else ""
    commit_message = f"Upload ACE-Step LoRA adapter from {Path(final_dir).name}"
    logger.info(f"Uploading adapter from {final_dir} to {args.upload_repo}/{path_in_repo}")
    api.upload_folder(
        repo_id=args.upload_repo,
        repo_type="model",
        folder_path=final_dir,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
    logger.info("Upload complete")


def main():
    args = _build_arg_parser().parse_args()

    dataset_dir = _resolve_dataset_dir(args)
    entries = scan_dataset_folder(dataset_dir)
    if not entries:
        raise RuntimeError(f"No audio files found in dataset: {dataset_dir}")

    from acestep.handler import AceStepHandler

    project_root = str(Path(__file__).resolve().parent)
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=project_root,
        config_path=args.model_config,
        device=args.device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=bool(args.offload_to_cpu),
        offload_dit_to_cpu=bool(args.offload_dit_to_cpu),
        prefer_source=args.prefer_source,
    )
    print(status)
    if not ok:
        raise RuntimeError("Model initialization failed")

    cfg = LoRATrainConfig(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        scheduler=args.scheduler,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        save_every_n_epochs=args.save_every,
        log_every_n_steps=args.log_every,
        shift=args.shift,
        max_duration_sec=args.max_duration_sec,
        output_dir=args.output_dir,
        resume_from=(args.resume_from.strip() if args.resume_from else None),
        device=args.device,
    )

    trainer = LoRATrainer(handler, cfg)
    trainer.prepare()

    start = time.time()

    def _progress(step, total, loss, epoch):
        elapsed = time.time() - start
        rate = step / elapsed if elapsed > 0 else 0.0
        remaining = max(0.0, total - step)
        eta_sec = remaining / rate if rate > 0 else -1.0
        eta_msg = f"{eta_sec/60:.1f}m" if eta_sec >= 0 else "unknown"
        logger.info(
            f"[progress] step={step}/{total} epoch={epoch+1} loss={loss:.6f} elapsed={elapsed/60:.1f}m eta={eta_msg}"
        )

    msg = trainer.train(entries, progress_callback=_progress)
    print(msg)

    final_dir = os.path.join(cfg.output_dir, "final")
    if os.path.isdir(final_dir):
        _upload_adapter_if_requested(args, final_dir)
        print(f"Final adapter directory: {final_dir}")
    else:
        print(f"Warning: final adapter directory not found at {final_dir}")


if __name__ == "__main__":
    main()
