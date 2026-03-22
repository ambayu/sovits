import gc
import inspect
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile
import torch

try:
    from inferrvc import RVC, ResampleCache, load_torchaudio
    from inferrvc.configs.config import Config as RVCConfig
    _INFERRVC_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local environment
    RVC = None
    ResampleCache = None
    load_torchaudio = None
    RVCConfig = None
    _INFERRVC_IMPORT_ERROR = exc


def _patch_inferrvc_runtime():
    if ResampleCache is None:
        return
    cache_cls = type(ResampleCache)
    if not getattr(cache_cls, "_sovits_dtype_patch", False):
        original_getitem = cache_cls.__getitem__

        def _patched_getitem(self, item):
            resampler = original_getitem(self, item)
            kernel = getattr(resampler, "kernel", None)
            if kernel is not None and kernel.dtype != torch.float32:
                resampler = resampler.float()
                self[item] = resampler
            return resampler

        def _patched_resample(self, fromto, audio, deviceto=None):
            target_device = deviceto if deviceto is not None else "cpu"
            if fromto[0] == fromto[1]:
                out = audio.to(target_device, non_blocking=True)
                out.frequency = fromto[1]
                return out
            resampler = self[fromto]
            kernel = getattr(resampler, "kernel", None)
            if kernel is not None:
                audio = audio.to(kernel.device, non_blocking=True)
                if audio.dtype != kernel.dtype:
                    audio = audio.to(kernel.dtype)
            out = resampler(audio).to(target_device, non_blocking=True)
            out.frequency = fromto[1]
            return out

        cache_cls.__getitem__ = _patched_getitem
        cache_cls.resample = _patched_resample
        cache_cls._sovits_dtype_patch = True

    if RVC is not None and not getattr(RVC, "_sovits_dtype_patch", False):
        for attr_name in ("_LOUD16K", "_LOUDOUTPUT"):
            transform = getattr(RVC, attr_name, None)
            if transform is not None:
                try:
                    setattr(RVC, attr_name, transform.float())
                except Exception:
                    pass
        RVC._sovits_dtype_patch = True


def _sanitize_stem(name):
    stem = str(name or "").strip().lower()
    out = []
    for ch in stem:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "rvc_model"


class RVCBackend:
    def __init__(self, model_root="logs/rvc_models"):
        self.model_root = Path(model_root)
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.engine = None
        self.model_path = None
        self.index_path = None
        self.device = "cpu"
        self.last_warning = ""
        self.chunk_seconds = 8.0
        self.chunk_overlap_seconds = 0.25
        self.session_invalid_reason = ""

    @property
    def is_available(self):
        return _INFERRVC_IMPORT_ERROR is None

    @property
    def dependency_error(self):
        if self.is_available:
            return ""
        return str(_INFERRVC_IMPORT_ERROR)

    def dependency_hint(self):
        if self.is_available:
            return "Dependency `inferrvc` siap digunakan."
        return (
            "Dependency `inferrvc` belum tersedia. Jalankan `pip install inferrvc` "
            f"lalu restart app. Detail: {self.dependency_error}"
        )

    def invalidate_session(self, reason):
        self.session_invalid_reason = str(reason or "").strip()

    def _clear_generic_state(self):
        free_generic_memory = getattr(RVC, "free_generic_memory", None) if RVC is not None else None
        if callable(free_generic_memory):
            try:
                free_generic_memory()
            except Exception:
                pass
        if ResampleCache is not None:
            try:
                ResampleCache.clear()
            except Exception:
                pass

    def _resolve_path(self, file_or_path):
        if file_or_path is None:
            return None
        if isinstance(file_or_path, str):
            return Path(file_or_path).expanduser()
        if hasattr(file_or_path, "name"):
            return Path(file_or_path.name).expanduser()
        raise ValueError("Format input path tidak dikenali.")

    def _normalize_device(self, device):
        val = str(device or "Auto")
        if val == "Auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if val == "CPU":
            return "cpu"
        low = val.lower()
        if low.startswith("cuda"):
            return low
        if "cuda" in low:
            return "cuda"
        return low

    def _safe_extract_zip(self, zip_path, out_dir):
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                member = Path(info.filename)
                if member.is_absolute() or ".." in member.parts:
                    continue
                target = (out_dir / member).resolve()
                if not str(target).startswith(str(out_dir)):
                    continue
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    def _pick_model_files(self, root_dir):
        root = Path(root_dir)
        pth_files = sorted(
            root.rglob("*.pth"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        index_files = sorted(root.rglob("*.index"))
        model_path = pth_files[0] if pth_files else None
        index_path = index_files[0] if index_files else None
        return model_path, index_path

    def _resolve_model_inputs(self, model_path, index_path):
        model_p = self._resolve_path(model_path)
        index_p = self._resolve_path(index_path)
        if model_p is None:
            raise ValueError("Model RVC belum dipilih.")
        if not model_p.exists():
            raise FileNotFoundError(f"File model tidak ditemukan: {model_p}")

        if model_p.suffix.lower() == ".zip":
            folder_name = _sanitize_stem(model_p.stem)
            extract_dir = self.model_root / folder_name
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            self._safe_extract_zip(model_p, extract_dir)
            model_from_zip, index_from_zip = self._pick_model_files(extract_dir)
            if model_from_zip is None:
                raise ValueError("ZIP model tidak berisi file `.pth`.")
            model_p = model_from_zip
            if index_p is None:
                index_p = index_from_zip
        elif model_p.is_dir():
            model_from_dir, index_from_dir = self._pick_model_files(model_p)
            if model_from_dir is None:
                raise ValueError("Folder model tidak berisi file `.pth`.")
            model_p = model_from_dir
            if index_p is None:
                index_p = index_from_dir
        elif model_p.suffix.lower() != ".pth":
            raise ValueError("Format model tidak valid. Gunakan `.pth`, `.zip`, atau folder model.")

        if index_p is not None and not index_p.exists():
            raise FileNotFoundError(f"File index tidak ditemukan: {index_p}")

        return model_p.resolve(), (index_p.resolve() if index_p else None)

    def _build_runtime_config(self):
        if RVCConfig is None:
            return None
        try:
            cfg = RVCConfig()
            target = str(self.device or "cpu").lower()
            if target.startswith("cpu"):
                cfg.device = "cpu"
                cfg.is_half = False
            elif target.startswith("cuda"):
                cfg.device = "cuda:0" if target == "cuda" else target
            else:
                cfg.device = target
            return cfg
        except Exception:
            return None

    def _create_engine(self, model_path, index_path):
        model_s = str(model_path)
        index_s = str(index_path) if index_path else None
        runtime_cfg = self._build_runtime_config()
        force_cpu = str(self.device or "").lower().startswith("cpu")
        attempts = []
        if index_s:
            attempts.append(lambda: RVC(model=model_s, index=index_s, config=runtime_cfg))
            if not force_cpu:
                attempts.extend(
                    [
                        lambda: RVC(model=model_s, index=index_s),
                        lambda: RVC(model_s, index=index_s),
                        lambda: RVC(model_s, index_s),
                    ]
                )
        else:
            attempts.append(lambda: RVC(model=model_s, config=runtime_cfg))
            if not force_cpu:
                attempts.extend(
                    [
                        lambda: RVC(model=model_s),
                        lambda: RVC(model_s),
                    ]
                )

        last_error = None
        for maker in attempts:
            try:
                return maker()
            except Exception as exc:  # pragma: no cover - depends on engine internals
                last_error = exc
        raise RuntimeError(f"Gagal inisialisasi engine RVC: {last_error}")

    def load_model(self, model_path, index_path=None, device="Auto"):
        if not self.is_available:
            raise RuntimeError(self.dependency_hint())
        if self.session_invalid_reason:
            raise RuntimeError(self.session_invalid_reason)
        _patch_inferrvc_runtime()

        resolved_model, resolved_index = self._resolve_model_inputs(model_path, index_path)
        if self.engine is not None:
            self.unload_model()
        else:
            self._clear_generic_state()

        self.device = self._normalize_device(device)
        try:
            self.engine = self._create_engine(resolved_model, resolved_index)
        except Exception:
            self._clear_generic_state()
            raise
        self.model_path = resolved_model
        self.index_path = resolved_index
        self.last_warning = ""
        if self.index_path is None:
            self.last_warning = "File `.index` tidak ditemukan. `index_rate` akan dipaksa menjadi 0."

        return {
            "status": "ok",
            "model_path": str(self.model_path),
            "index_path": str(self.index_path) if self.index_path else "",
            "device": self.device,
            "warning": self.last_warning,
        }

    def _load_audio(self, audio_path):
        if load_torchaudio is not None:
            return load_torchaudio(str(audio_path))
        wav, sr = soundfile.read(str(audio_path), always_2d=False)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        return torch.from_numpy(wav.astype(np.float32)), sr

    def _run_engine_path(self, audio_path, transpose=0, index_rate=0.75, protect=0.33):
        audio_p = self._resolve_path(audio_path)
        if audio_p is None or not audio_p.exists():
            raise FileNotFoundError("File audio input tidak ditemukan.")
        if self.index_path is None:
            index_rate = 0.0

        call_kwargs = {}
        try:
            params = inspect.signature(self.engine.__call__).parameters
        except Exception:
            params = {}

        if "f0_up_key" in params:
            call_kwargs["f0_up_key"] = int(transpose)
        if "index_rate" in params:
            call_kwargs["index_rate"] = float(index_rate)
        if "protect" in params:
            call_kwargs["protect"] = float(protect)
        if "output_device" in params:
            call_kwargs["output_device"] = "cpu"
        if "output_volume" in params and hasattr(RVC, "NO_CHANGE"):
            # MATCH_ORIGINAL can trigger CPU loudness ops that do not support Half tensors.
            call_kwargs["output_volume"] = RVC.NO_CHANGE

        cudnn_prev = torch.backends.cudnn.enabled
        # Some laptop GPUs hit CUDNN_STATUS_MAPPING_ERROR in torchaudio conv1d path.
        # Disabling cuDNN for this call trades speed for stability.
        torch.backends.cudnn.enabled = False
        try:
            try:
                result = self.engine(str(audio_p), **call_kwargs)
            except TypeError:
                result = self.engine(str(audio_p), int(transpose))
        finally:
            torch.backends.cudnn.enabled = cudnn_prev

        if isinstance(result, torch.Tensor):
            wav = result.detach().float().cpu().numpy()
        else:
            wav = np.asarray(result, dtype=np.float32)

        wav = np.squeeze(wav)
        if wav.ndim != 1:
            wav = wav.reshape(-1)
        out_sr = int(getattr(self.engine, "outputfreq", 44100) or 44100)
        return out_sr, wav.astype(np.float32)

    def _concat_with_crossfade(self, audio_parts, out_sr):
        if not audio_parts:
            return np.zeros(0, dtype=np.float32)
        out = np.asarray(audio_parts[0], dtype=np.float32)
        fade = int(out_sr * self.chunk_overlap_seconds)
        for current in audio_parts[1:]:
            current = np.asarray(current, dtype=np.float32)
            cross = min(fade, len(out), len(current))
            if cross <= 0:
                out = np.concatenate([out, current])
                continue
            ramp = np.linspace(0.0, 1.0, cross, dtype=np.float32)
            mixed = out[-cross:] * (1.0 - ramp) + current[:cross] * ramp
            out = np.concatenate([out[:-cross], mixed, current[cross:]])
        return out.astype(np.float32)

    def infer(self, input_audio_path, transpose=0, index_rate=0.75, protect=0.33):
        if self.engine is None:
            raise RuntimeError("Model RVC belum dimuat.")

        audio_path = self._resolve_path(input_audio_path)
        if audio_path is None or not audio_path.exists():
            raise FileNotFoundError("File audio input tidak ditemukan.")

        audio_tensor, source_sr = self._load_audio(audio_path)
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=0)
        audio_tensor = audio_tensor.float().cpu()

        chunk_size = int(source_sr * self.chunk_seconds)
        overlap_size = int(source_sr * self.chunk_overlap_seconds)
        if chunk_size <= 0 or audio_tensor.shape[-1] <= chunk_size:
            return self._run_engine_path(
                str(audio_path),
                transpose=transpose,
                index_rate=index_rate,
                protect=protect,
            )

        outputs = []
        step = max(1, chunk_size - overlap_size)
        out_sr = int(getattr(self.engine, "outputfreq", 44100) or 44100)
        temp_dir = self.model_root / "_tmp_chunks"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_paths = []
        try:
            for start in range(0, int(audio_tensor.shape[-1]), step):
                end = min(int(audio_tensor.shape[-1]), start + chunk_size)
                chunk = audio_tensor[start:end]
                if chunk.numel() == 0:
                    continue
                with tempfile.NamedTemporaryFile(
                    dir=str(temp_dir),
                    prefix="rvc_chunk_",
                    suffix=".wav",
                    delete=False,
                ) as tmp:
                    chunk_path = Path(tmp.name)
                temp_paths.append(chunk_path)
                soundfile.write(str(chunk_path), chunk.numpy(), source_sr, format="wav")
                _, chunk_out = self._run_engine_path(
                    str(chunk_path),
                    transpose=transpose,
                    index_rate=index_rate,
                    protect=protect,
                )
                outputs.append(chunk_out)
                if end >= int(audio_tensor.shape[-1]):
                    break
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        finally:
            for path in temp_paths:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass

        return out_sr, self._concat_with_crossfade(outputs, out_sr)

    def unload_model(self):
        if self.engine is None:
            return "Tidak ada model RVC yang perlu dicopot."

        for method_name in ("unload", "clear", "close"):
            fn = getattr(self.engine, method_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        self._clear_generic_state()
        self.engine = None
        self.model_path = None
        self.index_path = None
        self.last_warning = ""
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return "Model RVC berhasil dicopot."
