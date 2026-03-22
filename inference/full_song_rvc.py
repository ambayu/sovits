import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile


def demucs_is_available():
    return importlib.util.find_spec("demucs") is not None


def demucs_dependency_hint():
    if demucs_is_available():
        return "Dependency `demucs` siap digunakan."
    return "Dependency `demucs` belum tersedia. Jalankan `pip install demucs` lalu restart app."


def _normalize_device(device):
    val = str(device or "Auto")
    if val == "CPU":
        return "cpu"
    if val == "Auto":
        return "cuda"
    low = val.lower()
    if low.startswith("cuda") or "cuda" in low:
        return "cuda"
    return "cpu"


def _sanitize_stem(name):
    stem = str(name or "").strip().lower()
    out = []
    for ch in stem:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "audio"


def separate_song_with_demucs(input_audio_path, output_root, device="Auto", model_name="htdemucs"):
    if not demucs_is_available():
        raise RuntimeError(demucs_dependency_hint())

    input_path = Path(input_audio_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"File audio tidak ditemukan: {input_path}")

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"{_sanitize_stem(input_path.stem)}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / "_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        model_name,
        "--two-stems",
        "vocals",
        "-o",
        str(run_dir),
        "-d",
        _normalize_device(device),
        str(input_path),
    ]
    env = dict(os.environ)
    env["TORCH_HOME"] = str(cache_root / "torch")
    env["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Demucs gagal memisahkan audio. {detail}")

    target_dir = run_dir / model_name / input_path.stem
    vocals = target_dir / "vocals.wav"
    instrumental = target_dir / "no_vocals.wav"
    if not vocals.exists() or not instrumental.exists():
        vocals_candidates = list(run_dir.rglob("vocals.wav"))
        instrumental_candidates = list(run_dir.rglob("no_vocals.wav"))
        vocals = vocals_candidates[0] if vocals_candidates else vocals
        instrumental = instrumental_candidates[0] if instrumental_candidates else instrumental
    if not vocals.exists() or not instrumental.exists():
        raise RuntimeError("Output stem Demucs tidak ditemukan (`vocals.wav` / `no_vocals.wav`).")

    return {
        "run_dir": run_dir,
        "vocals_path": vocals,
        "instrumental_path": instrumental,
        "stdout": (result.stdout or "").strip(),
        "stderr": (result.stderr or "").strip(),
    }


def remix_vocals_with_instrumental(
    instrumental_path,
    converted_vocal_path,
    output_path,
    vocal_gain=1.0,
    instrumental_gain=1.0,
    normalize_output=True,
):
    inst_audio, inst_sr = librosa.load(str(instrumental_path), sr=None, mono=False)
    vocal_audio, vocal_sr = librosa.load(str(converted_vocal_path), sr=None, mono=False)

    target_sr = max(int(inst_sr), int(vocal_sr))
    if int(inst_sr) != target_sr:
        inst_audio = librosa.resample(inst_audio, orig_sr=inst_sr, target_sr=target_sr, axis=-1)
    if int(vocal_sr) != target_sr:
        vocal_audio = librosa.resample(vocal_audio, orig_sr=vocal_sr, target_sr=target_sr, axis=-1)

    if inst_audio.ndim == 1:
        inst_audio = np.expand_dims(inst_audio, axis=0)
    if vocal_audio.ndim == 1:
        vocal_audio = np.expand_dims(vocal_audio, axis=0)

    target_channels = max(inst_audio.shape[0], vocal_audio.shape[0])
    if inst_audio.shape[0] == 1 and target_channels > 1:
        inst_audio = np.repeat(inst_audio, target_channels, axis=0)
    if vocal_audio.shape[0] == 1 and target_channels > 1:
        vocal_audio = np.repeat(vocal_audio, target_channels, axis=0)

    target_len = max(inst_audio.shape[-1], vocal_audio.shape[-1])
    if inst_audio.shape[-1] < target_len:
        inst_audio = np.pad(inst_audio, ((0, 0), (0, target_len - inst_audio.shape[-1])))
    if vocal_audio.shape[-1] < target_len:
        vocal_audio = np.pad(vocal_audio, ((0, 0), (0, target_len - vocal_audio.shape[-1])))

    mixed = inst_audio.astype(np.float32) * float(instrumental_gain)
    mixed += vocal_audio.astype(np.float32) * float(vocal_gain)
    if normalize_output:
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if peak > 0.99:
            mixed = mixed / peak * 0.99

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(output_path), mixed.T, target_sr, format="wav")
    return output_path, target_sr, mixed
