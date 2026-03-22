import atexit
import asyncio
import logging
import os
import re
import time
import traceback
from pathlib import Path

import edge_tts
import gradio as gr
import librosa
import numpy as np
import soundfile
import torch

from inference.full_song_rvc import (
    demucs_dependency_hint,
    demucs_is_available,
    remix_vocals_with_instrumental,
    separate_song_with_demucs,
)
from inference.rvc_backend import RVCBackend

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("python_multipart").setLevel(logging.WARNING)
logging.getLogger("python_multipart.multipart").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

rvc_model_state = {"backend": RVCBackend()}
debug = False
WEBUI_LOCK_PATH = "logs/webui_instance.lock"

PAGE_INFO = {
    "overview": {
        "eyebrow": "RVC Workspace",
        "title": "Ringkas, fokus, dan khusus untuk workflow RVC",
        "desc": "Semua menu SoVITS, training, dan alat bantu lama dihapus dari antarmuka ini. Sekarang alurnya dipusatkan ke manajemen model RVC dan tiga mode konversi utama.",
    },
    "model": {
        "eyebrow": "Model Manager",
        "title": "Kelola model RVC dari satu tempat",
        "desc": "Pilih `.zip` atau `.pth`, tambahkan `.index` bila ada, lalu muat model sekali sebelum dipakai di halaman konversi lain.",
    },
    "audio": {
        "eyebrow": "Konversi Audio",
        "title": "Audio ke voice RVC",
        "desc": "Gunakan input audio biasa untuk diarahkan ke voice target. Model akan otomatis dimuat jika file model sudah dipilih di Model Manager.",
    },
    "tts": {
        "eyebrow": "Konversi Teks",
        "title": "Teks ke voice RVC",
        "desc": "Buat audio TTS Bahasa Indonesia terlebih dulu, lalu jalankan konversi dengan model RVC aktif.",
    },
    "full_song": {
        "eyebrow": "Full Song",
        "title": "Pisah vocal, konversi dengan RVC, lalu gabungkan lagi",
        "desc": "Mode ini memakai Demucs untuk memisahkan stem lagu, lalu menjalankan konversi vocal dengan engine RVC.",
    },
    "settings": {
        "eyebrow": "System",
        "title": "Pengaturan runtime dan catatan operasional",
        "desc": "Aktifkan debug jika perlu detail error yang lebih lengkap, dan gunakan halaman ini sebagai referensi cepat environment aplikasi.",
    },
}

APP_CSS = """
:root {
    --bg-0: #f3efe4;
    --bg-1: #fbf8f1;
    --bg-2: #fffdf7;
    --panel: rgba(255, 253, 247, 0.88);
    --panel-strong: rgba(252, 248, 239, 0.98);
    --line: #d8d0bc;
    --line-strong: #bfae88;
    --text: #1f2933;
    --muted: #5e6b66;
    --accent: #1c7c54;
    --accent-2: #d17b0f;
    --shadow: 0 24px 70px rgba(56, 43, 18, 0.10);
}

.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(209, 123, 15, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(28, 124, 84, 0.12), transparent 25%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 36%, var(--bg-2) 100%);
    color: var(--text);
    font-family: "Segoe UI Variable Text", "Bahnschrift", "Trebuchet MS", sans-serif;
}

.app-shell {
    gap: 18px;
}

.sidebar-panel,
.content-panel {
    border: 1px solid var(--line);
    border-radius: 24px;
    background: var(--panel);
    box-shadow: var(--shadow);
}

.sidebar-panel {
    padding: 18px 16px 16px 16px;
    backdrop-filter: blur(10px);
}

.content-panel {
    padding: 18px;
}

.brand-card {
    border: 1px solid rgba(28, 124, 84, 0.16);
    border-radius: 20px;
    padding: 18px;
    margin-bottom: 14px;
    background:
        linear-gradient(160deg, rgba(28, 124, 84, 0.16), rgba(209, 123, 15, 0.08)),
        rgba(255, 252, 246, 0.92);
}

.brand-kicker {
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: 8px;
}

.brand-title {
    font-size: 28px;
    line-height: 1.1;
    font-weight: 700;
    margin-bottom: 8px;
}

.brand-copy {
    font-size: 13px;
    line-height: 1.55;
    color: var(--muted);
}

.nav-accordion {
    border: 1px solid var(--line);
    border-radius: 16px;
    background: var(--panel-strong);
    margin-bottom: 12px;
    overflow: hidden;
}

.nav-subtitle {
    margin: 2px 0 10px 0;
    font-size: 12px;
    color: var(--muted);
}

.nav-btn {
    width: 100%;
    margin-bottom: 8px;
}

.nav-btn button {
    justify-content: flex-start !important;
    text-align: left !important;
    border-radius: 14px !important;
    border: 1px solid var(--line) !important;
    background: rgba(255, 255, 255, 0.62) !important;
    color: var(--text) !important;
    min-height: 46px !important;
    font-weight: 600 !important;
}

.nav-btn button:hover {
    border-color: var(--line-strong) !important;
    background: rgba(28, 124, 84, 0.08) !important;
}

.global-status {
    border: 1px solid rgba(28, 124, 84, 0.16);
    border-radius: 22px;
    padding: 16px 18px;
    margin-bottom: 16px;
    background:
        linear-gradient(120deg, rgba(255, 254, 250, 0.96), rgba(243, 248, 244, 0.92));
}

.status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-top: 14px;
}

.status-pill {
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 12px 14px;
    background: rgba(255, 255, 255, 0.74);
}

.status-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 6px;
}

.status-value {
    font-size: 14px;
    line-height: 1.45;
    font-weight: 600;
}

.page-hero {
    border: 1px solid rgba(191, 174, 136, 0.8);
    border-radius: 22px;
    padding: 20px 22px;
    margin-bottom: 18px;
    background:
        linear-gradient(135deg, rgba(255, 249, 238, 0.98), rgba(242, 248, 244, 0.94));
}

.page-hero .eyebrow {
    font-size: 12px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--accent-2);
    font-weight: 700;
    margin-bottom: 10px;
}

.page-hero .title {
    font-size: 30px;
    line-height: 1.1;
    font-weight: 750;
    margin-bottom: 10px;
}

.page-hero .desc {
    font-size: 14px;
    color: var(--muted);
    line-height: 1.6;
    max-width: 72ch;
}

.card-note {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 16px 18px;
    background: rgba(255, 255, 255, 0.68);
}

.card-note strong {
    color: var(--accent);
}
"""

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"


def _safe_cuda_empty_cache():
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _is_cuda_runtime_failure(err_text):
    txt = str(err_text or "").lower()
    if "cuda" not in txt and "cudnn" not in txt:
        return False
    keywords = ["out of memory", "cuda error", "cudnn_status", "cudnn error", "mapping error"]
    return any(keyword in txt for keyword in keywords)


def _file_obj_to_path(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    return getattr(file_obj, "name", None)


def _sanitize_filename_stem(name, max_len=80):
    stem = str(name or "").strip().lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        stem = "audio"
    return stem[:max_len]


def _edge_tts_rate_str(rate_value):
    rate_value = float(rate_value or 0)
    if rate_value >= 0:
        return "+{:.0%}".format(rate_value)
    return "{:.0%}".format(rate_value)


async def _save_edge_tts_to_file(text, rate_value, voice_name, output_path):
    communicate = edge_tts.Communicate(text, voice_name, rate=_edge_tts_rate_str(rate_value))
    await communicate.save(str(output_path))


def _rvc_tts_voice_name(voice_label):
    return {"Pria": "id-ID-ArdiNeural", "Wanita": "id-ID-GadisNeural"}.get(voice_label, "id-ID-ArdiNeural")


def rvc_dependency_notice():
    backend = rvc_model_state["backend"]
    if getattr(backend, "session_invalid_reason", ""):
        return backend.session_invalid_reason
    if backend.is_available:
        return "Engine RVC siap. Model `.zip` atau `.pth` bisa dimuat dari halaman Model Manager."
    return backend.dependency_hint()


def full_song_rvc_notice():
    backend = rvc_model_state["backend"]
    notes = []
    if getattr(backend, "session_invalid_reason", ""):
        notes.append(backend.session_invalid_reason)
    elif backend.is_available:
        notes.append("Engine RVC siap untuk konversi vocal.")
    else:
        notes.append(backend.dependency_hint())
    notes.append(demucs_dependency_hint())
    return "\n\n".join(notes)


def render_page_header(page_key):
    info = PAGE_INFO.get(page_key, PAGE_INFO["overview"])
    return f"""
    <div class="page-hero">
      <div class="eyebrow">{info["eyebrow"]}</div>
      <div class="title">{info["title"]}</div>
      <div class="desc">{info["desc"]}</div>
    </div>
    """


def render_global_status():
    backend = rvc_model_state["backend"]
    if getattr(backend, "session_invalid_reason", ""):
        runtime_state = "Perlu restart sesi"
        runtime_note = backend.session_invalid_reason
    elif backend.is_available:
        runtime_state = "Engine RVC aktif"
        runtime_note = "Backend inferensi siap dipakai."
    else:
        runtime_state = "Dependency belum lengkap"
        runtime_note = backend.dependency_hint()

    model_value = str(backend.model_path) if backend.model_path else "Belum ada model aktif"
    index_value = backend.index_path or "-"
    device_value = getattr(backend, "device", "cpu")
    demucs_value = "Siap" if demucs_is_available() else "Belum terpasang"

    warning_html = ""
    if backend.last_warning:
        warning_html = f"""
        <div class="status-pill">
          <div class="status-label">Warning terakhir</div>
          <div class="status-value">{backend.last_warning}</div>
        </div>
        """

    return f"""
    <div class="global-status">
      <div class="brand-kicker">Runtime Status</div>
      <div class="brand-title" style="font-size:22px; margin-bottom:6px;">RVC-only workspace aktif</div>
      <div class="brand-copy">{runtime_note}</div>
      <div class="status-grid">
        <div class="status-pill">
          <div class="status-label">Engine</div>
          <div class="status-value">{runtime_state}</div>
        </div>
        <div class="status-pill">
          <div class="status-label">Model aktif</div>
          <div class="status-value">{model_value}</div>
        </div>
        <div class="status-pill">
          <div class="status-label">Index aktif</div>
          <div class="status-value">{index_value}</div>
        </div>
        <div class="status-pill">
          <div class="status-label">Device</div>
          <div class="status-value">{device_value}</div>
        </div>
        <div class="status-pill">
          <div class="status-label">Demucs</div>
          <div class="status-value">{demucs_value}</div>
        </div>
        {warning_html}
      </div>
    </div>
    """


def refresh_runtime_panels(message="Status runtime diperbarui."):
    return message, render_global_status(), full_song_rvc_notice()


def rvc_model_load(zip_model_path, model_path, index_path, device):
    backend = rvc_model_state["backend"]
    try:
        if not backend.is_available:
            raise gr.Error(backend.dependency_hint())
        if getattr(backend, "session_invalid_reason", ""):
            raise gr.Error(backend.session_invalid_reason)
        zip_path = _file_obj_to_path(zip_model_path)
        pth_path = _file_obj_to_path(model_path)
        idx_path = _file_obj_to_path(index_path)
        source_model = zip_path or pth_path
        if source_model is None:
            raise gr.Error("Pilih file model RVC `.zip` atau `.pth` terlebih dahulu.")
        mapped_device = cuda[device] if "CUDA" in str(device) else device
        info = backend.load_model(source_model, index_path=idx_path, device=mapped_device)
        msg = (
            "Model RVC berhasil dimuat.\n"
            f"Model: {info['model_path']}\n"
            f"Index: {info['index_path'] or '-'}\n"
            f"Device: {info['device']}"
        )
        if info.get("warning"):
            msg += f"\nPeringatan: {info['warning']}"
        return msg
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def rvc_model_load_ui(zip_model_path, model_path, index_path, device):
    message = rvc_model_load(zip_model_path, model_path, index_path, device)
    return message, render_global_status(), full_song_rvc_notice()


def rvc_model_unload():
    backend = rvc_model_state["backend"]
    try:
        return backend.unload_model()
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def rvc_model_unload_ui():
    message = rvc_model_unload()
    return message, render_global_status(), full_song_rvc_notice()


def _ensure_rvc_loaded(zip_model_path, model_path, index_path, device):
    backend = rvc_model_state["backend"]
    if getattr(backend, "session_invalid_reason", ""):
        raise gr.Error(backend.session_invalid_reason)
    if backend.engine is not None:
        return backend
    zip_path = _file_obj_to_path(zip_model_path)
    pth_path = _file_obj_to_path(model_path)
    idx_path = _file_obj_to_path(index_path)
    source_model = zip_path or pth_path
    if source_model is None:
        raise gr.Error("Model RVC belum dimuat. Pilih file model di halaman Model Manager dulu.")
    mapped_device = cuda[device] if "CUDA" in str(device) else device
    backend.load_model(source_model, index_path=idx_path, device=mapped_device)
    return backend


def _gpu_restart_message(action_label, detail):
    message = (
        f"{action_label} GPU gagal pada sesi ini. "
        "State CUDA `inferrvc` tidak aman dipakai ulang di proses yang sama.\n\n"
        "Restart app untuk mencoba lagi, atau jalankan ulang dalam mode CPU:\n"
        "$env:CUDA_VISIBLE_DEVICES=\"-1\"\n"
        ".\\.venv\\Scripts\\python.exe webUI.py"
    )
    if str(detail).strip():
        message += f"\n\nDetail runtime: {detail}"
    return message


def rvc_vc_fn(zip_model_path, model_path, index_path, device, input_audio, transpose, index_rate, protect):
    backend = rvc_model_state["backend"]
    temp_path = None
    try:
        if input_audio is None:
            raise gr.Error("Anda perlu mengunggah audio.")
        backend = _ensure_rvc_loaded(zip_model_path, model_path, index_path, device)
        sampling_rate, audio = input_audio
        if np.issubdtype(audio.dtype, np.integer):
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        else:
            audio = audio.astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        temp_dir = Path("logs") / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"rvc_input_{int(time.time() * 1000)}.wav"
        soundfile.write(str(temp_path), audio, sampling_rate, format="wav")
        out_sr, out_audio = backend.infer(
            str(temp_path),
            transpose=int(transpose),
            index_rate=float(index_rate),
            protect=float(protect),
        )
        out_audio = out_audio.astype(np.float32)
        os.makedirs("results", exist_ok=True)
        timestamp = str(int(time.time()))
        model_stem = _sanitize_filename_stem(Path(backend.model_path).stem if backend.model_path else "rvc")
        filename = f"{model_stem}_rvc_{timestamp}.wav"
        output_file = Path("results") / filename
        soundfile.write(str(output_file), out_audio, out_sr, format="wav")
        msg = f"Konversi audio RVC berhasil. File tersimpan di results/{filename}"
        if backend.last_warning:
            msg += f"\nPeringatan: {backend.last_warning}"
        return msg, (out_sr, out_audio)
    except Exception as e:
        err_text = str(e).lower()
        if _is_cuda_runtime_failure(err_text):
            message = _gpu_restart_message("Inferensi audio RVC", e)
            if hasattr(backend, "invalidate_session"):
                backend.invalidate_session(message)
            try:
                backend.unload_model()
            except Exception:
                pass
            _safe_cuda_empty_cache()
            raise gr.Error(message)
        if debug:
            traceback.print_exc()
        raise gr.Error(e)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def rvc_tts_fn(zip_model_path, model_path, index_path, device, text_value, tts_rate, tts_voice, transpose, index_rate, protect):
    backend = rvc_model_state["backend"]
    temp_audio_path = None
    temp_tts_path = None
    try:
        text_value = str(text_value or "").strip()
        if not text_value:
            raise gr.Error("Masukkan teks terlebih dahulu.")
        backend = _ensure_rvc_loaded(zip_model_path, model_path, index_path, device)
        temp_dir = Path("logs") / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        temp_tts_path = temp_dir / f"rvc_tts_{stamp}.mp3"
        temp_audio_path = temp_dir / f"rvc_tts_{stamp}.wav"
        asyncio.run(
            _save_edge_tts_to_file(text_value, tts_rate, _rvc_tts_voice_name(tts_voice), temp_tts_path)
        )
        wav, sr = librosa.load(str(temp_tts_path), sr=None, mono=True)
        soundfile.write(str(temp_audio_path), wav.astype(np.float32), sr, format="wav")
        out_sr, out_audio = backend.infer(
            str(temp_audio_path),
            transpose=int(transpose),
            index_rate=float(index_rate),
            protect=float(protect),
        )
        out_audio = out_audio.astype(np.float32)
        os.makedirs("results", exist_ok=True)
        timestamp = str(int(time.time()))
        model_stem = _sanitize_filename_stem(Path(backend.model_path).stem if backend.model_path else "rvc")
        text_stem = _sanitize_filename_stem(text_value, max_len=32)
        filename = f"{text_stem}_{model_stem}_rvc_tts_{timestamp}.wav"
        output_file = Path("results") / filename
        soundfile.write(str(output_file), out_audio, out_sr, format="wav")
        msg = f"Konversi teks ke voice RVC berhasil. File tersimpan di results/{filename}"
        if backend.last_warning:
            msg += f"\nPeringatan: {backend.last_warning}"
        return msg, (out_sr, out_audio)
    except Exception as e:
        err_text = str(e).lower()
        if _is_cuda_runtime_failure(err_text):
            message = _gpu_restart_message("Konversi teks ke voice RVC", e)
            if hasattr(backend, "invalidate_session"):
                backend.invalidate_session(message)
            try:
                backend.unload_model()
            except Exception:
                pass
            _safe_cuda_empty_cache()
            raise gr.Error(message)
        if debug:
            traceback.print_exc()
        raise gr.Error(e)
    finally:
        for temp_path in (temp_tts_path, temp_audio_path):
            if temp_path is not None and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                except OSError:
                    pass


def full_song_rvc_fn(
    zip_model_path,
    model_path,
    index_path,
    rvc_device,
    separation_device,
    input_song,
    transpose,
    index_rate,
    protect,
    vocal_gain,
    instrumental_gain,
    normalize_mix,
):
    backend = rvc_model_state["backend"]
    try:
        if not demucs_is_available():
            raise gr.Error(demucs_dependency_hint())
        if input_song is None:
            raise gr.Error("Pilih file lagu penuh terlebih dahulu.")
        song_path = _file_obj_to_path(input_song)
        if song_path is None:
            raise gr.Error("Format file lagu tidak dikenali.")
        backend = _ensure_rvc_loaded(zip_model_path, model_path, index_path, rvc_device)
        work_root = Path("logs") / "full_song_rvc"
        work_root.mkdir(parents=True, exist_ok=True)
        separation = separate_song_with_demucs(song_path, output_root=work_root, device=separation_device)
        vocals_path = Path(separation["vocals_path"])
        instrumental_path = Path(separation["instrumental_path"])
        out_sr, out_audio = backend.infer(
            str(vocals_path),
            transpose=int(transpose),
            index_rate=float(index_rate),
            protect=float(protect),
        )
        out_audio = out_audio.astype(np.float32)
        os.makedirs("results", exist_ok=True)
        timestamp = str(int(time.time()))
        song_stem = _sanitize_filename_stem(Path(song_path).stem)
        model_stem = _sanitize_filename_stem(Path(backend.model_path).stem if backend.model_path else "rvc")
        converted_vocal_path = Path("results") / f"{song_stem}_{model_stem}_converted_vocal_{timestamp}.wav"
        soundfile.write(str(converted_vocal_path), out_audio, out_sr, format="wav")
        mixed_output_path = Path("results") / f"{song_stem}_{model_stem}_full_mix_{timestamp}.wav"
        remix_vocals_with_instrumental(
            instrumental_path=instrumental_path,
            converted_vocal_path=converted_vocal_path,
            output_path=mixed_output_path,
            vocal_gain=float(vocal_gain),
            instrumental_gain=float(instrumental_gain),
            normalize_output=bool(normalize_mix),
        )
        msg = (
            "Konversi full song selesai.\n"
            f"Vocal terpisah: {vocals_path}\n"
            f"Instrumental: {instrumental_path}\n"
            f"Vocal hasil RVC: results/{converted_vocal_path.name}\n"
            f"Hasil gabungan: results/{mixed_output_path.name}"
        )
        if backend.last_warning:
            msg += f"\nPeringatan: {backend.last_warning}"
        return msg, str(mixed_output_path), str(converted_vocal_path), str(instrumental_path)
    except Exception as e:
        err_text = str(e).lower()
        if _is_cuda_runtime_failure(err_text):
            message = _gpu_restart_message("Konversi full song RVC", e)
            if hasattr(backend, "invalidate_session"):
                backend.invalidate_session(message)
            try:
                backend.unload_model()
            except Exception:
                pass
            _safe_cuda_empty_cache()
            raise gr.Error(message)
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def set_debug_mode(value):
    global debug
    debug = bool(value)


def navigate(page_key):
    page_key = page_key if page_key in PAGE_INFO else "overview"
    keys = ["overview", "model", "audio", "tts", "full_song", "settings"]
    outputs = [render_page_header(page_key)]
    for key in keys:
        outputs.append(gr.update(visible=(key == page_key)))
    return outputs


def _pid_is_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_webui_lock_or_exit():
    os.makedirs(os.path.dirname(WEBUI_LOCK_PATH), exist_ok=True)
    current_pid = os.getpid()
    if os.path.exists(WEBUI_LOCK_PATH):
        try:
            old_pid = int(Path(WEBUI_LOCK_PATH).read_text(encoding="utf-8").strip())
        except Exception:
            old_pid = 0
        if _pid_is_alive(old_pid) and old_pid != current_pid:
            print(f"[LOCK] webUI sudah berjalan (PID {old_pid}). Tutup instance lama dulu.")
            raise SystemExit(0)
    Path(WEBUI_LOCK_PATH).write_text(str(current_pid), encoding="utf-8")

    def _release_lock():
        try:
            if os.path.exists(WEBUI_LOCK_PATH):
                owner = int(Path(WEBUI_LOCK_PATH).read_text(encoding="utf-8").strip())
                if owner == current_pid:
                    os.remove(WEBUI_LOCK_PATH)
        except Exception:
            pass

    atexit.register(_release_lock)


with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        font=["Segoe UI Variable Text", "Bahnschrift", "Trebuchet MS", "sans-serif"],
        font_mono=["JetBrains Mono", "Consolas", "Courier New"],
    ),
    css=APP_CSS,
) as app:
    rvc_available = rvc_model_state["backend"].is_available
    full_song_available = rvc_available and demucs_is_available()

    with gr.Row(elem_classes=["app-shell"]):
        with gr.Column(scale=1, min_width=285, elem_classes=["sidebar-panel"]):
            gr.HTML(
                """
                <div class="brand-card">
                  <div class="brand-kicker">RVC Console</div>
                  <div class="brand-title">Voice Conversion yang lebih rapi</div>
                  <div class="brand-copy">
                    Sidebar ini memisahkan workflow ke beberapa grup. Fokus UI sekarang hanya pada model RVC, konversi audio, konversi teks, dan full song.
                  </div>
                </div>
                """
            )
            with gr.Accordion("Inti", open=True, elem_classes=["nav-accordion"]):
                gr.Markdown("<div class='nav-subtitle'>Halaman utama dan manajemen model.</div>")
                nav_overview = gr.Button("Ringkasan", elem_classes=["nav-btn"])
                nav_model = gr.Button("Model Manager", elem_classes=["nav-btn"])
            with gr.Accordion("Konversi", open=True, elem_classes=["nav-accordion"]):
                gr.Markdown("<div class='nav-subtitle'>Pilih mode input untuk workflow RVC.</div>")
                nav_audio = gr.Button("Audio ke Voice", elem_classes=["nav-btn"])
                nav_tts = gr.Button("Teks ke Voice", elem_classes=["nav-btn"])
                nav_full_song = gr.Button("Full Song", elem_classes=["nav-btn"])
            with gr.Accordion("Sistem", open=False, elem_classes=["nav-accordion"]):
                gr.Markdown("<div class='nav-subtitle'>Status runtime dan pengaturan ringan.</div>")
                nav_settings = gr.Button("Pengaturan", elem_classes=["nav-btn"])

        with gr.Column(scale=4, min_width=780, elem_classes=["content-panel"]):
            global_status = gr.HTML(value=render_global_status())
            page_header = gr.HTML(value=render_page_header("overview"))

            with gr.Column(visible=True) as page_overview:
                overview_notice = gr.Markdown(value=rvc_dependency_notice())
                gr.HTML(
                    """
                    <div class="card-note">
                      <strong>Workflow yang disarankan:</strong><br>
                      1. Masuk ke <b>Model Manager</b> dan pilih model `.zip` atau `.pth`.<br>
                      2. Muat model sekali, lalu pindah ke halaman konversi yang dibutuhkan.<br>
                      3. Untuk <b>Full Song</b>, pastikan dependency <code>demucs</code> tersedia.
                    </div>
                    """
                )
                with gr.Row():
                    quick_model = gr.Button("Buka Model Manager", variant="primary")
                    quick_audio = gr.Button("Buka Audio ke Voice")
                    quick_full_song = gr.Button("Buka Full Song")

            with gr.Column(visible=False) as page_model:
                with gr.Row(variant="panel"):
                    with gr.Column():
                        rvc_zip_model_path = gr.File(label="Pilih ZIP model RVC (.zip) [opsional]")
                        rvc_model_path = gr.File(label="Pilih file model RVC (.pth)")
                        rvc_index_path = gr.File(label="Pilih file index RVC (.index) [opsional]")
                        rvc_device = gr.Dropdown(label="Perangkat inferensi RVC", choices=["Auto", *cuda.keys(), "CPU"], value="Auto")
                    with gr.Column():
                        rvc_model_load_button = gr.Button(value="Muat Model RVC", variant="primary", interactive=rvc_available)
                        rvc_model_unload_button = gr.Button(value="Copot Model RVC", variant="secondary", interactive=rvc_available)
                        runtime_refresh_button = gr.Button(value="Refresh Status Runtime", interactive=True)
                        rvc_status_output = gr.Textbox(label="Status Model", lines=8, value="" if rvc_available else rvc_dependency_notice())

            with gr.Column(visible=False) as page_audio:
                gr.Markdown("Pilih audio input lalu jalankan konversi dengan model RVC aktif.")
                with gr.Row(variant="panel"):
                    rvc_transpose = gr.Number(label="Transpose / semitone", value=0)
                    rvc_index_rate = gr.Number(label="index_rate (0-1)", value=0.75)
                    rvc_protect = gr.Number(label="protect (0-0.5 disarankan)", value=0.33)
                rvc_input_audio = gr.Audio(label="Pilih Audio")
                rvc_submit = gr.Button("Konversi Audio RVC", variant="primary", interactive=rvc_available)
                with gr.Row():
                    with gr.Column():
                        rvc_output1 = gr.Textbox(label="Pesan Output", lines=6)
                    with gr.Column():
                        rvc_output2 = gr.Audio(label="Audio Output", interactive=False)

            with gr.Column(visible=False) as page_tts:
                gr.Markdown("Masukkan teks, pilih suara TTS dasar, lalu teruskan ke model RVC.")
                rvc_tts_text = gr.Textbox(label="Masukkan teks")
                with gr.Row(variant="panel"):
                    rvc_tts_rate = gr.Number(label="Kecepatan TTS", value=0)
                    rvc_tts_voice = gr.Radio(label="Jenis Kelamin", choices=["Pria", "Wanita"], value="Pria")
                rvc_tts_submit = gr.Button("Konversi Teks ke Voice RVC", variant="primary", interactive=rvc_available)
                with gr.Row():
                    with gr.Column():
                        rvc_tts_output_text = gr.Textbox(label="Pesan Output", lines=6)
                    with gr.Column():
                        rvc_tts_output_audio = gr.Audio(label="Audio Output", interactive=False)

            with gr.Column(visible=False) as page_full_song:
                full_song_notice = gr.Markdown(value=full_song_rvc_notice())
                with gr.Row(variant="panel"):
                    with gr.Column():
                        full_song_input = gr.File(label="Pilih Lagu Full (audio)")
                        full_song_demucs_device = gr.Dropdown(label="Perangkat stem separation (Demucs)", choices=["Auto", *cuda.keys(), "CPU"], value="Auto")
                    with gr.Column():
                        full_song_transpose = gr.Number(label="Transpose / semitone", value=0)
                        full_song_index_rate = gr.Number(label="index_rate (0-1)", value=0.75)
                        full_song_protect = gr.Number(label="protect (0-0.5 disarankan)", value=0.33)
                with gr.Row(variant="panel"):
                    full_song_vocal_gain = gr.Number(label="Gain vocal hasil", value=1.0)
                    full_song_instrumental_gain = gr.Number(label="Gain instrumental", value=1.0)
                    full_song_normalize = gr.Checkbox(label="Normalize hasil gabungan", value=True)
                full_song_submit = gr.Button("Konversi Lagu Full dengan RVC", variant="primary", interactive=full_song_available)
                with gr.Row():
                    with gr.Column():
                        full_song_output_text = gr.Textbox(label="Pesan Output", lines=8)
                        full_song_mix_output = gr.Audio(label="Hasil Gabungan", interactive=False, type="filepath")
                    with gr.Column():
                        full_song_vocal_output = gr.Audio(label="Vocal Hasil RVC", interactive=False, type="filepath")
                        full_song_instrumental_output = gr.Audio(label="Instrumental", interactive=False, type="filepath")

            with gr.Column(visible=False) as page_settings:
                gr.HTML(
                    """
                    <div class="card-note">
                      <strong>Catatan:</strong><br>
                      UI ini sekarang fokus penuh ke RVC. Fitur SoVITS, mixing model, slicer, dan training manager tidak lagi ditampilkan atau dipakai saat boot.
                    </div>
                    """
                )
                debug_button = gr.Checkbox(label="Mode Debug", value=debug)
                env_note = gr.Markdown(
                    value="Gunakan halaman ini untuk memantau perilaku runtime. Jika engine GPU crash, restart aplikasi sebelum mencoba lagi."
                )

    nav_outputs = [page_header, page_overview, page_model, page_audio, page_tts, page_full_song, page_settings]
    nav_overview.click(lambda: navigate("overview"), outputs=nav_outputs, queue=False)
    nav_model.click(lambda: navigate("model"), outputs=nav_outputs, queue=False)
    nav_audio.click(lambda: navigate("audio"), outputs=nav_outputs, queue=False)
    nav_tts.click(lambda: navigate("tts"), outputs=nav_outputs, queue=False)
    nav_full_song.click(lambda: navigate("full_song"), outputs=nav_outputs, queue=False)
    nav_settings.click(lambda: navigate("settings"), outputs=nav_outputs, queue=False)
    quick_model.click(lambda: navigate("model"), outputs=nav_outputs, queue=False)
    quick_audio.click(lambda: navigate("audio"), outputs=nav_outputs, queue=False)
    quick_full_song.click(lambda: navigate("full_song"), outputs=nav_outputs, queue=False)

    debug_button.change(set_debug_mode, [debug_button], [], queue=False)
    runtime_refresh_button.click(refresh_runtime_panels, [], [rvc_status_output, global_status, full_song_notice], queue=False)
    rvc_model_load_button.click(
        rvc_model_load_ui,
        [rvc_zip_model_path, rvc_model_path, rvc_index_path, rvc_device],
        [rvc_status_output, global_status, full_song_notice],
    )
    rvc_model_unload_button.click(rvc_model_unload_ui, [], [rvc_status_output, global_status, full_song_notice])
    rvc_submit.click(
        rvc_vc_fn,
        [rvc_zip_model_path, rvc_model_path, rvc_index_path, rvc_device, rvc_input_audio, rvc_transpose, rvc_index_rate, rvc_protect],
        [rvc_output1, rvc_output2],
    )
    rvc_tts_submit.click(
        rvc_tts_fn,
        [rvc_zip_model_path, rvc_model_path, rvc_index_path, rvc_device, rvc_tts_text, rvc_tts_rate, rvc_tts_voice, rvc_transpose, rvc_index_rate, rvc_protect],
        [rvc_tts_output_text, rvc_tts_output_audio],
    )
    full_song_submit.click(
        full_song_rvc_fn,
        [rvc_zip_model_path, rvc_model_path, rvc_index_path, rvc_device, full_song_demucs_device, full_song_input, full_song_transpose, full_song_index_rate, full_song_protect, full_song_vocal_gain, full_song_instrumental_gain, full_song_normalize],
        [full_song_output_text, full_song_mix_output, full_song_vocal_output, full_song_instrumental_output],
    )
    app.load(
        lambda: [render_global_status(), rvc_dependency_notice(), full_song_rvc_notice()],
        [],
        [global_status, overview_notice, full_song_notice],
        queue=False,
    )


if __name__ == "__main__":
    _acquire_webui_lock_or_exit()
    app.queue().launch(server_name="127.0.0.1", show_api=False)
