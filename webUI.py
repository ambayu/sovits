import io
import os
import sys
import shutil
import zipfile
import atexit

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import gradio.processing_utils as gr_pu
import librosa
import numpy as np
import soundfile
from inference.infer_tool import Svc
from inference.full_song_rvc import demucs_dependency_hint, demucs_is_available, remix_vocals_with_instrumental, separate_song_with_demucs
from inference.rvc_backend import RVCBackend
import logging
import re
import json

import subprocess
import edge_tts
import asyncio
from scipy.io import wavfile
import librosa
import torch
import time
import traceback
import hashlib
from datetime import datetime
from itertools import chain
import threading
from utils import mix_model
from pathlib import Path
from inference import slicer as dataset_slicer

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)
logging.getLogger('python_multipart').setLevel(logging.WARNING)
logging.getLogger('python_multipart.multipart').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

sovits_model_state = {"model": None}
rvc_model_state = {"backend": RVCBackend()}
debug = False
TRAIN_STATE_PATH = "logs/webui_train_state.json"
MODEL_REGISTRY_PATH = "logs/webui_prepared_models.json"
WEBUI_LOCK_PATH = "logs/webui_instance.lock"
TRAIN_STEP_ORDER = ["resample", "flist_config", "hubert_f0", "train"]
TRAIN_STEP_LABELS = {
    "resample": "1) Resample Audio",
    "flist_config": "2) Generate Filelist + Config",
    "hubert_f0": "3) Extract Hubert + F0",
    "train": "4) Train Model",
}
TRAIN_STEP_REQUIREMENTS = {
    "resample": [],
    "flist_config": ["resample"],
    "hubert_f0": ["flist_config"],
    "train": ["hubert_f0"],
}
DEFAULT_DRIVE_SYNC_PATH = r"H:\My Drive\sovits_run"
APP_CSS = """
.tm-hero {
    border: 1px solid #dbe4db;
    border-radius: 12px;
    padding: 12px 14px;
    background: linear-gradient(120deg, #f7fbf8 0%, #eef5f0 100%);
}
.tm-steps {
    margin-top: 8px;
    padding-left: 18px;
}
#tm-progress .noUi-connect {
    background: linear-gradient(90deg, #16a34a 0%, #22c55e 100%);
}
#tm-progress {
    border: 1px solid #d3e7d7;
    border-radius: 10px;
    padding: 8px;
    background: #f7fcf8;
}
.tm-log-note {
    color: #35523d;
    font-size: 12px;
    margin-top: 2px;
}
"""
TRAIN_RUNTIME = {
    "lock": threading.Lock(),
    "proc": None,
    "stop_requested": False,
}

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
        # After CUDA OOM, some cleanup calls can also fail; ignore and continue fallback flow.
        pass


def _is_cuda_runtime_failure(err_text):
    txt = str(err_text or "").lower()
    if "cuda" not in txt and "cudnn" not in txt:
        return False
    keywords = [
        "out of memory",
        "cuda error",
        "cudnn_status",
        "cudnn error",
        "mapping error",
    ]
    return any(k in txt for k in keywords)

def upload_mix_append_file(files,sfiles):
    try:
        if(sfiles == None):
            file_paths = [file.name for file in files]
        else:
            file_paths = [file.name for file in chain(files,sfiles)]
        p = {file:100 for file in file_paths}
        return file_paths,mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)

def mix_submit_click(js,mode):
    try:
        assert js.lstrip()!=""
        modes = {"Kombinasi Konveks":0, "Kombinasi Linear":1}
        mode = modes[mode]
        data = json.loads(js)
        data = list(data.items())
        model_path,mix_rate = zip(*data)
        path = mix_model(model_path,mix_rate,mode)
        return f"Berhasil, file disimpan di {path}"
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)

def updata_mix_info(files):
    try:
        if files == None : return mix_model_output1.update(value="")
        p = {file.name:100 for file in files}
        return mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)

def modelAnalysis(model_path,config_path,cluster_model_path,device,enhance):
    try:
        if model_path is None or config_path is None:
            raise gr.Error("Anda perlu memilih file model dan konfigurasi So-VITS.")
        device = cuda[device] if "CUDA" in device else device
        old_model = sovits_model_state.get("model")
        if old_model is not None:
            try:
                old_model.unload_model()
            except Exception:
                pass
        loaded_model = Svc(
            model_path.name,
            config_path.name,
            device=device if device!="Auto" else None,
            cluster_model_path=cluster_model_path.name if cluster_model_path != None else "",
            nsf_hifigan_enhance=enhance,
        )
        sovits_model_state["model"] = loaded_model
        spks = list(loaded_model.spk2id.keys())
        device_name = torch.cuda.get_device_properties(loaded_model.dev).name if "cuda" in str(loaded_model.dev) else str(loaded_model.dev)
        msg = f"Berhasil memuat model ke perangkat {device_name}\n"
        if cluster_model_path is None:
            msg += "Model clustering tidak dimuat\n"
        else:
            msg += f"Model clustering {cluster_model_path.name} berhasil dimuat\n"
        msg += "Warna suara yang tersedia pada model saat ini:\n"
        for i in spks:
            msg += i + " "
        return sid.update(choices = spks,value=spks[0]), msg
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)

    
def modelUnload():
    model = sovits_model_state.get("model")
    if model is None:
        return sid.update(choices = [],value=""),"Tidak ada model yang perlu dicopot!"
    else:
        model.unload_model()
        sovits_model_state["model"] = None
        _safe_cuda_empty_cache()
        return sid.update(choices = [],value=""),"Model berhasil dicopot!"


def vc_fn(sid, input_audio, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,F0_mean_pooling,enhancer_adaptive_key,cr_threshold):
    model = sovits_model_state.get("model")
    try:
        if input_audio is None:
            raise gr.Error("Anda perlu mengunggah audio")
        if model is None:
            raise gr.Error("Anda perlu memilih model")
        sampling_rate, audio = input_audio
        # print(audio.shape,sampling_rate)
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        temp_path = "temp.wav"
        soundfile.write(temp_path, audio, sampling_rate, format="wav")
        _audio = model.slice_inference(temp_path, sid, vc_transform, slice_db, cluster_ratio, auto_f0, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,F0_mean_pooling,enhancer_adaptive_key,cr_threshold)
        model.clear_empty()
        os.remove(temp_path)
        # Membuat path penyimpanan file dan menyimpan ke folder results
        try:
            timestamp = str(int(time.time()))
            filename = sid + "_" + timestamp + ".wav"
            output_file = os.path.join("./results", filename)
            soundfile.write(output_file, _audio, model.target_sample, format="wav")
            return f"Inferensi berhasil, file audio disimpan di results/{filename}", (model.target_sample, _audio)
        except Exception as e:
            if debug: traceback.print_exc()
            return f"Gagal menyimpan file, silakan simpan secara manual", (model.target_sample, _audio)
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)


def tts_func(_text,_rate,_voice):
    # Menggunakan edge-tts untuk mengubah teks menjadi audio
    voice = "zh-CN-YunxiNeural" # Pria
    if ( _voice == "Wanita" ) : voice = "zh-CN-XiaoyiNeural"
    output_file = _text[0:10]+".wav"
    if _rate>=0:
        ratestr="+{:.0%}".format(_rate)
    elif _rate<0:
        ratestr="{:.0%}".format(_rate) # tanda minus otomatis

    p=subprocess.Popen("edge-tts "+
                        " --text "+_text+
                        " --write-media "+output_file+
                        " --voice "+voice+
                        " --rate="+ratestr
                        ,shell=True,
                        stdout=subprocess.PIPE,
                        stdin=subprocess.PIPE)
    p.wait()
    return output_file

def text_clear(text):
    return re.sub(r"[\n\,\(\) ]", "", text)


def _edge_tts_rate_str(rate_value):
    rate_value = float(rate_value or 0)
    if rate_value >= 0:
        return "+{:.0%}".format(rate_value)
    return "{:.0%}".format(rate_value)


async def _save_edge_tts_to_file(text, rate_value, voice_name, output_path):
    communicate = edge_tts.Communicate(text, voice_name, rate=_edge_tts_rate_str(rate_value))
    await communicate.save(str(output_path))


def _rvc_tts_voice_name(voice_label):
    return {
        "Pria": "id-ID-ArdiNeural",
        "Wanita": "id-ID-GadisNeural",
    }.get(voice_label, "id-ID-ArdiNeural")

def vc_fn2(sid, input_audio, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,text2tts,tts_rate,tts_voice,F0_mean_pooling,enhancer_adaptive_key,cr_threshold):
    # Menggunakan edge-tts untuk mengubah teks menjadi audio
    text2tts=text_clear(text2tts)
    output_file=tts_func(text2tts,tts_rate,tts_voice)

    # Menyesuaikan sample rate
    sr2=44100
    wav, sr = librosa.load(output_file)
    wav2 = librosa.resample(wav, orig_sr=sr, target_sr=sr2)
    save_path2= text2tts[0:10]+"_44k"+".wav"
    wavfile.write(save_path2,sr2,
                (wav2 * np.iinfo(np.int16).max).astype(np.int16)
                )

    # Membaca audio
    sample_rate, data=gr_pu.audio_from_file(save_path2)
    vc_input=(sample_rate, data)

    a,b=vc_fn(sid, vc_input, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,F0_mean_pooling,enhancer_adaptive_key,cr_threshold)
    os.remove(output_file)
    os.remove(save_path2)
    return a,b


def _file_obj_to_path(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    return getattr(file_obj, "name", None)


def rvc_dependency_notice():
    backend = rvc_model_state["backend"]
    if getattr(backend, "session_invalid_reason", ""):
        return backend.session_invalid_reason
    if backend.is_available:
        return "Engine RVC siap. Anda bisa memuat model `.zip` atau `.pth`."
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
            raise gr.Error("Pilih file model RVC `.zip` atau `.pth`.")
        sovits_model = sovits_model_state.get("model")
        if sovits_model is not None:
            try:
                sovits_model.unload_model()
            except Exception:
                pass
            sovits_model_state["model"] = None
            _safe_cuda_empty_cache()
        mapped_device = cuda[device] if "CUDA" in device else device
        info = backend.load_model(source_model, index_path=idx_path, device=mapped_device)
        msg = (
            f"Model RVC berhasil dimuat.\n"
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


def rvc_model_unload():
    backend = rvc_model_state["backend"]
    try:
        return backend.unload_model()
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


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
        raise gr.Error("Model RVC belum dimuat. Pilih file `.zip` atau `.pth`, lalu klik `Muat Model RVC` atau langsung `Konversi Audio RVC` lagi.")

    mapped_device = cuda[device] if "CUDA" in device else device
    backend.load_model(source_model, index_path=idx_path, device=mapped_device)
    return backend


def rvc_vc_fn(zip_model_path, model_path, index_path, device, input_audio, transpose, index_rate, protect):
    backend = rvc_model_state["backend"]
    temp_path = None
    try:
        if input_audio is None:
            raise gr.Error("Anda perlu mengunggah audio")
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

        msg = f"Inferensi RVC berhasil, file audio disimpan di results/{filename}"
        if backend.last_warning:
            msg += f"\nPeringatan: {backend.last_warning}"
        return msg, (out_sr, out_audio)
    except Exception as e:
        err_text = str(e).lower()
        if _is_cuda_runtime_failure(err_text):
            msg = (
                "Inferensi RVC GPU gagal pada sesi ini. "
                "State CUDA `inferrvc` tidak aman dipakai ulang di proses yang sama. "
                "Restart app untuk mencoba lagi, atau jalankan ulang dalam mode CPU:\n"
                "$env:CUDA_VISIBLE_DEVICES=\"-1\"\n"
                ".\\.venv310\\Scripts\\python.exe webUI.py"
            )
            if str(e).strip():
                msg += f"\n\nDetail runtime: {e}"
            if hasattr(backend, "invalidate_session"):
                backend.invalidate_session(msg)
            try:
                backend.unload_model()
            except Exception:
                pass
            _safe_cuda_empty_cache()
            raise gr.Error(msg)
        if debug:
            traceback.print_exc()
        raise gr.Error(e)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
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
    work_dir = None
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
        separation = separate_song_with_demucs(
            song_path,
            output_root=work_root,
            device=separation_device,
        )
        work_dir = Path(separation["run_dir"])

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
            "Konversi lagu full berhasil.\n"
            f"Vocal terpisah: {vocals_path}\n"
            f"Instrumental: {instrumental_path}\n"
            f"Vocal hasil RVC: results/{converted_vocal_path.name}\n"
            f"Hasil gabungan: results/{mixed_output_path.name}"
        )
        if backend.last_warning:
            msg += f"\nPeringatan: {backend.last_warning}"
        return (
            msg,
            str(mixed_output_path),
            str(converted_vocal_path),
            str(instrumental_path),
        )
    except Exception as e:
        err_text = str(e).lower()
        if _is_cuda_runtime_failure(err_text):
            msg = (
                "Konversi lagu full dengan RVC GPU gagal pada sesi ini. "
                "Restart app untuk mencoba lagi, atau jalankan ulang dalam mode CPU:\n"
                "$env:CUDA_VISIBLE_DEVICES=\"-1\"\n"
                ".\\.venv310\\Scripts\\python.exe webUI.py"
            )
            if str(e).strip():
                msg += f"\n\nDetail runtime: {e}"
            if hasattr(backend, "invalidate_session"):
                backend.invalidate_session(msg)
            try:
                backend.unload_model()
            except Exception:
                pass
            _safe_cuda_empty_cache()
            raise gr.Error(msg)
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


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
            _save_edge_tts_to_file(
                text_value,
                tts_rate,
                _rvc_tts_voice_name(tts_voice),
                temp_tts_path,
            )
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

        msg = f"Teks ke voice RVC berhasil, file audio disimpan di results/{filename}"
        if backend.last_warning:
            msg += f"\nPeringatan: {backend.last_warning}"
        return msg, (out_sr, out_audio)
    except Exception as e:
        err_text = str(e).lower()
        if _is_cuda_runtime_failure(err_text):
            msg = (
                "Teks ke voice RVC gagal pada sesi GPU ini. "
                "Restart app untuk mencoba lagi, atau jalankan ulang dalam mode CPU:\n"
                "$env:CUDA_VISIBLE_DEVICES=\"-1\"\n"
                ".\\.venv310\\Scripts\\python.exe webUI.py"
            )
            if str(e).strip():
                msg += f"\n\nDetail runtime: {e}"
            if hasattr(backend, "invalidate_session"):
                backend.invalidate_session(msg)
            try:
                backend.unload_model()
            except Exception:
                pass
            _safe_cuda_empty_cache()
            raise gr.Error(msg)
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


def _sanitize_speaker_name(name):
    cleaned = re.sub(r"[^a-zA-Z0-9_\- ]", "", str(name or "").strip())
    cleaned = cleaned.replace(" ", "_")
    return cleaned


def _sanitize_filename_stem(name, max_len=80):
    stem = str(name or "").strip().lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        stem = "audio"
    return stem[:max_len]


def list_dataset_raw_models():
    root = Path("dataset_raw")
    if not root.exists():
        return []
    models = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        has_wav = any(p.glob("*.wav"))
        if has_wav:
            models.append(p.name)
    return models


def _pick_selected_model(selected_model_value):
    models = list_dataset_raw_models()
    if not models:
        return "", models
    selected_model = (selected_model_value or "").strip()
    if selected_model in models:
        return selected_model, models
    return models[0], models


def render_train_target_label(selected_model_value):
    selected_model, _ = _pick_selected_model(selected_model_value)
    if not selected_model:
        return "<small>Model yang akan di-train saat ini: -</small>"
    return f"<small>Model yang akan di-train saat ini: <b>{selected_model}</b></small>"


def slice_audio_stream(audio_path, speaker_name, db_thresh, min_len):
    if audio_path is not None and not isinstance(audio_path, str) and hasattr(audio_path, "name"):
        audio_path = audio_path.name
    speaker_name = _sanitize_speaker_name(speaker_name)
    start_time = time.time()
    try:
        if not speaker_name:
            yield "Gagal: Nama Speaker tidak valid", 0
            return
        if not audio_path or not os.path.isfile(audio_path):
            yield "Gagal: File audio tidak ditemukan", 0
            return

        out_dir = os.path.join("dataset_raw", speaker_name)
        os.makedirs(out_dir, exist_ok=True)

        yield "Menganalisis audio (25%)", 25
        chunks = dataset_slicer.cut(audio_path, db_thresh=float(db_thresh), min_len=int(min_len))

        yield "Menyusun potongan audio (45%)", 45
        audio_data, audio_sr = dataset_slicer.chunks2audio(audio_path, chunks)

        keep_segments = [data for (slice_tag, data) in audio_data if not slice_tag and len(data) > 0]
        total = len(keep_segments)
        if total == 0:
            yield "Error: Tidak ada potongan yang lolos. Coba turunkan ambang diam atau kecilkan panjang minimum", 0
            return

        stem = _sanitize_filename_stem(Path(audio_path).stem)
        for idx, chunk_data in enumerate(keep_segments, start=1):
            filename = f"{stem}_{idx:04d}.wav"
            output_path = os.path.join(out_dir, filename)
            soundfile.write(output_path, chunk_data, audio_sr, format="wav")
            progress = 45 + int((idx / total) * 54)
            yield f"Menyimpan potongan {idx}/{total} ({min(progress, 99)}%)", min(progress, 99)

        elapsed = time.time() - start_time
        yield f"Selesai: {total} potongan disimpan ke {out_dir} (waktu {elapsed:.1f} detik)", 100
    except Exception as e:
        if debug:
            traceback.print_exc()
        yield f"Error: {e}", 0


def _compute_dataset_fingerprint(dataset_dir="dataset_raw", selected_model=""):
    hasher = hashlib.sha256()
    file_count = 0
    if not os.path.isdir(dataset_dir):
        return "", 0
    selected_model = (selected_model or "").strip()
    for root, _, files in os.walk(dataset_dir):
        for filename in sorted(files):
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, dataset_dir).replace("\\", "/")
            if selected_model and not rel.startswith(f"{selected_model}/"):
                continue
            try:
                st = os.stat(path)
                hasher.update(f"{rel}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8"))
                file_count += 1
            except OSError:
                continue
    return hasher.hexdigest(), file_count


def _compute_dir_fingerprint(src_dir):
    src = Path(src_dir)
    if not src.exists():
        return ""
    hasher = hashlib.sha256()
    for root, _, files in os.walk(src):
        for filename in sorted(files):
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, src).replace("\\", "/")
            try:
                st = os.stat(path)
                hasher.update(f"{rel}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8"))
            except OSError:
                continue
    return hasher.hexdigest()


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _zip_directory(src_dir, out_zip_path, top_level_name=None):
    src = Path(src_dir)
    if not src.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {src_dir}")
    tmp_zip = Path(out_zip_path)
    if tmp_zip.exists():
        tmp_zip.unlink()
    if top_level_name:
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(src):
                for filename in files:
                    full = Path(root) / filename
                    rel = full.relative_to(src).as_posix()
                    arcname = f"{top_level_name}/{rel}"
                    zf.write(full, arcname=arcname)
    else:
        shutil.make_archive(str(tmp_zip.with_suffix("")), "zip", root_dir=str(src))
    return str(tmp_zip)


def sync_artifacts_to_drive_stream(drive_sync_path, model_name_value):
    logs = []
    drive_sync_path = (drive_sync_path or "").strip()
    if not drive_sync_path:
        yield refresh_train_status(model_name_value), "Path sinkron kosong."
        return
    model_name_safe = _sanitize_filename_stem(model_name_value or "44k", max_len=48)
    dst_dir = Path(drive_sync_path) / model_name_safe
    dst_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dst_dir / "sync_manifest.json"

    exports_dir = Path("logs") / "exports_tmp"
    exports_dir.mkdir(parents=True, exist_ok=True)

    current_manifest = {
        "model_name": model_name_safe,
        "dataset_fp": _compute_dir_fingerprint("dataset"),
        "configs_fp": _compute_dir_fingerprint("configs"),
        "filelists_fp": _compute_dir_fingerprint("filelists"),
    }
    model_files = {
        "dataset": f"{model_name_safe}_dataset.zip",
        "configs": f"{model_name_safe}_configs.zip",
        "filelists": f"{model_name_safe}_filelists.zip",
    }

    if manifest_path.exists():
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            old_manifest = {}
        if old_manifest == current_manifest:
            need_files = [model_files["dataset"], model_files["configs"], model_files["filelists"]]
            if all((dst_dir / f).exists() for f in need_files):
                logs.append(f"[SKIP] Data model '{model_name_safe}' tidak berubah. Tidak perlu zip/copy ulang.")
                logs.append(f"[PATH] {dst_dir}")
                yield refresh_train_status(model_name_value), "\n".join(logs)
                return

    artifact_specs = [
        ("dataset", Path("dataset"), exports_dir / "dataset.zip", dst_dir / model_files["dataset"]),
        ("configs", Path("configs"), exports_dir / "configs.zip", dst_dir / model_files["configs"]),
        ("filelists", Path("filelists"), exports_dir / "filelists.zip", dst_dir / model_files["filelists"]),
    ]

    for label, src_dir, local_zip, target_zip in artifact_specs:
        try:
            logs.append(f"[ZIP] Membuat {local_zip.name} dari {src_dir} ...")
            if label == "dataset":
                _zip_directory(src_dir, local_zip, top_level_name=f"dataset_{model_name_safe}")
            else:
                _zip_directory(src_dir, local_zip)
            local_hash = _hash_file(local_zip)

            if target_zip.exists():
                remote_hash = _hash_file(target_zip)
                if remote_hash == local_hash:
                    logs.append(f"[SKIP] {target_zip.name} sudah sama (hash identik).")
                    if len(logs) > 200:
                        logs = logs[-200:]
                    yield refresh_train_status(model_name_value), "\n".join(logs)
                    continue

            shutil.copy2(local_zip, target_zip)
            logs.append(f"[SYNC] {target_zip.name} diperbarui ke: {target_zip}")
        except Exception as e:
            logs.append(f"[ERROR] {label}: {e}")
            if len(logs) > 200:
                logs = logs[-200:]
            yield refresh_train_status(model_name_value), "\n".join(logs)
            return

        if len(logs) > 200:
            logs = logs[-200:]
        yield refresh_train_status(model_name_value), "\n".join(logs)

    manifest_path.write_text(json.dumps(current_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logs.append("[DONE] Sinkronisasi artefak selesai tanpa duplikasi.")
    logs.append(f"[PATH] Tersimpan di folder model: {dst_dir}")
    yield refresh_train_status(model_name_value), "\n".join(logs)


def _default_train_state():
    return {
        "selected_model": "",
        "dataset_fingerprint": "",
        "dataset_file_count": 0,
        "active_step": "",
        "steps": {k: {"done": False, "updated_at": "", "exit_code": None} for k in TRAIN_STEP_ORDER},
        "last_error": "",
    }


def _load_train_state():
    if not os.path.exists(TRAIN_STATE_PATH):
        return _default_train_state()
    try:
        with open(TRAIN_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return _default_train_state()
    if "steps" not in state:
        state["steps"] = {}
    for step in TRAIN_STEP_ORDER:
        state["steps"].setdefault(step, {"done": False, "updated_at": "", "exit_code": None})
    state.setdefault("selected_model", "")
    state.setdefault("dataset_fingerprint", "")
    state.setdefault("dataset_file_count", 0)
    state.setdefault("active_step", "")
    state.setdefault("last_error", "")
    return state


def _save_train_state(state):
    os.makedirs(os.path.dirname(TRAIN_STATE_PATH), exist_ok=True)
    with open(TRAIN_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _clear_stale_active_step_if_needed(state, reason_text=""):
    active = (state.get("active_step") or "").strip()
    if not active:
        return state, False
    with TRAIN_RUNTIME["lock"]:
        proc = TRAIN_RUNTIME.get("proc")
    has_live_proc = bool(proc and proc.poll() is None)
    if has_live_proc:
        return state, False
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["active_step"] = ""
    if active in state.get("steps", {}):
        state["steps"][active]["done"] = False
        state["steps"][active]["updated_at"] = now
        if state["steps"][active].get("exit_code") in (None, 0):
            state["steps"][active]["exit_code"] = 130
    state["last_error"] = reason_text or "Status langkah aktif dibersihkan karena proses tidak ditemukan."
    _save_train_state(state)
    return state, True


def _count_matching_files(base_dir, pattern):
    p = Path(base_dir)
    if not p.exists():
        return 0
    return sum(1 for _ in p.glob(pattern))


def _read_total_epochs(config_path_value):
    path = (config_path_value or "configs/config.json").strip()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("train", {}).get("epochs", 0))
    except Exception:
        return 0


def _extract_train_epoch_progress(log_text):
    if not log_text:
        return None
    matches = re.findall(r"Train Epoch:\s*(\d+)\s*\[(\d+)%\]", log_text)
    if not matches:
        return None
    epoch, pct = matches[-1]
    return int(epoch), int(pct)


def _is_noise_log_line(line):
    lowered = (line or "").lower()
    noisy_tokens = [
        "pkg_resources is deprecated",
        "from pkg_resources import",
        "refain from using this package",
        "setuptools<81",
    ]
    return any(tok in lowered for tok in noisy_tokens)


def _trim_server_logs_for_ui(lines):
    cleaned = []
    for raw in lines:
        text = (raw or "").strip()
        if not text:
            continue
        if _is_noise_log_line(text):
            continue
        cleaned.append(text)
    if len(cleaned) > 240:
        cleaned = cleaned[-240:]
    return cleaned


def _estimate_step_progress_percent(state, step_key, log_text, config_path_value):
    selected_model = (state.get("selected_model") or "").strip()
    if not selected_model:
        return 0
    if step_key == "resample":
        raw_total = _count_matching_files(Path("dataset_raw") / selected_model, "*.wav")
        if raw_total <= 0:
            return 0
        out_total = _count_matching_files(Path("dataset/44k") / selected_model, "*.wav")
        return max(0, min(99, int((out_total / raw_total) * 100)))
    if step_key == "flist_config":
        train_list = Path("filelists/train.txt")
        if not train_list.exists():
            return 20
        try:
            content = train_list.read_text(encoding="utf-8", errors="ignore")
            return 99 if f"/{selected_model}/" in content.replace("\\", "/") else 40
        except Exception:
            return 40
    if step_key == "hubert_f0":
        spk_dir = Path("dataset/44k") / selected_model
        wav_total = _count_matching_files(spk_dir, "*.wav")
        if wav_total <= 0:
            return 0
        soft_total = _count_matching_files(spk_dir, "*.wav.soft.pt")
        f0_total = _count_matching_files(spk_dir, "*.wav.f0.npy")
        done = min(soft_total, f0_total)
        return max(0, min(99, int((done / wav_total) * 100)))
    if step_key == "train":
        epoch_info = _extract_train_epoch_progress(log_text)
        total_epochs = _read_total_epochs(config_path_value)
        if not epoch_info or total_epochs <= 0:
            return 1
        epoch, batch_pct = epoch_info
        epoch_clamped = max(1, min(epoch, total_epochs))
        frac = ((epoch_clamped - 1) + (batch_pct / 100.0)) / max(1, total_epochs)
        return max(1, min(99, int(frac * 100)))
    return 0


def _build_progress_ui(state, log_text="", config_path_value="configs/config.json"):
    done_steps = sum(1 for s in TRAIN_STEP_ORDER if state["steps"][s].get("done"))
    active = (state.get("active_step") or "").strip()
    total_steps = len(TRAIN_STEP_ORDER)
    if active and active in TRAIN_STEP_ORDER:
        step_idx = TRAIN_STEP_ORDER.index(active)
        step_pct = _estimate_step_progress_percent(state, active, log_text, config_path_value)
        pipeline_pct = int(((step_idx + (step_pct / 100.0)) / total_steps) * 100)
        phase_text = f"Sedang berjalan: {TRAIN_STEP_LABELS.get(active, active)} ({step_pct}%)."
        return max(0, min(100, pipeline_pct)), phase_text
    pipeline_pct = int((done_steps / total_steps) * 100)
    if state.get("last_error"):
        return pipeline_pct, f"Perlu perhatian: {state['last_error']}"
    if done_steps == total_steps:
        return 100, "Selesai. Semua langkah training sudah selesai."
    next_step = next((s for s in TRAIN_STEP_ORDER if not state["steps"][s].get("done")), TRAIN_STEP_ORDER[-1])
    return pipeline_pct, f"Siap lanjut ke: {TRAIN_STEP_LABELS.get(next_step, next_step)}"


def _sync_state_with_dataset(state, selected_model=""):
    selected_model = (selected_model or state.get("selected_model") or "").strip()
    dataset_fp, file_count = _compute_dataset_fingerprint("dataset_raw", selected_model=selected_model)
    if dataset_fp and state.get("dataset_fingerprint") and state["dataset_fingerprint"] != dataset_fp:
        state["steps"] = {k: {"done": False, "updated_at": "", "exit_code": None} for k in TRAIN_STEP_ORDER}
        state["active_step"] = ""
        state["last_error"] = "Dataset berubah, progres step di-reset otomatis."
    if state.get("selected_model") and selected_model and state["selected_model"] != selected_model:
        state["steps"] = {k: {"done": False, "updated_at": "", "exit_code": None} for k in TRAIN_STEP_ORDER}
        state["active_step"] = ""
        state["last_error"] = f"Model diubah ke '{selected_model}', progres step model sebelumnya di-reset."
    state["selected_model"] = selected_model
    state["dataset_fingerprint"] = dataset_fp
    state["dataset_file_count"] = file_count
    return state


def _render_train_summary(state):
    lines = []
    if state.get("selected_model"):
        lines.append(f"Model dipilih: {state['selected_model']}")
    else:
        lines.append("Model dipilih: belum dipilih.")
    lines.append(f"File dataset_raw terdeteksi (model ini): {state.get('dataset_file_count', 0)}")
    pipeline_pct, _ = _build_progress_ui(state)
    lines.append(f"Progres pipeline: {pipeline_pct}%")
    if not state.get("dataset_fingerprint"):
        lines.append("Fingerprint dataset: belum ada (folder dataset_raw kosong atau belum ditemukan).")
    else:
        lines.append(f"Fingerprint dataset: {state['dataset_fingerprint'][:12]}...")
    for step in TRAIN_STEP_ORDER:
        info = state["steps"][step]
        status = "Selesai" if info.get("done") else "Belum"
        ts = info.get("updated_at") or "-"
        lines.append(f"{TRAIN_STEP_LABELS[step]}: {status} | update: {ts}")
    if state.get("active_step"):
        lines.append(f"Step aktif: {TRAIN_STEP_LABELS.get(state['active_step'], state['active_step'])}")
    if state.get("last_error"):
        lines.append(f"Catatan: {state['last_error']}")
    if not state["steps"]["resample"]["done"]:
        lines.append("Langkah berikutnya: jalankan '1) Siapkan Dataset Lokal (Step 1-3)'.")
    elif not state["steps"]["flist_config"]["done"]:
        lines.append("Langkah berikutnya: lanjutkan '1) Siapkan Dataset Lokal (Step 1-3)'.")
    elif not state["steps"]["hubert_f0"]["done"]:
        lines.append("Langkah berikutnya: lanjutkan '1) Siapkan Dataset Lokal (Step 1-3)'.")
    elif not state["steps"]["train"]["done"]:
        if torch.cuda.is_available():
            lines.append("Langkah berikutnya: jalankan training lokal (GPU CUDA terdeteksi).")
        else:
            lines.append("Langkah berikutnya: klik '2) Sync Artifacts ke Google Drive', lalu training di Google Colab.")
    else:
        lines.append("Semua langkah selesai untuk dataset saat ini.")
    return "\n".join(lines)


def refresh_train_status(selected_model_value=""):
    state = _load_train_state()
    selected_model, _ = _pick_selected_model(selected_model_value)
    state = _sync_state_with_dataset(state, selected_model=selected_model)
    state, _ = _clear_stale_active_step_if_needed(state)
    _save_train_state(state)
    return _render_train_summary(state)


def refresh_train_status_panel(selected_model_value="", config_path_value="configs/config.json"):
    state = _load_train_state()
    selected_model, _ = _pick_selected_model(selected_model_value)
    state = _sync_state_with_dataset(state, selected_model=selected_model)
    state, _ = _clear_stale_active_step_if_needed(state)
    _save_train_state(state)
    progress_pct, phase_text = _build_progress_ui(state, config_path_value=config_path_value)
    return _render_train_summary(state), progress_pct, phase_text


def refresh_train_status_with_models(selected_model_value="", config_path_value="configs/config.json"):
    selected_model, models = _pick_selected_model(selected_model_value)
    state = _load_train_state()
    state = _sync_state_with_dataset(state, selected_model=selected_model)
    state, _ = _clear_stale_active_step_if_needed(state)
    _save_train_state(state)
    progress_pct, phase_text = _build_progress_ui(state, config_path_value=config_path_value)
    status = _render_train_summary(state)
    return status, gr.Dropdown.update(choices=models, value=selected_model), progress_pct, phase_text


def stop_active_training_step(selected_model_value="", config_path_value="configs/config.json"):
    with TRAIN_RUNTIME["lock"]:
        TRAIN_RUNTIME["stop_requested"] = True
        proc = TRAIN_RUNTIME.get("proc")
    is_running = bool(proc and proc.poll() is None)
    if is_running:
        try:
            proc.terminate()
        except Exception:
            pass
    state = _load_train_state()
    selected_model, _ = _pick_selected_model(selected_model_value)
    state = _sync_state_with_dataset(state, selected_model=selected_model)
    if is_running:
        state["last_error"] = "Permintaan stop dikirim. Menunggu proses berhenti..."
    else:
        state, _ = _clear_stale_active_step_if_needed(
            state,
            reason_text="Step aktif dibersihkan karena proses train sudah tidak berjalan.",
        )
        with TRAIN_RUNTIME["lock"]:
            TRAIN_RUNTIME["proc"] = None
            TRAIN_RUNTIME["stop_requested"] = False
    _save_train_state(state)
    progress_pct, phase_text = _build_progress_ui(state, config_path_value=config_path_value)
    msg = "[STOP] Proses aktif dihentikan." if is_running else "[STOP] Tidak ada proses aktif. Lock status dibersihkan."
    return _render_train_summary(state), msg, progress_pct, phase_text


def reset_training_runtime(selected_model_value="", config_path_value="configs/config.json"):
    with TRAIN_RUNTIME["lock"]:
        proc = TRAIN_RUNTIME.get("proc")
        TRAIN_RUNTIME["stop_requested"] = True
    if proc and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
    state = _load_train_state()
    selected_model, _ = _pick_selected_model(selected_model_value)
    state = _sync_state_with_dataset(state, selected_model=selected_model)
    state, _ = _clear_stale_active_step_if_needed(
        state,
        reason_text="Session runtime di-reset manual. Silakan jalankan step lagi.",
    )
    state["active_step"] = ""
    _save_train_state(state)
    with TRAIN_RUNTIME["lock"]:
        TRAIN_RUNTIME["proc"] = None
        TRAIN_RUNTIME["stop_requested"] = False
    progress_pct, phase_text = _build_progress_ui(state, config_path_value=config_path_value)
    return _render_train_summary(state), "[RESET] Runtime training berhasil di-reset.", progress_pct, phase_text


def sync_artifacts_to_drive_stream_panel(drive_sync_path, model_name_value, config_path_value):
    for status_text, logs_text in sync_artifacts_to_drive_stream(drive_sync_path, model_name_value):
        state = _load_train_state()
        selected_model, _ = _pick_selected_model(model_name_value)
        state = _sync_state_with_dataset(state, selected_model=selected_model)
        _save_train_state(state)
        progress_pct, phase_text = _build_progress_ui(state, logs_text, config_path_value)
        yield status_text, logs_text, progress_pct, phase_text


def _default_model_registry():
    return {"models": {}}


def _load_model_registry():
    if not os.path.exists(MODEL_REGISTRY_PATH):
        return _default_model_registry()
    try:
        with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return _default_model_registry()
    if "models" not in data or not isinstance(data["models"], dict):
        data["models"] = {}
    return data


def _save_model_registry(data):
    os.makedirs(os.path.dirname(MODEL_REGISTRY_PATH), exist_ok=True)
    with open(MODEL_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _is_prepared_output_exists(model_name):
    spk_dir = Path("dataset/44k") / model_name
    if not spk_dir.exists():
        return False
    return any(spk_dir.glob("*.wav"))


def _cleanup_model_registry(data):
    removed = []
    for model_name in list(data.get("models", {}).keys()):
        if not _is_prepared_output_exists(model_name):
            data["models"].pop(model_name, None)
            removed.append(model_name)
    return data, removed


def _run_command_stream(cmd, step_key, state, log_buffer):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["active_step"] = step_key
    state["last_error"] = ""
    state["steps"][step_key]["updated_at"] = now
    _save_train_state(state)
    with TRAIN_RUNTIME["lock"]:
        TRAIN_RUNTIME["stop_requested"] = False
    proc = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with TRAIN_RUNTIME["lock"]:
        TRAIN_RUNTIME["proc"] = proc
    stopped_by_user = False
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        line_text = line.rstrip("\n")
        if not _is_noise_log_line(line_text):
            log_buffer.append(line_text)
        if len(log_buffer) > 400:
            log_buffer[:] = log_buffer[-400:]
        with TRAIN_RUNTIME["lock"]:
            stop_requested = TRAIN_RUNTIME["stop_requested"]
        if stop_requested and proc.poll() is None:
            stopped_by_user = True
            try:
                proc.terminate()
            except Exception:
                pass
        ui_logs = _trim_server_logs_for_ui(log_buffer)
        log_text_ui = "\n".join(ui_logs)
        progress_pct, phase_text = _build_progress_ui(state, log_text_ui, cmd[3] if step_key == "train" and len(cmd) > 3 else "configs/config.json")
        yield _render_train_summary(state), log_text_ui, progress_pct, phase_text
    proc.wait()
    exit_code = proc.returncode
    state["active_step"] = ""
    state["steps"][step_key]["exit_code"] = exit_code
    state["steps"][step_key]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if exit_code == 0:
        state["steps"][step_key]["done"] = True
        state["last_error"] = ""
    else:
        if stopped_by_user:
            state["last_error"] = f"Step {TRAIN_STEP_LABELS.get(step_key, step_key)} dihentikan pengguna."
        else:
            state["last_error"] = f"Step {TRAIN_STEP_LABELS.get(step_key, step_key)} gagal (exit code {exit_code})."
    _save_train_state(state)
    with TRAIN_RUNTIME["lock"]:
        TRAIN_RUNTIME["proc"] = None
        TRAIN_RUNTIME["stop_requested"] = False
    ui_logs = _trim_server_logs_for_ui(log_buffer)
    log_text_ui = "\n".join(ui_logs)
    progress_pct, phase_text = _build_progress_ui(state, log_text_ui, cmd[3] if step_key == "train" and len(cmd) > 3 else "configs/config.json")
    yield _render_train_summary(state), log_text_ui, progress_pct, phase_text


def run_train_step_stream(step_key, force_rerun, config_path_value, model_name_value, selected_model_value):
    state = _load_train_state()
    selected_model, _ = _pick_selected_model(selected_model_value)
    state = _sync_state_with_dataset(state, selected_model=selected_model)
    state, _ = _clear_stale_active_step_if_needed(state)
    if not selected_model:
        progress_pct, phase_text = _build_progress_ui(state, config_path_value=config_path_value)
        yield _render_train_summary(state), "Tidak ada folder model di dataset_raw. Isi dataset_raw dulu.", progress_pct, phase_text
        return
    if state.get("active_step"):
        msg = f"Masih ada step berjalan: {TRAIN_STEP_LABELS.get(state['active_step'], state['active_step'])}."
        state["last_error"] = msg
        _save_train_state(state)
        progress_pct, phase_text = _build_progress_ui(state, config_path_value=config_path_value)
        yield _render_train_summary(state), msg, progress_pct, phase_text
        return
    if not step_key or step_key not in TRAIN_STEP_ORDER:
        progress_pct, phase_text = _build_progress_ui(state, config_path_value=config_path_value)
        yield _render_train_summary(state), "Pilih step yang valid.", progress_pct, phase_text
        return
    if step_key == "train" and not torch.cuda.is_available():
        progress_pct, phase_text = _build_progress_ui(state, config_path_value=config_path_value)
        yield _render_train_summary(state), "Training lokal butuh GPU CUDA (NVIDIA). Perangkat ini tidak mendukung CUDA. Lanjutkan training di Google Colab.", progress_pct, phase_text
        return
    if state.get("dataset_file_count", 0) == 0:
        progress_pct, phase_text = _build_progress_ui(state, config_path_value=config_path_value)
        yield _render_train_summary(state), f"Folder dataset_raw/{selected_model} kosong. Isi data dulu sebelum training.", progress_pct, phase_text
        return
    if state["steps"][step_key]["done"] and not force_rerun:
        progress_pct, phase_text = _build_progress_ui(state, config_path_value=config_path_value)
        yield _render_train_summary(state), f"Step '{TRAIN_STEP_LABELS[step_key]}' sudah selesai untuk dataset ini. Centang 'Paksa Ulang Step' jika ingin menjalankan lagi.", progress_pct, phase_text
        return
    missing = [req for req in TRAIN_STEP_REQUIREMENTS[step_key] if not state["steps"][req]["done"]]
    if missing and not force_rerun:
        missing_label = ", ".join(TRAIN_STEP_LABELS[m] for m in missing)
        progress_pct, phase_text = _build_progress_ui(state, config_path_value=config_path_value)
        yield _render_train_summary(state), f"Step belum siap. Selesaikan dulu: {missing_label}.", progress_pct, phase_text
        return

    step_cmd = {
        "resample": [
            sys.executable,
            "resample.py",
            "--in_dir",
            "dataset_raw",
            "--out_dir2",
            "dataset/44k",
            "--speaker_filter",
            selected_model,
        ],
        "flist_config": [
            sys.executable,
            "preprocess_flist_config.py",
            "--source_dir",
            "dataset/44k",
            "--speaker_filter",
            selected_model,
        ],
        "hubert_f0": [
            sys.executable,
            "preprocess_hubert_f0.py",
            "--in_dir",
            "dataset/44k",
            "--speaker_filter",
            selected_model,
        ],
        "train": [sys.executable, "train.py", "-c", (config_path_value or "configs/config.json"), "-m", (model_name_value or "44k")],
    }
    cmd = step_cmd[step_key]
    log_buffer = [f"[START] {' '.join(cmd)}"]
    _save_train_state(state)
    for out in _run_command_stream(cmd, step_key, state, log_buffer):
        yield out


def run_next_pending_step_stream(force_rerun, config_path_value, model_name_value, selected_model_value):
    state = _load_train_state()
    selected_model, _ = _pick_selected_model(selected_model_value)
    state = _sync_state_with_dataset(state, selected_model=selected_model)
    next_step = None
    for step in TRAIN_STEP_ORDER:
        if not state["steps"][step]["done"]:
            next_step = step
            break
    if next_step is None and not force_rerun:
        progress_pct, phase_text = _build_progress_ui(state, config_path_value=config_path_value)
        yield _render_train_summary(state), "Semua step sudah selesai untuk dataset ini.", progress_pct, phase_text
        return
    target = next_step or "train"
    for out in run_train_step_stream(target, force_rerun, config_path_value, model_name_value, selected_model):
        yield out


def run_local_train_stream(force_rerun, config_path_value, model_name_value, selected_model_value):
    selected_model, _ = _pick_selected_model(selected_model_value)
    for out in run_train_step_stream("train", force_rerun, config_path_value, model_name_value, selected_model):
        yield out


def run_local_prepare_stream(force_rerun, sanitize_before, config_path_value, model_name_value, selected_model_value):
    selected_model, models = _pick_selected_model(selected_model_value)
    if not selected_model:
        status, progress_pct, phase_text = refresh_train_status_panel("", config_path_value)
        yield status, "Tidak ada folder model di dataset_raw. Tambahkan dulu dataset_raw/<nama_model>*.wav", progress_pct, phase_text
        return
    registry = _load_model_registry()
    registry, removed = _cleanup_model_registry(registry)
    if removed:
        _save_model_registry(registry)
    if selected_model in registry.get("models", {}) and not force_rerun:
        msg = (
            f"Model '{selected_model}' sudah pernah disiapkan (tercatat di JSON). "
            "Tidak dijalankan ulang. Hapus dataset/44k model ini atau centang 'Paksa Ulang Preprocess' jika ingin ulang."
        )
        status, progress_pct, phase_text = refresh_train_status_panel(selected_model, config_path_value)
        yield status, msg, progress_pct, phase_text
        return
    if sanitize_before:
        for status_text, logs_text in sanitize_existing_dataset_stream(selected_model):
            state = _load_train_state()
            progress_pct, phase_text = _build_progress_ui(state, logs_text, config_path_value)
            yield status_text, logs_text, progress_pct, phase_text
    for step in ["resample", "flist_config", "hubert_f0"]:
        for out in run_train_step_stream(step, force_rerun, config_path_value, model_name_value, selected_model):
            yield out
    state = _load_train_state()
    state = _sync_state_with_dataset(state, selected_model=selected_model)
    _save_train_state(state)
    if state["steps"]["hubert_f0"]["done"]:
        registry = _load_model_registry()
        registry, _ = _cleanup_model_registry(registry)
        registry["models"][selected_model] = {
            "prepared_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_raw_fingerprint": state.get("dataset_fingerprint", ""),
            "dataset_raw_file_count": state.get("dataset_file_count", 0),
        }
        _save_model_registry(registry)
        progress_pct, phase_text = _build_progress_ui(state, config_path_value=config_path_value)
        yield _render_train_summary(state), f"[DONE] Model '{selected_model}' dicatat sebagai sudah disiapkan.", progress_pct, phase_text


def sanitize_existing_dataset_stream(target_model=""):
    speaker_root = Path("dataset/44k")
    if not speaker_root.exists():
        yield refresh_train_status(target_model), "Folder dataset/44k tidak ditemukan."
        return
    target_model = (target_model or "").strip()
    if target_model:
        target_dir = speaker_root / target_model
        if not target_dir.exists():
            yield refresh_train_status(target_model), f"[SKIP] Folder dataset/44k/{target_model} belum ada. Lanjut tanpa sanitize."
            return
        speakers = [target_dir]
    else:
        speakers = sorted([p for p in speaker_root.iterdir() if p.is_dir()])
    if not speakers:
        yield refresh_train_status(target_model), "Tidak ada folder speaker di dataset/44k."
        return

    total_groups = 0
    total_files = 0
    logs = []

    for spk_dir in speakers:
        speaker_name = _sanitize_filename_stem(spk_dir.name, max_len=48)
        wavs = sorted(spk_dir.glob("*.wav"))
        if not wavs:
            logs.append(f"[SKIP] {spk_dir.name}: tidak ada wav.")
            continue

        mapping = {}
        for i, wav in enumerate(wavs, start=1):
            mapping[wav.stem] = f"{speaker_name}_{i:04d}"

        rename_pairs = []
        for old_base, new_base in mapping.items():
            prefix = old_base + "."
            for p in spk_dir.iterdir():
                if p.is_file() and p.name.startswith(prefix):
                    suffix = p.name[len(old_base):]
                    rename_pairs.append((p, spk_dir / f"{new_base}{suffix}"))

        if not rename_pairs:
            logs.append(f"[SKIP] {spk_dir.name}: tidak ada file yang perlu diubah.")
            continue

        target_lower = [str(dst).lower() for _, dst in rename_pairs]
        if len(target_lower) != len(set(target_lower)):
            logs.append(f"[ERROR] {spk_dir.name}: target rename bentrok, dibatalkan.")
            continue

        salt = hashlib.md5(f"{spk_dir}-{time.time()}".encode("utf-8")).hexdigest()[:8]
        temp_pairs = []
        for src, dst in rename_pairs:
            tmp = src.with_name(f"__tmp_{salt}_{src.name}")
            os.replace(src, tmp)
            temp_pairs.append((tmp, dst))
        for tmp, dst in temp_pairs:
            os.replace(tmp, dst)

        total_groups += len(mapping)
        total_files += len(rename_pairs)
        logs.append(f"[OK] {spk_dir.name}: {len(mapping)} grup, {len(rename_pairs)} file direname.")
        yield refresh_train_status(target_model), "\n".join(logs[-80:])

    logs.append(f"[DONE] Total grup: {total_groups}, total file: {total_files}.")
    logs.append("[NEXT] Regenerate filelist/config...")
    yield refresh_train_status(target_model), "\n".join(logs[-120:])

    proc = subprocess.Popen(
        [sys.executable, "preprocess_flist_config.py"],
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        logs.append(line.rstrip("\n"))
        if len(logs) > 200:
            logs = logs[-200:]
        yield refresh_train_status(target_model), "\n".join(logs)
    proc.wait()
    if proc.returncode == 0:
        logs.append("[DONE] preprocess_flist_config.py sukses.")
    else:
        logs.append(f"[ERROR] preprocess_flist_config.py gagal (exit {proc.returncode}).")
    yield refresh_train_status(target_model), "\n".join(logs[-200:])


def debug_change():
    global debug
    debug = debug_button.value


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
        primary_hue = gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New']
    ),
    css=APP_CSS,
) as app:
    with gr.Tabs():
        with gr.TabItem("Inferensi So-VITS"):
            gr.Markdown(value="""
                So-vits-svc 4.0 Inferensi WebUI
                """)
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=2> Pengaturan Model</font>
                        """)
                    model_path = gr.File(label="Pilih file model")
                    config_path = gr.File(label="Pilih file konfigurasi")
                    cluster_model_path = gr.File(label="Pilih file model clustering (opsional, boleh tidak dipilih)")
                    device = gr.Dropdown(label="Perangkat inferensi, default otomatis memilih CPU dan GPU", choices=["Auto",*cuda.keys(),"CPU"], value="Auto")
                    enhance = gr.Checkbox(label="Gunakan peningkatan NSF_HIFIGAN. Opsi ini dapat meningkatkan kualitas suara untuk model dengan data latih sedikit, tetapi berdampak negatif pada model yang sudah terlatih baik. Default: nonaktif", value=False)
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=3>Setelah semua file di sebelah kiri dipilih (semua modul file menampilkan download), klik "Muat Model" untuk memproses:</font>
                        """)
                    model_load_button = gr.Button(value="Muat Model", variant="primary")
                    model_unload_button = gr.Button(value="Copot Model", variant="primary")
                    sid = gr.Dropdown(label="Warna Suara (Pembicara)")
                    sid_output = gr.Textbox(label="Pesan Output")


            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=2> Pengaturan Inferensi</font>
                        """)
                    auto_f0 = gr.Checkbox(label="Prediksi f0 otomatis, bekerja lebih baik dengan model clustering. Akan menonaktifkan fungsi transposisi (hanya untuk konversi suara bicara, JANGAN aktifkan untuk lagu karena akan menyebabkan nada sangat meleset)", value=False)
                    F0_mean_pooling = gr.Checkbox(label="Gunakan filter rata-rata (pooling) pada F0, dapat memperbaiki beberapa suara serak. Perhatian: mengaktifkan opsi ini akan memperlambat kecepatan inferensi. Default: nonaktif", value=False)
                    vc_transform = gr.Number(label="Transposisi (bilangan bulat, bisa positif/negatif, dalam satuan semitone, naik 1 oktaf = 12)", value=0)
                    cluster_ratio = gr.Number(label="Rasio campuran model clustering, antara 0-1, 0 berarti tidak menggunakan clustering. Menggunakan clustering dapat meningkatkan kemiripan warna suara, tetapi mengurangi kejelasan pengucapan (disarankan sekitar 0.5)", value=0)
                    slice_db = gr.Number(label="Ambang batas pemotongan (slice threshold)", value=-40)
                    noise_scale = gr.Number(label="noise_scale - disarankan tidak diubah, mempengaruhi kualitas suara, parameter eksperimental", value=0.4)
                with gr.Column():
                    pad_seconds = gr.Number(label="Padding audio inferensi (detik). Karena alasan tertentu, awal dan akhir audio bisa berisik. Menambahkan sedikit jeda hening akan menghilangkannya", value=0.5)
                    cl_num = gr.Number(label="Pemotongan audio otomatis, 0 berarti tidak memotong, satuan dalam detik (s)", value=0)
                    lg_num = gr.Number(label="Panjang crossfade antar potongan audio. Jika suara terputus-putus setelah pemotongan otomatis, sesuaikan nilai ini. Jika sudah lancar, gunakan default 0. Perhatian: pengaturan ini mempengaruhi kecepatan inferensi (detik)", value=0)
                    lgr_num = gr.Number(label="Setelah pemotongan audio otomatis, bagian awal dan akhir setiap potongan perlu dibuang. Parameter ini mengatur rasio panjang crossfade yang dipertahankan, rentang 0-1", value=0.75)
                    enhancer_adaptive_key = gr.Number(label="Buat enhancer menyesuaikan nada yang lebih tinggi (satuan semitone) | Default: 0", value=0)
                    cr_threshold = gr.Number(label="Ambang batas filter F0, hanya efektif saat f0_mean_pooling diaktifkan. Rentang 0-1. Menurunkan nilai ini mengurangi kemungkinan nada meleset, tetapi meningkatkan suara serak", value=0.05)
            with gr.Tabs():
                with gr.TabItem("Audio ke Audio"):
                    vc_input3 = gr.Audio(label="Pilih Audio")
                    vc_submit = gr.Button("Konversi Audio", variant="primary")
                with gr.TabItem("Teks ke Audio"):
                    text2tts=gr.Textbox(label="Masukkan teks yang ingin dikonversi di sini. Catatan: disarankan mengaktifkan prediksi F0 saat menggunakan fitur ini, jika tidak hasilnya akan terdengar aneh")
                    tts_rate = gr.Number(label="Kecepatan TTS", value=0)
                    tts_voice = gr.Radio(label="Jenis Kelamin",choices=["Pria","Wanita"], value="Pria")
                    vc_submit2 = gr.Button("Konversi Teks", variant="primary")
            with gr.Row():
                with gr.Column():
                    vc_output1 = gr.Textbox(label="Pesan Output")
                with gr.Column():
                    vc_output2 = gr.Audio(label="Audio Output", interactive=False)

        with gr.TabItem("Inferensi RVC"):
            gr.Markdown(value="""
                RVC Inferensi (v1/v2) - Muat model `.zip` atau `.pth` + `.index` opsional.
                """)
            rvc_available = rvc_model_state["backend"].is_available
            rvc_notice = gr.Markdown(value=rvc_dependency_notice())
            with gr.Row(variant="panel"):
                with gr.Column():
                    rvc_zip_model_path = gr.File(label="Pilih ZIP model RVC (.zip) [opsional]")
                    rvc_model_path = gr.File(label="Pilih file model RVC (.pth)")
                    rvc_index_path = gr.File(label="Pilih file index RVC (.index) [opsional]")
                    rvc_device = gr.Dropdown(
                        label="Perangkat inferensi RVC",
                        choices=["Auto", *cuda.keys(), "CPU"],
                        value="Auto",
                    )
                with gr.Column():
                    rvc_model_load_button = gr.Button(
                        value="Muat Model RVC",
                        variant="primary",
                        interactive=rvc_available,
                    )
                    rvc_model_unload_button = gr.Button(
                        value="Copot Model RVC",
                        variant="secondary",
                        interactive=rvc_available,
                    )
                    rvc_status_output = gr.Textbox(
                        label="Pesan Output",
                        value="" if rvc_available else rvc_dependency_notice(),
                    )
            with gr.Row(variant="panel"):
                rvc_transpose = gr.Number(
                    label="Transpose / semitone",
                    value=0,
                )
                rvc_index_rate = gr.Number(
                    label="index_rate (0-1)",
                    value=0.75,
                )
                rvc_protect = gr.Number(
                    label="protect (0-0.5 disarankan)",
                    value=0.33,
                )
            with gr.Tabs():
                with gr.TabItem("Audio ke Voice RVC"):
                    rvc_input_audio = gr.Audio(label="Pilih Audio")
                    rvc_submit = gr.Button(
                        "Konversi Audio RVC",
                        variant="primary",
                        interactive=rvc_available,
                    )
                with gr.TabItem("Teks ke Voice RVC"):
                    rvc_tts_text = gr.Textbox(label="Masukkan teks")
                    with gr.Row():
                        rvc_tts_rate = gr.Number(label="Kecepatan TTS", value=0)
                        rvc_tts_voice = gr.Radio(label="Jenis Kelamin", choices=["Pria", "Wanita"], value="Pria")
                    rvc_tts_submit = gr.Button(
                        "Konversi Teks ke Voice RVC",
                        variant="primary",
                        interactive=rvc_available,
                    )
            with gr.Row():
                with gr.Column():
                    rvc_output1 = gr.Textbox(label="Pesan Output")
                with gr.Column():
                    rvc_output2 = gr.Audio(label="Audio Output", interactive=False)

        with gr.TabItem("Full Song RVC"):
            gr.Markdown(value="""
                Lagu penuh -> pisahkan vocal & instrumental -> ubah vocal dengan RVC -> gabungkan lagi.
                """)
            full_song_available = rvc_model_state["backend"].is_available and demucs_is_available()
            full_song_notice = gr.Markdown(value=full_song_rvc_notice())
            with gr.Row(variant="panel"):
                with gr.Column():
                    full_song_zip_model_path = gr.File(label="Pilih ZIP model RVC (.zip) [opsional]")
                    full_song_model_path = gr.File(label="Pilih file model RVC (.pth)")
                    full_song_index_path = gr.File(label="Pilih file index RVC (.index) [opsional]")
                    full_song_input = gr.File(label="Pilih Lagu Full (audio)")
                    full_song_rvc_device = gr.Dropdown(
                        label="Perangkat inferensi RVC",
                        choices=["Auto", *cuda.keys(), "CPU"],
                        value="Auto",
                    )
                    full_song_demucs_device = gr.Dropdown(
                        label="Perangkat stem separation (Demucs)",
                        choices=["Auto", *cuda.keys(), "CPU"],
                        value="Auto",
                    )
                with gr.Column():
                    full_song_transpose = gr.Number(label="Transpose / semitone", value=0)
                    full_song_index_rate = gr.Number(label="index_rate (0-1)", value=0.75)
                    full_song_protect = gr.Number(label="protect (0-0.5 disarankan)", value=0.33)
            with gr.Row(variant="panel"):
                full_song_vocal_gain = gr.Number(label="Gain vocal hasil", value=1.0)
                full_song_instrumental_gain = gr.Number(label="Gain instrumental", value=1.0)
                full_song_normalize = gr.Checkbox(label="Normalize hasil gabungan", value=True)
            full_song_submit = gr.Button(
                "Konversi Lagu Full dengan RVC",
                variant="primary",
                interactive=full_song_available,
            )
            with gr.Row():
                with gr.Column():
                    full_song_output_text = gr.Textbox(label="Pesan Output", lines=6)
                    full_song_mix_output = gr.Audio(label="Hasil Gabungan", interactive=False, type="filepath")
                with gr.Column():
                    full_song_vocal_output = gr.Audio(label="Vocal Hasil RVC", interactive=False, type="filepath")
                    full_song_instrumental_output = gr.Audio(label="Instrumental", interactive=False, type="filepath")

        with gr.TabItem("Alat Bantu / Fitur Eksperimental"):
            gr.Markdown(value="""
                        <font size=2> So-vits-svc 4.0 Alat Bantu / Fitur Eksperimental</font>
                        """)
            with gr.Tabs():
                with gr.TabItem("Penggabungan Suara Statis"):
                    gr.Markdown(value="""
                        <font size=2> Penjelasan: Fitur ini dapat menggabungkan beberapa model suara menjadi satu model suara (kombinasi konveks atau linear dari parameter model), sehingga menghasilkan warna suara yang tidak ada di dunia nyata.
                                          Catatan:
                                          1. Fitur ini hanya mendukung model dengan satu pembicara
                                          2. Jika menggunakan model multi-pembicara, pastikan jumlah pembicara di semua model sama, sehingga suara di bawah SpeakerID yang sama dapat dicampur
                                          3. Pastikan field "model" di config.json semua model yang akan dicampur adalah sama
                                          4. Model hasil penggabungan dapat menggunakan config.json dari salah satu model, tetapi model clustering tidak dapat digunakan
                                          5. Saat mengunggah model secara massal, sebaiknya tempatkan semua model dalam satu folder lalu unggah sekaligus
                                          6. Rasio campuran disarankan antara 0-100, bisa juga angka lain, tetapi pada mode kombinasi linear dapat menghasilkan efek yang tidak terduga
                                          7. Setelah penggabungan selesai, file akan disimpan di direktori utama proyek dengan nama output.pth
                                          8. Mode kombinasi konveks akan menerapkan Softmax pada rasio campuran sehingga totalnya menjadi 1, sedangkan mode kombinasi linear tidak
                        </font>
                        """)
                    mix_model_path = gr.Files(label="Pilih file model yang akan dicampur")
                    mix_model_upload_button = gr.UploadButton("Pilih/Tambah file model yang akan dicampur", file_count="multiple", variant="primary")
                    mix_model_output1 = gr.Textbox(
                                            label="Penyesuaian rasio campuran, satuan/%",
                                            interactive = True
                                         )
                    mix_mode = gr.Radio(choices=["Kombinasi Konveks", "Kombinasi Linear"], label="Mode Penggabungan",value="Kombinasi Konveks",interactive = True)
                    mix_submit = gr.Button("Mulai Penggabungan Suara", variant="primary")
                    mix_model_output2 = gr.Textbox(
                                            label="Pesan Output"
                                         )
                    mix_model_path.change(updata_mix_info,[mix_model_path],[mix_model_output1])
                    mix_model_upload_button.upload(upload_mix_append_file, [mix_model_upload_button,mix_model_path], [mix_model_path,mix_model_output1])
                    mix_submit.click(mix_submit_click, [mix_model_output1,mix_mode], [mix_model_output2])
                with gr.TabItem("Audio Slicer (Pemotong Audio)"):
                    gr.Markdown(value="""
                        <font size=2> Penggunaan: Memotong audio panjang menjadi potongan pendek (5-15 detik) berdasarkan jeda diam untuk dataset training. Proses berjalan di background dan progres akan diperbarui otomatis. </font>
                        """)
                    with gr.Row():
                        slicer_input = gr.File(label="Pilih Audio Panjang", file_types=["audio"], type="file")
                        slicer_speaker = gr.Textbox(label="Nama Speaker/Artis (Folder)", placeholder="Contoh: artis_a")
                    with gr.Row():
                        slicer_db = gr.Number(label="Ambang Batas Diam (dB)", value=-40)
                        slicer_min_len = gr.Number(label="Panjang Minimum (ms)", value=5000)
                    slicer_submit = gr.Button("Mulai Pemotongan Audio (Background)", variant="primary")
                    slicer_progress = gr.Slider(label="Progres Pemotongan (%)", minimum=0, maximum=100, value=0, step=1, interactive=False)
                    slicer_output = gr.Textbox(label="Status Pemotongan", lines=3)
                with gr.TabItem("Training Manager"):
                    gr.Markdown(value="""
                        <div class='tm-hero'>
                          <b>Training untuk pemula (3 klik utama)</b>
                          <ol class='tm-steps'>
                            <li>Klik <b>Siapkan Dataset</b> sekali sampai Step 1-3 selesai.</li>
                            <li>(Opsional) klik <b>Sync ke Drive</b> kalau ingin lanjut training di Colab.</li>
                            <li>Klik <b>Train Lokal</b> jika GPU NVIDIA tersedia.</li>
                          </ol>
                        </div>
                        """)
                    default_model, initial_models = _pick_selected_model("")
                    initial_status, initial_progress, initial_phase = refresh_train_status_panel(default_model, "configs/config.json")
                    with gr.Row():
                        force_rerun = gr.Checkbox(label="Paksa Ulang Step", value=False)
                        sanitize_before = gr.Checkbox(label="Rapikan Nama File Dulu", value=True)
                    target_dataset_model = gr.Dropdown(
                        label="Folder Model dari dataset_raw (satu model per proses)",
                        choices=initial_models,
                        value=default_model,
                    )
                    train_target_hint = gr.Markdown(value=render_train_target_label(default_model))
                    train_progress = gr.Slider(label="Loading Bar Progres Pipeline (%)", minimum=0, maximum=100, value=initial_progress, step=1, interactive=False, elem_id="tm-progress")
                    train_phase = gr.Textbox(label="Status Berjalan (bahasa sederhana)", lines=2, value=initial_phase)
                    gr.Markdown(value="<div class='tm-log-note'>Status dan loading bar auto-refresh setiap 5 detik.</div>")
                    with gr.Accordion("Pengaturan Lanjutan", open=False):
                        with gr.Row():
                            train_config_path = gr.Textbox(label="Path Config Training", value="configs/config.json")
                            train_model_name = gr.Textbox(label="Nama Folder Model", value="44k")
                    drive_sync_path = gr.Textbox(label="Path Drive Sync (Desktop, parent folder)", value=DEFAULT_DRIVE_SYNC_PATH)
                    with gr.Row():
                        refresh_train_btn = gr.Button("Refresh Status", variant="secondary")
                        run_prepare_btn = gr.Button("1) Siapkan Dataset (Step 1-3)", variant="primary")
                        sync_drive_btn = gr.Button("2) Sync Artifacts ke Google Drive")
                        run_train_btn = gr.Button("3) Train Model (GPU lokal)", variant="primary")
                        stop_train_btn = gr.Button("Stop Proses Aktif", variant="secondary")
                        reset_runtime_btn = gr.Button("Reset Session Runtime", variant="secondary")
                    train_status = gr.Textbox(label="Status Pipeline", lines=10, value=initial_status)
                    gr.Markdown(value="<div class='tm-log-note'>Menampilkan log server training yang penting (warning teknis berulang disembunyikan).</div>")
                    train_logs = gr.Textbox(label="Log Server Training (tail)", lines=18)

        with gr.Row(variant="panel"):
            with gr.Column():
                gr.Markdown(value="""
                    <font size=2> Pengaturan WebUI</font>
                    """)
                debug_button = gr.Checkbox(label="Mode Debug. Aktifkan jika ingin melaporkan bug ke komunitas. Setelah diaktifkan, konsol akan menampilkan pesan error yang lebih detail", value=debug)
        vc_submit.click(vc_fn, [sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,F0_mean_pooling,enhancer_adaptive_key,cr_threshold], [vc_output1, vc_output2])
        vc_submit2.click(vc_fn2, [sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,text2tts,tts_rate,tts_voice,F0_mean_pooling,enhancer_adaptive_key,cr_threshold], [vc_output1, vc_output2])
        debug_button.change(debug_change,[],[])
        model_load_button.click(modelAnalysis,[model_path,config_path,cluster_model_path,device,enhance],[sid,sid_output])
        model_unload_button.click(modelUnload,[],[sid,sid_output])
        rvc_model_load_button.click(
            rvc_model_load,
            [rvc_zip_model_path, rvc_model_path, rvc_index_path, rvc_device],
            [rvc_status_output],
        )
        rvc_model_unload_button.click(
            rvc_model_unload,
            [],
            [rvc_status_output],
        )
        rvc_submit.click(
            rvc_vc_fn,
            [rvc_zip_model_path, rvc_model_path, rvc_index_path, rvc_device, rvc_input_audio, rvc_transpose, rvc_index_rate, rvc_protect],
            [rvc_output1, rvc_output2],
        )
        rvc_tts_submit.click(
            rvc_tts_fn,
            [
                rvc_zip_model_path,
                rvc_model_path,
                rvc_index_path,
                rvc_device,
                rvc_tts_text,
                rvc_tts_rate,
                rvc_tts_voice,
                rvc_transpose,
                rvc_index_rate,
                rvc_protect,
            ],
            [rvc_output1, rvc_output2],
        )
        full_song_submit.click(
            full_song_rvc_fn,
            [
                full_song_zip_model_path,
                full_song_model_path,
                full_song_index_path,
                full_song_rvc_device,
                full_song_demucs_device,
                full_song_input,
                full_song_transpose,
                full_song_index_rate,
                full_song_protect,
                full_song_vocal_gain,
                full_song_instrumental_gain,
                full_song_normalize,
            ],
            [
                full_song_output_text,
                full_song_mix_output,
                full_song_vocal_output,
                full_song_instrumental_output,
            ],
        )
        slicer_submit.click(slice_audio_stream, [slicer_input, slicer_speaker, slicer_db, slicer_min_len], [slicer_output, slicer_progress], queue=True)
        refresh_train_btn.click(
            refresh_train_status_with_models,
            [target_dataset_model, train_config_path],
            [train_status, target_dataset_model, train_progress, train_phase],
            queue=False,
        )
        refresh_train_btn.click(
            render_train_target_label,
            [target_dataset_model],
            [train_target_hint],
            queue=False,
        )
        target_dataset_model.change(
            refresh_train_status_with_models,
            [target_dataset_model, train_config_path],
            [train_status, target_dataset_model, train_progress, train_phase],
            queue=False,
        )
        train_config_path.change(
            refresh_train_status_panel,
            [target_dataset_model, train_config_path],
            [train_status, train_progress, train_phase],
            queue=False,
        )
        app.load(
            refresh_train_status_panel,
            [target_dataset_model, train_config_path],
            [train_status, train_progress, train_phase],
            every=5,
            queue=False,
        )
        target_dataset_model.change(
            render_train_target_label,
            [target_dataset_model],
            [train_target_hint],
            queue=False,
        )
        run_prepare_btn.click(
            run_local_prepare_stream,
            [force_rerun, sanitize_before, train_config_path, train_model_name, target_dataset_model],
            [train_status, train_logs, train_progress, train_phase],
            queue=True,
        )
        sync_drive_btn.click(
            sync_artifacts_to_drive_stream_panel,
            [drive_sync_path, target_dataset_model, train_config_path],
            [train_status, train_logs, train_progress, train_phase],
            queue=True,
        )
        run_train_btn.click(
            run_local_train_stream,
            [force_rerun, train_config_path, train_model_name, target_dataset_model],
            [train_status, train_logs, train_progress, train_phase],
            queue=True,
        )
        stop_train_btn.click(
            stop_active_training_step,
            [target_dataset_model, train_config_path],
            [train_status, train_logs, train_progress, train_phase],
            queue=False,
        )
        reset_runtime_btn.click(
            reset_training_runtime,
            [target_dataset_model, train_config_path],
            [train_status, train_logs, train_progress, train_phase],
            queue=False,
        )
if __name__ == "__main__":
    _acquire_webui_lock_or_exit()
    app.queue().launch(server_name="127.0.0.1", show_api=False)
