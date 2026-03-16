import io
import os
import sys
import shutil
import zipfile

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import gradio.processing_utils as gr_pu
import librosa
import numpy as np
import soundfile
from inference.infer_tool import Svc
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

model = None
spk = None
debug = False
TRAIN_STATE_PATH = "logs/webui_train_state.json"
MODEL_REGISTRY_PATH = "logs/webui_prepared_models.json"
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

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"

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
    global model
    try:
        device = cuda[device] if "CUDA" in device else device
        model = Svc(model_path.name, config_path.name, device=device if device!="Auto" else None, cluster_model_path = cluster_model_path.name if cluster_model_path != None else "",nsf_hifigan_enhance=enhance)
        spks = list(model.spk2id.keys())
        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
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
    global model
    if model is None:
        return sid.update(choices = [],value=""),"Tidak ada model yang perlu dicopot!"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return sid.update(choices = [],value=""),"Model berhasil dicopot!"


def vc_fn(sid, input_audio, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,F0_mean_pooling,enhancer_adaptive_key,cr_threshold):
    global model
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
    _save_train_state(state)
    return _render_train_summary(state)


def refresh_train_status_with_models(selected_model_value=""):
    selected_model, models = _pick_selected_model(selected_model_value)
    status = refresh_train_status(selected_model)
    return status, gr.Dropdown.update(choices=models, value=selected_model)


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
    proc = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        log_buffer.append(line.rstrip("\n"))
        if len(log_buffer) > 400:
            log_buffer[:] = log_buffer[-400:]
        yield _render_train_summary(state), "\n".join(log_buffer)
    proc.wait()
    exit_code = proc.returncode
    state["active_step"] = ""
    state["steps"][step_key]["exit_code"] = exit_code
    state["steps"][step_key]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if exit_code == 0:
        state["steps"][step_key]["done"] = True
    else:
        state["last_error"] = f"Step {TRAIN_STEP_LABELS.get(step_key, step_key)} gagal (exit code {exit_code})."
    _save_train_state(state)
    yield _render_train_summary(state), "\n".join(log_buffer)


def run_train_step_stream(step_key, force_rerun, config_path_value, model_name_value, selected_model_value):
    state = _load_train_state()
    selected_model, _ = _pick_selected_model(selected_model_value)
    state = _sync_state_with_dataset(state, selected_model=selected_model)
    if not selected_model:
        yield _render_train_summary(state), "Tidak ada folder model di dataset_raw. Isi dataset_raw dulu."
        return
    if state.get("active_step"):
        msg = f"Masih ada step berjalan: {TRAIN_STEP_LABELS.get(state['active_step'], state['active_step'])}."
        state["last_error"] = msg
        _save_train_state(state)
        yield _render_train_summary(state), msg
        return
    if not step_key or step_key not in TRAIN_STEP_ORDER:
        yield _render_train_summary(state), "Pilih step yang valid."
        return
    if step_key == "train" and not torch.cuda.is_available():
        yield _render_train_summary(state), "Training lokal butuh GPU CUDA (NVIDIA). Perangkat ini tidak mendukung CUDA. Lanjutkan training di Google Colab."
        return
    if state.get("dataset_file_count", 0) == 0:
        yield _render_train_summary(state), f"Folder dataset_raw/{selected_model} kosong. Isi data dulu sebelum training."
        return
    if state["steps"][step_key]["done"] and not force_rerun:
        yield _render_train_summary(state), f"Step '{TRAIN_STEP_LABELS[step_key]}' sudah selesai untuk dataset ini. Centang 'Paksa Ulang Step' jika ingin menjalankan lagi."
        return
    missing = [req for req in TRAIN_STEP_REQUIREMENTS[step_key] if not state["steps"][req]["done"]]
    if missing and not force_rerun:
        missing_label = ", ".join(TRAIN_STEP_LABELS[m] for m in missing)
        yield _render_train_summary(state), f"Step belum siap. Selesaikan dulu: {missing_label}."
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
    for status_text, logs_text in _run_command_stream(cmd, step_key, state, log_buffer):
        yield status_text, logs_text


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
        yield _render_train_summary(state), "Semua step sudah selesai untuk dataset ini."
        return
    target = next_step or "train"
    for out in run_train_step_stream(target, force_rerun, config_path_value, model_name_value, selected_model):
        yield out


def run_local_prepare_stream(force_rerun, sanitize_before, config_path_value, model_name_value, selected_model_value):
    selected_model, models = _pick_selected_model(selected_model_value)
    if not selected_model:
        yield refresh_train_status(""), "Tidak ada folder model di dataset_raw. Tambahkan dulu dataset_raw/<nama_model>/*.wav"
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
        yield refresh_train_status(selected_model), msg
        return
    if sanitize_before:
        for status_text, logs_text in sanitize_existing_dataset_stream(selected_model):
            yield status_text, logs_text
    for step in ["resample", "flist_config", "hubert_f0"]:
        for status_text, logs_text in run_train_step_stream(step, force_rerun, config_path_value, model_name_value, selected_model):
            yield status_text, logs_text
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
        yield _render_train_summary(state), f"[DONE] Model '{selected_model}' dicatat sebagai sudah disiapkan."


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

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue = gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New']
    ),
) as app:
    with gr.Tabs():
        with gr.TabItem("Inferensi"):
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
                        slicer_input = gr.File(label="Pilih Audio Panjang", file_types=["audio"], type="filepath")
                        slicer_speaker = gr.Textbox(label="Nama Speaker/Artis (Folder)", placeholder="Contoh: artis_a")
                    with gr.Row():
                        slicer_db = gr.Number(label="Ambang Batas Diam (dB)", value=-40)
                        slicer_min_len = gr.Number(label="Panjang Minimum (ms)", value=5000)
                    slicer_submit = gr.Button("Mulai Pemotongan Audio (Background)", variant="primary")
                    slicer_progress = gr.Slider(label="Progres Pemotongan (%)", minimum=0, maximum=100, value=0, step=1, interactive=False)
                    slicer_output = gr.Textbox(label="Status Pemotongan", lines=3)
                with gr.TabItem("Training Manager"):
                    gr.Markdown(value="""
                        <font size=2>Mode sederhana (disarankan): siapkan dataset lokal dulu, lalu sinkronkan ke Google Drive untuk training di Colab.</font>
                        """)
                    default_model, initial_models = _pick_selected_model("")
                    with gr.Row():
                        force_rerun = gr.Checkbox(label="Paksa Ulang Preprocess", value=False)
                        sanitize_before = gr.Checkbox(label="Rapikan Nama File Dulu", value=True)
                    target_dataset_model = gr.Dropdown(
                        label="Pilih Folder Model dari dataset_raw (diproses per-model)",
                        choices=initial_models,
                        value=default_model,
                    )
                    with gr.Row():
                        train_config_path = gr.Textbox(label="Path Config Training", value="configs/config.json")
                        train_model_name = gr.Textbox(label="Nama Folder Model", value="44k")
                    drive_sync_path = gr.Textbox(label="Path Drive Sync (Desktop, parent folder)", value=DEFAULT_DRIVE_SYNC_PATH)
                    with gr.Row():
                        refresh_train_btn = gr.Button("Refresh Status")
                        run_prepare_btn = gr.Button("1) Siapkan Dataset Lokal (Step 1-3)", variant="primary")
                        sync_drive_btn = gr.Button("2) Sync Artifacts ke Google Drive (No Duplicate)")
                    train_status = gr.Textbox(label="Status Pipeline", lines=10, value=refresh_train_status(default_model))
                    train_logs = gr.Textbox(label="Log Proses (tail)", lines=16)

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
        slicer_submit.click(slice_audio_stream, [slicer_input, slicer_speaker, slicer_db, slicer_min_len], [slicer_output, slicer_progress], queue=True)
        refresh_train_btn.click(
            refresh_train_status_with_models,
            [target_dataset_model],
            [train_status, target_dataset_model],
            queue=False,
        )
        target_dataset_model.change(
            refresh_train_status_with_models,
            [target_dataset_model],
            [train_status, target_dataset_model],
            queue=False,
        )
        run_prepare_btn.click(
            run_local_prepare_stream,
            [force_rerun, sanitize_before, train_config_path, train_model_name, target_dataset_model],
            [train_status, train_logs],
            queue=True,
        )
        sync_drive_btn.click(
            sync_artifacts_to_drive_stream,
            [drive_sync_path, target_dataset_model],
            [train_status, train_logs],
            queue=True,
        )
if __name__ == "__main__":
    app.queue().launch(server_name="127.0.0.1", show_api=False)
