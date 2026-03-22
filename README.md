# RVC WebUI Fork

Fork ini sekarang difokuskan untuk **RVC-only workflow**.

Semua menu dan alur lama yang berhubungan dengan:
- SoVITS / SVC
- training
- preprocessing dataset
- audio slicer
- model mixing

sudah dihapus dari antarmuka utama `webUI.py`.

UI sekarang dipisah dengan **sidebar**, **group menu**, dan **sub-menu** untuk workflow berikut:
- `Ringkasan`
- `Model Manager`
- `Audio ke Voice`
- `Teks ke Voice`
- `Full Song`
- `Pengaturan`

## Fitur

- Load model RVC dari `.pth` atau `.zip`
- Dukungan `.index` opsional
- Konversi `audio -> voice`
- Konversi `text -> TTS -> RVC`
- Konversi `full song -> stem separation -> RVC -> remix`
- Layout baru berbasis sidebar

## Struktur UI

Sidebar dibagi menjadi beberapa grup:

1. `Inti`
   - `Ringkasan`
   - `Model Manager`
2. `Konversi`
   - `Audio ke Voice`
   - `Teks ke Voice`
   - `Full Song`
3. `Sistem`
   - `Pengaturan`

## Requirement

Lingkungan yang sudah diuji di workspace ini:

- Python `3.10`
- Windows
- virtual environment lokal di `.venv`

Dependency utama:

- `gradio==3.50.2`
- `inferrvc`
- `demucs`
- `torch`
- `torchaudio`
- `edge_tts`

Catatan:
- Untuk dependency lama seperti `fairseq` dan `omegaconf`, gunakan `pip<24.1`.
- `demucs` akan mengunduh model tambahan saat pertama kali dipakai.

## Instalasi

### Windows PowerShell

```powershell
cd E:\Project\sovits

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install "pip<24.1"
python -m pip install -r requirements_win.txt
```

Kalau ingin memastikan dependency utama terpasang:

```powershell
python -m pip install inferrvc demucs
```

## Menjalankan Aplikasi

```powershell
cd E:\Project\sovits
.\.venv\Scripts\Activate.ps1
python webUI.py
```

Lalu buka:

```text
http://127.0.0.1:7860
```

## Cara Pakai

### 1. Load Model

Masuk ke `Model Manager`:

- pilih model `.pth` atau `.zip`
- pilih file `.index` jika ada
- pilih device
- klik `Muat Model RVC`

### 2. Audio ke Voice

Masuk ke `Audio ke Voice`:

- upload audio
- atur `transpose`
- atur `index_rate`
- atur `protect`
- klik `Konversi Audio RVC`

Hasil audio akan disimpan ke folder:

```text
results/
```

### 3. Teks ke Voice

Masuk ke `Teks ke Voice`:

- masukkan teks
- pilih jenis suara TTS
- atur `transpose`, `index_rate`, `protect`
- klik `Konversi Teks ke Voice RVC`

### 4. Full Song

Masuk ke `Full Song`:

- pilih file lagu penuh
- pilih device untuk RVC
- pilih device untuk Demucs
- atur parameter konversi
- klik `Konversi Lagu Full dengan RVC`

Output yang dihasilkan:

- vocal hasil RVC
- instrumental hasil separation
- mix final

## Output Folder

Folder yang umum dipakai:

- `results/` untuk hasil audio
- `logs/rvc_models/` untuk ekstraksi model zip sementara

## Catatan Device

- Jika ada NVIDIA CUDA, inferensi biasanya lebih cepat
- Jika CPU-only, aplikasi tetap bisa jalan, tetapi proses akan lebih lambat
- Mode `Full Song` paling berat karena melibatkan `demucs` + `RVC`

## Catatan Repo

Repo ini masih mungkin menyimpan beberapa file lama dari basis proyek sebelumnya, tetapi:

- UI utama sekarang hanya memakai workflow RVC
- dokumentasi ini mengikuti kondisi UI saat ini
- file training lama tidak lagi menjadi alur utama aplikasi

## Troubleshooting

### `Dependency inferrvc belum tersedia`

Jalankan:

```powershell
python -m pip install inferrvc
```

### `Dependency demucs belum tersedia`

Jalankan:

```powershell
python -m pip install demucs
```

### `fairseq / omegaconf` gagal saat install

Turunkan pip dulu:

```powershell
python -m pip install "pip<24.1"
```

Lalu ulangi install dependency.

### App tidak bisa dibuka karena lock

Tutup instance lama `webUI.py`, lalu jalankan ulang.

## Disclaimer

Gunakan repo ini dengan tanggung jawab sendiri.

- Pastikan Anda memiliki hak untuk memakai audio sumber dan model yang digunakan
- Jangan gunakan untuk pelanggaran privasi, penipuan, atau penyalahgunaan identitas
- Pengguna bertanggung jawab penuh atas input, output, dan distribusi hasil
