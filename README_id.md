[🇬🇧 English](README.md)
# Rangkuman
ICVLPR adalah *repository* penelitian yang berfokus pada *machine learning* dan *computer vision*, dibuat untuk memenuhi syarat kelulusan tugas akhir. *Repository* ini bertujuan menguji kemampuan model *state-of-the-art* [ANPR](https://en.wikipedia.org/wiki/Automatic_number-plate_recognition) dalam mengenali plat nomor kendaraan niaga di Indonesia.

# Dataset
Dataset terdiri dari sekitar 800 gambar plat nomor kendaraan niaga di Indonesia, yang dikumpulkan dari video YouTube dan pengambilan langsung di jalanan Semarang, Indonesia.

Dataset ini dibagi menjadi tiga bagian, yaitu `train`, `val`, dan `test`, dengan proporsi masing-masing sebesar 80%, 10%, dan 10%.

# Penggunaan
## Training
Untuk melakukan *training*, pastikan Anda sudah menginstal *package* yang diperlukan:
- PyTorch 2.4.0
- NumPy
- PIL (Python Imaging Library)
- tqdm
- wandb (opsional)

Untuk memulai *training*, jalankan perintah berikut di *shell* Anda:
```shell
python train.py
```
Anda juga bisa memodifikasi _hyperparameters_ selama proses pelatihan model. Untuk melihat opsi yang tersedia, gunakan _flag_ `--help`:
```shell
python train.py --help
```
