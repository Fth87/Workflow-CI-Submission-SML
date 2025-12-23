# Cancer Classification Project - MLflow CI/CD

## Dataset Information

### Breast Cancer Wisconsin (Diagnostic) Dataset

**Sumber Dataset:**

- **Nama**: Breast Cancer Wisconsin (Diagnostic) Data Set
- **Sumber**: UCI Machine Learning Repository & Scikit-learn Datasets
- **Link**: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
- **Scikit-learn**: `sklearn.datasets.load_breast_cancer()`

### Deskripsi Dataset

Dataset Breast Cancer Wisconsin adalah dataset klasifikasi yang berisi informasi tentang karakteristik sel tumor payudara. Dataset ini digunakan untuk memprediksi apakah tumor bersifat **ganas (malignant)** atau **jinak (benign)**.

### Karakteristik Dataset

- **Jumlah Sampel**: 569 sampel
- **Jumlah Fitur**: 30 fitur numerik
- **Target Variable**:
  - 0 = Malignant (Ganas)
  - 1 = Benign (Jinak)
- **Distribusi Kelas**:
  - Malignant: 212 sampel (37.3%)
  - Benign: 357 sampel (62.7%)

### Fitur Dataset

Setiap fitur dihitung dari gambar digital aspirasi jarum halus (FNA) dari massa payudara. Fitur-fitur ini menggambarkan karakteristik inti sel yang terdapat dalam gambar. Untuk setiap inti sel, 10 karakteristik berikut dihitung:

1. **radius**: Jarak rata-rata dari pusat ke titik-titik di perimeter
2. **texture**: Standar deviasi nilai skala abu-abu
3. **perimeter**: Keliling inti sel
4. **area**: Area inti sel
5. **smoothness**: Variasi lokal dalam panjang radius
6. **compactness**: (perimeter² / area) - 1.0
7. **concavity**: Tingkat keparahan bagian cekung dari kontur
8. **concave points**: Jumlah bagian cekung dari kontur
9. **symmetry**: Simetri inti sel
10. **fractal dimension**: "Perkiraan garis pantai" - 1

Untuk setiap karakteristik di atas, dihitung 3 nilai statistik:

- **Mean** (rata-rata)
- **Standard Error** (kesalahan standar)
- **Worst** (nilai terburuk/terbesar)

Sehingga total fitur menjadi: 10 karakteristik × 3 nilai = **30 fitur**

### Contoh Nama Fitur

- mean radius, mean texture, mean perimeter, mean area, ...
- radius error, texture error, perimeter error, area error, ...
- worst radius, worst texture, worst perimeter, worst area, ...

### Tujuan Proyek

Proyek ini bertujuan untuk:

1. Melakukan eksperimen machine learning menggunakan dataset Breast Cancer Wisconsin
2. Melatih model klasifikasi menggunakan Random Forest Classifier
3. Mengimplementasikan CI/CD workflow dengan GitHub Actions dan MLflow
4. Melakukan tracking eksperimen, parameter, dan metrik model menggunakan MLflow
5. Memonitor performa model secara otomatis

### Model yang Digunakan

- **Algorithm**: Random Forest Classifier
- **Parameters**:
  - n_estimators: 50
  - max_depth: 5
  - random_state: 42

### MLflow Tracking

Proyek ini menggunakan MLflow untuk:

- Tracking eksperimen dan parameter model
- Logging metrik performa model
- Versioning model
- Model registry

### Workflow CI/CD

Workflow GitHub Actions akan otomatis menjalankan training model setiap kali ada push ke repository, menggunakan perintah `mlflow run` untuk memastikan eksperimen tercatat dengan rapi.

---

**Author**: Muhammad Fatih Al Fawwaz  
**Project**: Final Submission - Machine Learning System dan Monitoring Learning
