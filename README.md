# 😷 Yüz Maskesi Algılama 

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-green.svg)](https://opencv.org/)

Bu proje, derin öğrenme ve bilgisayarlı görü teknikleri kullanılarak gerçek zamanlı yüz maskesi algılama sistemi geliştirmek amacıyla hazırlanmıştır. Önceden eğitilmiş **MobileNetV2** mimarisi kullanılarak transfer öğrenme (transfer learning) uygulanmış ve model Keras ile ince ayardan (fine-tuning) geçirilmiştir.

## 🚀 Özellikler

* **Transfer Öğrenme:** Güçlü ve hafif MobileNetV2 modeli kullanılarak yüksek doğruluk oranı.
* **Gerçek Zamanlı Tespit:** OpenCV ve Haar Cascade kullanılarak web kamerasından anlık yüz tespiti ve maske sınıflandırması.
* **Veri Çoğaltma (Data Augmentation):** Modelin farklı açı ve aydınlatmalarda daha iyi çalışması için veri zenginleştirme.
* **Hafif ve Hızlı:** Sadece CPU üzerinde bile gerçek zamanlı çalışabilecek şekilde optimize edilmiştir.

## 🛠️ Kullanılan Teknolojiler

* **Python 3**
* **TensorFlow / Keras** (Model mimarisi ve eğitim)
* **OpenCV** (Görüntü işleme ve kamera entegrasyonu)
* **Scikit-Learn** (Veri seti ayırma ve etiketleme)
* **NumPy & Matplotlib** (Matris işlemleri ve görselleştirme)

## 📁 Klasör Yapısı
```text
Yuz-Maskesi-Algilama/
│
├── dataset/                            # Maskeli ve maskesiz eğitim verileri 
├── detect_mask_webcam.py               # Kameradan gerçek zamanlı test kodu
├── download_dataset.py                 # Veri setini otomatik indiren betik
├── train_model.py                      # Keras MobileNetV2 model eğitim kodu
├── haarcascade_frontalface_default.xml # OpenCV yüz tanıma modeli
├── mask_detector.h5                    # Eğitilmiş model dosyası
├── plot.png                            # Eğitim başarı/kayıp grafiği
└── README.md                           # Proje açıklaması
```

## ⚙️ Kurulum ve Çalıştırma

**1. Projeyi Klonlayın:**
```bash
git clone https://github.com/Abidin-Isik-Yilmazer/Yuz-Maskesi-Algilama.git
```

**2. Proje Klasörüne Girin:**
```bash
cd Yuz-Maskesi-Algilama
```

**3. Gerekli Kütüphaneleri Yükleyin:**
```bash
pip install tensorflow opencv-contrib-python scikit-learn matplotlib numpy
```

**4. (Opsiyonel) Modeli Yeniden Eğitmek İsterseniz:**
```bash
python download_dataset.py
python train_model.py
```

**5. Uygulamayı Başlatın (Kamera Testi):**
```bash
python detect_mask_webcam.py
```


