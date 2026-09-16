import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Yüz tespiti için OpenCV'nin hazır Haar Cascade modelini yüklüyoruz
# (Bu, resimdeki yüzün nerede olduğunu bulmamızı sağlayan klasik bir algoritmadır)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 2. Kendi eğittiğimiz yapay zeka modelini yüklüyoruz
print("Eğitilmiş model yükleniyor...")
model = load_model("mask_detector.h5")

# 3. Kamerayı başlat (0 genelde bilgisayarın kendi kamerasıdır)
print("Kamera başlatılıyor...")
cap = cv2.VideoCapture(0)

while True:
    # Kameradan anlık kare (frame) oku
    ret, frame = cap.read()
    if not ret:
        break

    # Haar Cascade yüz tespiti gri tonlamalı resimlerde çok daha hızlı çalışır
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Karedeki yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # Tespit edilen her bir yüz için döngüye gir
    for (x, y, w, h) in faces:
        # Yüzün olduğu bölgeyi (Region of Interest - ROI) kareden kesip alıyoruz
        face_roi = frame[y:y + h, x:x + w]

        # Kestiğimiz yüzü modelimizin eğitimde alıştığı 224x224 boyutuna ve RGB formatına getiriyoruz
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = img_to_array(face_roi)
        face_roi = preprocess_input(face_roi)

        # Keras modelleri veriyi yığın (batch) halinde beklediği için boyutunu genişletiyoruz
        face_roi = np.expand_dims(face_roi, axis=0)

        # EĞİTTİĞİMİZ MODELE TAHMİN YAPTIRIYORUZ
        (mask, withoutMask) = model.predict(face_roi, verbose=0)[0]

        # Hangi olasılık daha yüksekse etiketimizi ve rengimizi (Yeşil/Kırmızı) ona göre seçiyoruz
        label = "Maskeli" if mask > withoutMask else "Maskesiz"
        color = (0, 255, 0) if label == "Maskeli" else (0, 0, 255)

        # Ekrana yazdırılacak metni, tahmin oranıyla birlikte hazırlıyoruz
        label_text = "{}: {:.1f}%".format(label, max(mask, withoutMask) * 100)

        # Orijinal karenin üzerine renkli dikdörtgeni ve tahmin yazısını çiziyoruz
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # İşlenmiş, dikdörtgen çizilmiş kareyi ekranda göster
    cv2.imshow("Yuz Maskesi Algilama (Cikmak icin klavyeden 'q' tusuna basin)", frame)

    # Klavyeden 'q' tuşuna basılırsa sonsuz döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# İşimiz bittiğinde kamerayı serbest bırak ve tüm pencereleri kapat
cap.release()
cv2.destroyAllWindows()