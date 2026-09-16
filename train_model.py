import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

dataset_path = "dataset"
categories = ["with_mask", "without_mask"]

data = []
labels = []

print("Görüntüler yükleniyor ve yapay sinir ağı için matrislere dönüştürülüyor. Lütfen bekleyin...")

# Her bir klasörü (with_mask ve without_mask) geziyoruz
for category in categories:
    path = os.path.join(dataset_path, category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)

        # 1. OpenCV ile resmi oku
        image = cv2.imread(img_path)
        if image is None:  # Eğer bozuk bir resim dosyası varsa hata vermemesi için atla
            continue

        # 2. OpenCV resimleri varsayılan olarak BGR (Mavi-Yeşil-Kırmızı) okur.
        # Biz bunu standart olan RGB formatına çeviriyoruz.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. MobileNetV2'nin beklediği 224x224 boyutuna yeniden boyutlandır (Resize)
        image = cv2.resize(image, (224, 224))

        # 4. Resmi Keras'ın işleyebileceği bir sayısal diziye (array) çevir
        image = img_to_array(image)

        # 5. MobileNetV2'nin özel ölçeklendirmesini uygula (Piksel değerlerini -1 ile 1 arasına çeker)
        image = preprocess_input(image)

        # İşlenmiş resmi 'data', etiketini ise 'labels' listesine ekle
        data.append(image)
        labels.append(category)

# Listeleri yüksek hesaplama hızına sahip Numpy dizilerine (tensörlere) çeviriyoruz
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Bilgisayar "with_mask" yazısını anlamaz. Bunları 0 ve 1 gibi sayılara çeviriyoruz (One-Hot Encoding)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Verimizin %80'ini modeli eğitmek (train), %20'sini ise test etmek (test) için ayırıyoruz.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

print("İşlem tamamlandı!")
print(f"Eğitim (Train) için ayrılan resim sayısı: {trainX.shape[0]}")
print(f"Test (Validation) için ayrılan resim sayısı: {testX.shape[0]}")

print("MobileNetV2 temel modeli yükleniyor...")
# 1. Temel Modeli Yükle (include_top=False ile orijinal sınıflandırma kısmını atıyoruz)
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# 2. Kendi Sınıflandırma Başlığımızı (Head Model) Oluşturuyoruz
headModel = baseModel.output
headModel = MaxPooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel) # 2 sınıf: Maskeli ve Maskesiz

# 3. Temel model ile kendi başlığımızı birleştiriyoruz
model = Model(inputs=baseModel.input, outputs=headModel)

# 4. Temel modelin katmanlarını donduruyoruz (Eğitim sırasında ağırlıkları bozulmasın diye)
for layer in baseModel.layers:
    layer.trainable = False

# 5. Modeli Derleme (Compile)
print("Model derleniyor...")
opt = Adam(learning_rate=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("Model mimarisi başarıyla kuruldu ve eğitime hazır!")

# Hiperparametreler (Eğitim Ayarları)
EPOCHS = 20  # Veri setinin modelden kaç kez tam olarak geçeceği
BS = 32      # Modelin tek seferde işleyeceği resim sayısı (Batch Size)

print("Veri çoğaltma (Data Augmentation) işlemi ayarlanıyor...")
# Veri setimizdeki resimleri rastgele döndürerek, kaydırarak ve yakınlaştırarak
# yapay olarak çoğaltıyoruz. Bu, modelin farklı açılardaki yüzleri de tanımasını sağlar.
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

print(f"Yapay zeka eğitime başlıyor ({EPOCHS} Epoch sürecek)...")
print("NOT: Bu işlem bilgisayarınızın işlemci/ekran kartı hızına göre 5-15 dakika arası sürebilir.")

# Modeli Eğitme (Fit) İşlemi
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)

print("Eğitim tamamlandı! Model 'mask_detector.h5' olarak kaydediliyor...")
model.save("mask_detector.h5") # Gelecekte tekrar eğitmemek için modeli diske kaydediyoruz

print("Eğitim grafiği çiziliyor...")
# Başarı (Accuracy) ve Hata (Loss) grafiğini çizdirme
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="Eğitim Hatası (train_loss)")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="Test Hatası (val_loss)")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="Eğitim Doğruluğu (train_acc)")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="Test Doğruluğu (val_acc)")
plt.title("Eğitim Kaybı ve Doğruluğu")
plt.xlabel("Epoch Numarası")
plt.ylabel("Kayıp/Doğruluk (Loss/Accuracy)")
plt.legend(loc="lower left")
plt.savefig("plot.png") # Grafiği resim olarak kaydet

print("BÜTÜN İŞLEMLER BAŞARIYLA TAMAMLANDI!")