import os
import urllib.request
import zipfile
import shutil

# GitHub'daki veri setinin direkt indirme linki
url = "https://github.com/prajnasb/observations/archive/refs/heads/master.zip"
zip_path = "dataset.zip"

print("1. Veri seti GitHub'dan indiriliyor, bu internet hızına bağlı olarak biraz sürebilir. Lütfen bekleyin...")
urllib.request.urlretrieve(url, zip_path)
print("İndirme tamamlandı!")

print("2. Zip dosyası dışarı çıkartılıyor...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("temp_data")

# Klasör yapımızı oluşturuyoruz
base_dir = "dataset"
with_mask_dir = os.path.join(base_dir, "with_mask")
without_mask_dir = os.path.join(base_dir, "without_mask")

os.makedirs(with_mask_dir, exist_ok=True)
os.makedirs(without_mask_dir, exist_ok=True)

print("3. Fotoğraflar 'dataset' klasörü altındaki ilgili yerlerine taşınıyor...")
# İndirilen verinin içindeki orijinal klasör yolları
source_with_mask = "temp_data/observations-master/experiements/data/with_mask"
source_without_mask = "temp_data/observations-master/experiements/data/without_mask"

# with_mask resimlerini taşı
for file_name in os.listdir(source_with_mask):
    shutil.move(os.path.join(source_with_mask, file_name), os.path.join(with_mask_dir, file_name))

# without_mask resimlerini taşı
for file_name in os.listdir(source_without_mask):
    shutil.move(os.path.join(source_without_mask, file_name), os.path.join(without_mask_dir, file_name))

print("4. Temizlik yapılıyor (Gereksiz zip ve temp dosyaları siliniyor)...")
os.remove(zip_path)
shutil.rmtree("temp_data")

print("Veri seti başarıyla oluşturuldu ve resimler klasörlere yerleştirildi.")