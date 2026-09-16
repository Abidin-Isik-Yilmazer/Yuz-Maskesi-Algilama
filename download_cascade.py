import urllib.request

url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
print("Yüz tanıma dosyası indiriliyor...")
urllib.request.urlretrieve(url, "haarcascade_frontalface_default.xml")
print("Başarıyla indirildi! Proje klasöründe görebilirsin.")