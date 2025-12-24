import cv2
import os
import sys

# Klasör yolu (Scriptin çalıştığı yerin tam yolunu alalım ki hata olmasın)
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(script_dir, 'render_frames_hq')
video_name = 'final_blackhole.mp4'

# 1. Klasör kontrolü
if not os.path.exists(image_folder):
    print(f"HATA: '{image_folder}' yolu bulunamadı! Klasör ismini veya yerini kontrol et.")
    sys.exit()

# 2. Resim listeleme
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort() 

# 3. Resim sayısı kontrolü (Senin hatanın sebebi burası)
if len(images) == 0:
    print(f"HATA: '{image_folder}' klasörünün içinde hiç .png dosyası bulunamadı!")
    print("Dosyalar orada mı? Uzantıları .png mi? Kontrol et.")
    sys.exit()

print(f"Toplam {len(images)} kare bulundu. Video işleniyor...")

# Video oluşturma kısmı
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 24, (width, height))

for i, image in enumerate(images):
    img_path = os.path.join(image_folder, image)
    video.write(cv2.imread(img_path))
    # İlerleme durumunu gösterelim ki dondu sanma
    if i % 50 == 0:
        print(f"İşleniyor: {i}/{len(images)}")

cv2.destroyAllWindows()
video.release()
print(f"Video başarıyla oluşturuldu: {video_name}")