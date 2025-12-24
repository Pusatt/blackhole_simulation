import cv2
import numpy as np
import time
import math
import os
import gc
from PIL import Image

# CPU Renderer (Fallback)
try:
    from numba_renderer import render_numba
except ImportError:
    pass

# --- AYARLAR ---
WIDTH = 1280 
HEIGHT = 720

# CUDA Kontrol
HAS_CUDA = False
try:
    from numba import cuda
    if cuda.is_available():
        from cuda_renderer import render_cuda
        HAS_CUDA = True
        print(f"CUDA Başlatıldı! Tek kare render alınacak.")
    else:
        print("CUDA Bulunamadı! CPU kullanılıyor (Yavaş olabilir).")
except ImportError:
    print("Numba CUDA modülü yok.")

def load_background():
    # Arka planı yükle
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    # Background yolunu assets klasörüne göre ayarladık
    bg_path = os.path.join(parent_dir, "assets", "bg2.jpg")
    try:
        bg_pil = Image.open(bg_path)
        bg = np.array(bg_pil)
        if HAS_CUDA:
            return np.ascontiguousarray(bg)
        return bg
    except:
        # Bulamazsa siyah ekran
        return np.zeros((100, 200, 3), dtype=np.uint8)

def generate_noise_texture(size=1024):
    # Bulutsu efekti için gürültü üret
    noise = np.zeros((size, size), dtype=np.float32)
    scales = [4, 8, 16, 32]
    weights = [0.5, 0.25, 0.125, 0.05]
    for s, w in zip(scales, weights):
        r_noise = np.random.rand(s, s).astype(np.float32)
        r_upscaled = cv2.resize(r_noise, (size, size), interpolation=cv2.INTER_CUBIC)
        noise += r_upscaled * w
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return np.ascontiguousarray(noise)


def main():
    # --- RENDER AYARLARI ---
    FOV = 80.0
    
    # KALİTE AYARI (DLAM):
    DLAM = 0.02  
    MAXLAM = 100.0 
    
    RS = 2.0
    DISK_INNER = 2.6
    DISK_OUTER = 12.0
    
    target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    
    # Varlıkları GPU'ya Yükle
    bg_data = load_background()
    bg_device = None
    noise_device = None
    
    if HAS_CUDA:
        print("Varlıklar GPU belleğine yükleniyor...")
        noise_tex = generate_noise_texture(1024)
        bg_device = cuda.to_device(bg_data)
        noise_device = cuda.to_device(noise_tex)

    print("Render Başlıyor... ")
    
    filename = "render_output.png"
    
    frame_start = time.time()
    
    # --- KAMERA POZİSYONU (MANUEL AYAR) ---
    # Buradaki 'angle' değerini değiştirerek karadeliğe farklı açılardan bakabilirsin.
    angle = 0.5 # 0.0 tam karşı, değer arttıkça etrafında döner
    radius = 25.0
    
    cam_x = radius * math.sin(angle)
    cam_y = -radius * math.cos(angle)
    cam_z = 2.5 + 1.5 * math.sin(angle * 2) # Hafif yukarıdan bakış
    
    cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float64)
    
    # Vektör Hesaplamaları
    fwd = target - cam_pos; fwd /= np.linalg.norm(fwd)
    
    global_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(fwd, global_up)) > 0.99: global_up = np.array([0.0, 1.0, 0.0])
    
    right = np.cross(fwd, global_up); right /= np.linalg.norm(right)
    up = np.cross(right, fwd); up /= np.linalg.norm(up)
    
    img_result = None
    
    print("GPU Hesaplaması yapılıyor...")
    
    if HAS_CUDA:
        d_cam = cuda.to_device(cam_pos)
        d_fwd = cuda.to_device(fwd)
        d_right = cuda.to_device(right)
        d_up = cuda.to_device(up)

        # Render Çağrısı
        img_result = render_cuda(
            WIDTH, HEIGHT, FOV,
            d_cam, d_fwd, d_right, d_up,
            RS, bg_device, noise_device,
            DISK_INNER, DISK_OUTER, DLAM, MAXLAM
        )
        
        cuda.synchronize()
    else:
        # CPU Fallback (Eğer GPU yoksa)
        pass 

    # Görüntü İşleme (Bloom Efekti)
    print("Görüntü işleniyor ve kaydediliyor...")
    img_uint8 = (img_result * 255.0).astype(np.uint8)
    blur = cv2.GaussianBlur(img_uint8, (0, 0), sigmaX=15, sigmaY=15)
    img_bloom = cv2.addWeighted(img_uint8, 0.8, blur, 0.5, 0)
    img_bgr = cv2.cvtColor(img_bloom, cv2.COLOR_RGB2BGR)
    
    # Kaydet
    cv2.imwrite(filename, img_bgr)
    
    # Bellek Temizliği
    del img_result, img_uint8, img_bloom, img_bgr
    if HAS_CUDA: del d_cam, d_fwd, d_right, d_up
    gc.collect()
    
    sure = time.time() - frame_start
    print(f"Render tamamlandı! Süre: {sure:.2f}s")
    print(f"Dosya konumu: {filename}")

if __name__ == "__main__":
    main()
