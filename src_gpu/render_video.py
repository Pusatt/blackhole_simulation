import cv2
import numpy as np
import time
import math
import os
import gc
from PIL import Image
import sys

# CPU Renderer (Fallback)
render_numba = None
try:
    from numba_renderer import render_numba
except ImportError:
    pass

# --- VİDEO RENDER AYARLARI ---
WIDTH = 1280   
HEIGHT = 720  
FPS = 24
DURATION_SEC = 10 
TOTAL_FRAMES = FPS * DURATION_SEC

# Çıktı Klasörü
OUTPUT_DIR = "render_frames_video"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# CUDA Kontrol
HAS_CUDA = False
try:
    from numba import cuda
    if cuda.is_available():
        from cuda_renderer import render_cuda
        HAS_CUDA = True
        print(f"CUDA Başlatıldı! Hedef: {TOTAL_FRAMES} kare.")
    else:
        print("CUDA Bulunamadı! CPU moduna geçiliyor...")
except ImportError:
    print("Numba CUDA modülü yok. CPU moduna geçiliyor...")

def load_background():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    bg_path = os.path.join(parent_dir, "assets", "bg2.jpg")
    try:
        bg_pil = Image.open(bg_path)
        bg = np.array(bg_pil)
        if HAS_CUDA:
            return np.ascontiguousarray(bg)
        return bg
    except:
        return np.zeros((100, 200, 3), dtype=np.uint8)

def generate_noise_texture(size=1024):
    noise = np.zeros((size, size), dtype=np.float32)
    scales = [4, 8, 16, 32]
    weights = [0.5, 0.25, 0.125, 0.05]
    for s, w in zip(scales, weights):
        r_noise = np.random.rand(s, s).astype(np.float32)
        r_upscaled = cv2.resize(r_noise, (size, size), interpolation=cv2.INTER_CUBIC)
        noise += r_upscaled * w
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return np.ascontiguousarray(noise)

def get_camera_position(frame_idx, total_frames):
    # Kamera karadeliğin etrafında döner
    angle = (frame_idx / total_frames) * 2 * math.pi
    radius = 25.0
    
    x = radius * math.sin(angle)
    y = -radius * math.cos(angle)
    z = 2.5 + 1.5 * math.sin(angle * 2) # Hafif yukarı aşağı salınım
    return np.array([x, y, z], dtype=np.float64)

def main():
    # --- RENDER AYARLARI ---
    FOV = 80.0
    DLAM = 0.02   
    MAXLAM = 100.0 
    
    RS = 2.0
    DISK_INNER = 2.6
    DISK_OUTER = 12.0
    
    # CPU fallback için
    disk_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    
    # Varlıkları Yükle
    bg_data = load_background()
    bg_device = None
    noise_device = None
    
    if HAS_CUDA:
        print("Varlıklar GPU belleğine yükleniyor...")
        noise_tex = generate_noise_texture(1024)
        bg_device = cuda.to_device(bg_data)
        noise_device = cuda.to_device(noise_tex)

    print(f"Video Render Başlıyor... Klasör: {OUTPUT_DIR}")

    for i in range(TOTAL_FRAMES):
        filename = os.path.join(OUTPUT_DIR, f"frame_{i:04d}.png")
        
        # Kaldığı yerden devam etme özelliği
        if os.path.exists(filename):
            print(f"Frame {i+1} zaten var, atlanıyor.")
            continue
        
        frame_start = time.time()
        
        # Kamera Hesabı
        cam_pos = get_camera_position(i, TOTAL_FRAMES)
        fwd = target - cam_pos; fwd /= np.linalg.norm(fwd)
        
        global_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fwd, global_up)) > 0.99: global_up = np.array([0.0, 1.0, 0.0])
        
        right = np.cross(fwd, global_up); right /= np.linalg.norm(right)
        up = np.cross(right, fwd); up /= np.linalg.norm(up)
        
        img_result = None
        
        if HAS_CUDA:
            d_cam = cuda.to_device(cam_pos)
            d_fwd = cuda.to_device(fwd)
            d_right = cuda.to_device(right)
            d_up = cuda.to_device(up)

            img_result = render_cuda(
                WIDTH, HEIGHT, FOV,
                d_cam, d_fwd, d_right, d_up,
                RS, bg_device, noise_device,
                DISK_INNER, DISK_OUTER, DLAM, MAXLAM
            )
            cuda.synchronize()
        else:
            # CPU Fallback
            if render_numba is None:
                print("HATA: CPU render modülü (numba_renderer) yüklenemedi.")
                break
                
            img_result = render_numba(
                WIDTH, HEIGHT, FOV,
                cam_pos, fwd, right, up,
                RS, bg_data, 
                DISK_INNER, DISK_OUTER, disk_color, 
                DLAM, MAXLAM
            )

        # Kaydetme
        if img_result is not None:
            img_uint8 = (img_result * 255.0).astype(np.uint8)
            blur = cv2.GaussianBlur(img_uint8, (0, 0), sigmaX=15, sigmaY=15) # Hafif blur
            img_bloom = cv2.addWeighted(img_uint8, 0.8, blur, 0.5, 0)
            img_bgr = cv2.cvtColor(img_bloom, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(filename, img_bgr)
            
            del img_result, img_uint8, img_bloom, img_bgr
            if HAS_CUDA: del d_cam, d_fwd, d_right, d_up
            gc.collect()
            
            sure = time.time() - frame_start
            kalan_dk = (TOTAL_FRAMES - i - 1) * sure / 60
            print(f"Frame {i+1}/{TOTAL_FRAMES} bitti. Süre: {sure:.2f}s. Tahmini Kalan: {kalan_dk:.1f} dk")

    print("\n--- RENDER TAMAMLANDI ---")

if __name__ == "__main__":
    main()
