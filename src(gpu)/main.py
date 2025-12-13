import numpy as np
from PIL import Image
from scene import Scene, BlackHole
from camera import Camera
from gpu_renderer import render_gpu
from realtime_sim import run_realtime


scene = Scene()
BlackHole(scene, [0,0,0], 2.0)

width, height = 480, 270 
cam = Camera(scene, width, height, 80, 0.01, 80)

print("GPU render başlıyor...")
image = render_gpu(scene, cam)

image = (image * 255).clip(0,255).astype(np.uint8)
Image.fromarray(image).show()

run_realtime(scene)