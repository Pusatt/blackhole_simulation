from scene import Scene, BlackHole, Sphere
from shaders import Background
from camera import Camera

from PIL import Image
import numpy as np


bg_index = 1

bh_position = [0,0,0]
bh_rs = 2.0

width = 1920
height = 1080
fov = 80
step_size = 0.001
max_steps = 100


space = Scene()

bg = Background(space, bg_index)
bh = BlackHole(space, bh_position, bh_rs)

#Sphere(space, [5.0, 2.0, 0.0], 1.0, [1.0, 0.0, 0.0]) # Kırmızı
#Sphere(space, [-4.0, -2.0, 3.0], 1.5, [0.0, 1.0, 0.0]) # Yeşil
#Sphere(space, [6.0, 1.0, 1.0], 1.5, [0.8, 0.4, 0.2]) #Kızıl
#Sphere(space, [-4.0, -3.0, 0.0], 1.0, [0.2, 0.6, 0.9]) #Parlak mavi

cam = Camera(space, width, height, fov, step_size, max_steps)

image = np.zeros((height, width, 3))

print(f"Render başlıyor: {width}x{height} piksel...")

# 3. Render Döngüsü
for h in range(height):
    if h % 10 == 0: print(f"Satır {h}/{height} işleniyor...") # İlerleme çubuğu gibi
    for w in range(width):
        pixel_color = cam.ray_trace(h, w)
        image[h, w] = pixel_color

if image.dtype != np.uint8:
    image_to_show = (image * 255).astype(np.uint8)
else:
    image_to_show = image

img = Image.fromarray(image_to_show)
img.show()
#scale_factor = 8 
#new_size = (img.width * scale_factor, img.height * scale_factor)
#img_large = img.resize(new_size, resample=Image.NEAREST)
#img_large.show()


