from utils import normalize_angles

import numpy as np
from PIL import Image
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
bg_path = os.path.join(parent_dir, "assets", "bg.jpg")

bg_pil = Image.open(bg_path)
bg = np.array(bg_pil)

bg_height = len(bg)
bg_width = len(bg[0])

pi = np.pi

height_increment = bg_height/pi
width_increment = bg_width/(2*pi)

#our normalize angles will return theta value between 0 and pi and phi value between 0 and 2 pi
#merkezimiz theta için pi/2 phi için ise pi
def get_background_color(vector):
    angles = normalize_angles(vector)
    theta = angles[1]  # [0, pi]
    phi = angles[2]    # [0, 2pi]

    # Tam float koordinatları bul
    x = phi * width_increment   # Yatay (X) ekseni
    y = theta * height_increment # Dikey (Y) ekseni
    
    # Sol üst köşenin koordinatları (x0, y0)
    x0 = int(x) % bg_width
    y0 = int(y)
    
    # Sağ alt köşenin koordinatları (x1, y1)
    x1 = (x0 + 1) % bg_width  
    y1 = min(y0 + 1, bg_height - 1) 

    # Dikeyde taşmayı önlemek için y0'ı da sınırla
    y0 = min(y0, bg_height - 1)

    # x_weight: Sağdaki piksele ne kadar yakınız? (0.0 ile 1.0 arası)
    x_weight = x - int(x)
    # y_weight: Alttaki piksele ne kadar yakınız?
    y_weight = y - int(y)

    c00 = bg[y0, x0].astype(float) # Sol Üst
    c10 = bg[y0, x1].astype(float) # Sağ Üst
    c01 = bg[y1, x0].astype(float) # Sol Alt
    c11 = bg[y1, x1].astype(float) # Sağ Alt

    top_color = c00 * (1 - x_weight) + c10 * x_weight
    bottom_color = c01 * (1 - x_weight) + c11 * x_weight

    final_color = top_color * (1 - y_weight) + bottom_color * y_weight

    return final_color / 255.0


