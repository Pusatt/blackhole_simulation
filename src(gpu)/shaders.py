from utils import normalize_angles

import numpy as np
from PIL import Image
import os

pi = np.pi

class Background:

    def __init__(self, scene, index=1):
        scene.add(self)
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        parent_dir = os.path.dirname(current_dir)
        bg_path = os.path.join(parent_dir, "assets", "bg" + str(index) + ".jpg")

        bg_pil = Image.open(bg_path)
        self.bg = np.array(bg_pil)

        self.bg_height = len(self.bg)
        self.bg_width = len(self.bg[0])

        self.height_increment = self.bg_height/pi
        self.width_increment = self.bg_width/(2*pi)

    #our normalize angles will return theta value between 0 and pi and phi value between 0 and 2 pi
    #merkezimiz theta için pi/2 phi için ise pi
    def get_background_color(self, vector):
        angles = normalize_angles(vector)
        theta = angles[1]  # [0, pi]
        phi = angles[2]    # [0, 2pi]

        # Tam float koordinatları bul
        x = phi * self.width_increment   # Yatay (X) ekseni
        y = theta * self.height_increment # Dikey (Y) ekseni
        
        # Sol üst köşenin koordinatları (x0, y0)
        x0 = int(x) % self.bg_width
        y0 = int(y)
        
        # Sağ alt köşenin koordinatları (x1, y1)
        x1 = (x0 + 1) % self.bg_width  
        y1 = min(y0 + 1, self.bg_height - 1) 

        # Dikeyde taşmayı önlemek için y0'ı da sınırla
        y0 = min(y0, self.bg_height - 1)

        # x_weight: Sağdaki piksele ne kadar yakınız? (0.0 ile 1.0 arası)
        x_weight = x - int(x)
        # y_weight: Alttaki piksele ne kadar yakınız?
        y_weight = y - int(y)

        c00 = self.bg[y0, x0].astype(float) # Sol Üst
        c10 = self.bg[y0, x1].astype(float) # Sağ Üst
        c01 = self.bg[y1, x0].astype(float) # Sol Alt
        c11 = self.bg[y1, x1].astype(float) # Sağ Alt

        top_color = c00 * (1 - x_weight) + c10 * x_weight
        bottom_color = c01 * (1 - x_weight) + c11 * x_weight

        final_color = top_color * (1 - y_weight) + bottom_color * y_weight

        return final_color / 255.0


