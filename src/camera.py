from utils import cartesian_to_spherical, spherical_to_cartesian
from scene import Sphere

import numpy as np
import math

class SchwarzschildGeodesic:
    def __init__(self, rs):
        # rs: Schwarzschild yarıçapı (2GM/c^2)
        self.rs = rs # kara delik yarıçapı

    def get_derivatives(self, state):
        # PDF'teki ivme denklemleri bu kısımda 
        # Girdi: [t, r, theta, phi, dt_dlam, dr_dlam, dtheta_dlam, dphi_dlam]
        # Çıktı: [dt, dr, dtheta, dphi, d2t, d2r, d2theta, d2phi]
        t, r, theta, phi, ut, ur, utheta, uphi = state 
        rs = self.rs
        
        # Singularity kontrolü. Eğer ışık olay ufkuna çarparsa durur.
        if r <= rs:
            return np.zeros_like(state) # Olay ufkuna çarptı

        # Ortak terimler
        r_minus_rs = r - rs
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # ******************** İVME DENKLEMLERİ ********************
        
        # A) Zaman İvmesi (d2t)
        # PDF Formülü: -(rs / (r * (r - rs))) * dr * dt
        # Bu kısım karadeliğin zamanı nasıl büktüğünü hesaplar. PDF'teki d2t denklemi
        a_t = - (rs / (r * r_minus_rs)) * ur * ut

        # B) Radyal İvme (d2r)
        # Bu en karmaşık denklem
        term1 = (rs * r_minus_rs) / (2 * r**3) * (ut**2)
        term2 = rs / (2 * r * r_minus_rs) * (ur**2)
        # Term1 ve Term2 ışığın kütleçekim ile çekilmesini hesaplar
        term3 = r_minus_rs * (utheta**2)
        term4 = r_minus_rs * (sin_theta**2) * (uphi**2)
        # Term4 merkezkaç etkisidir 
        
        # PDF'teki formül: -(term1 - term2 - term3 - term4)
        a_r = - (term1 - term2 - term3 - term4)
        # Bu denklemde a_r negatif çıkarsa ışık içeri çekilir, pozitif çıkarsa dışarı itilir.

        # C) Polar Açı İvmesi (d2theta)
        # PDF Formülü: -( (2/r)*dr*dtheta - sin*cos*dphi^2 )
        # Parantez içindeki eksi işareti dağılınca + olur. Açısal Momentumun korunumu.
        a_theta = - ((2.0 / r) * ur * utheta - sin_theta * cos_theta * (uphi**2))

        # D) Azimutal Açı İvmesi (d2phi)
        # PDF Formülü: -( (2/r)*dr*dphi + 2*cot*dtheta*dphi )
        cot_theta = cos_theta / sin_theta
        a_phi = - ((2.0 / r) * ur * uphi + 2.0 * cot_theta * utheta * uphi)

        # Türevleri döndüren kısım, buradaki çıktı bize bir sonraki adımı söyler.
        return np.array([ut, ur, utheta, uphi, a_t, a_r, a_theta, a_phi])

    def rk4_step(self, state, h):
        # h: Adım büyüklüğü (step size)
        k1 = h * self.get_derivatives(state)
        k2 = h * self.get_derivatives(state + 0.5 * k1)
        k3 = h * self.get_derivatives(state + 0.5 * k2)
        k4 = h * self.get_derivatives(state + k3)
        
        new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        # Bu kısım kabaca durumlar içinden ağırlıklı ortalama alır ve yeni durumu hesaplar
        return new_state

#Center cartesian koordinatlarda olacak
#real position vector is in cartesian coordinates will hold real position
#relative position is in spherical coordinates will hold relative position to center
#direction vector is in spherical coordinates with r = 1
#Bu kamera Orbit camera olacak şekilde tasarlanacak
class Camera:
    
    def __init__(self, scene, width, height, fov, dλ, maxλ):
        self.scene = scene
        self.center = np.array(self.scene.black_hole.position, dtype=float)
        self.real_position = self.center + np.array([5.0, 5.0, 5.0], dtype=float)
        self.relative_position = cartesian_to_spherical(self.real_position-self.center)
        self.direction = self.find_camera_direction()
        self.width = width
        self.height = height
        self.aspect_ratio = self.width/self.height
        self.fov = fov
        self.dλ = dλ
        self.maxλ = maxλ

    def find_camera_direction(self):
        vector = np.array(self.center - self.real_position) #Kameradan merkeze olan vektörü bulduk
        length = np.linalg.norm(vector)

        if length != 0:
            cartesian_unit_vector = vector / length
        else:
            cartesian_unit_vector = vector

        spherical_unit_vector =  cartesian_to_spherical(cartesian_unit_vector)
        spherical_unit_vector[0] = 1.0

        return spherical_unit_vector

    def update_real_position(self):
        cartesian_relative_position = spherical_to_cartesian(self.relative_position)
        self.real_position = self.center + cartesian_relative_position
        self.direction = self.find_camera_direction()

    def find_ray_direction(self, height_index, width_index):
        fwd_spherical = self.direction 
        forward = spherical_to_cartesian(fwd_spherical)
        forward = forward / np.linalg.norm(forward) # Birim vektör yap

        global_up = np.array([0, 0, 1.0])
        
        if abs(np.dot(forward, global_up)) > 0.99:
             global_up = np.array([0, 1.0, 0])

        # Right (Sağ) Vektörü = Forward x GlobalUp
        right = np.cross(forward, global_up)
        right = right / np.linalg.norm(right)

        # Up (Kamera Yukarısı) Vektörü = Right x Forward
        cam_up = np.cross(right, forward)
        cam_up = cam_up / np.linalg.norm(cam_up)

        #EKRAN DÜZLEMİNDEKİ PİKSELİN YERİ
        scale = math.tan(math.radians(self.fov) / 2.0)

        # Pikselin normalize edilmiş koordinatları (-1 ile +1 arası)
        # x (genişlik) için:
        px = (2 * (width_index + 0.5) / self.width - 1) * scale * self.aspect_ratio
        # y (yükseklik) için (y ekseni genelde ters çalışır, o yüzden 1 - ... yaptık):
        py = (1 - 2 * (height_index + 0.5) / self.height) * scale

        #SONUÇ IŞIN YÖNÜ
        ray_direction = forward + (px * right) + (py * cam_up)
        
        #Birim vektör
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        # Doğrudan Kartezyen döndürüyoruz
        return ray_direction

    
    def ray_trace(self, height_index, width_index):
            λ = 0.0
            dλ = self.dλ
            maxλ = self.maxλ
            bh = self.scene.black_hole
            bg = self.scene.back_ground
            
            initial_t = 0.0
            initial_r = self.relative_position[0]
            initial_theta = self.relative_position[1]
            initial_phi = self.relative_position[2]

            initial_x, initial_y, initial_z = spherical_to_cartesian(self.relative_position)
            
            xy2 = initial_x**2 + initial_y**2
            
            dir_x, dir_y, dir_z = self.find_ray_direction(height_index, width_index) 

            initial_dr = (initial_x*dir_x + initial_y*dir_y + initial_z*dir_z)/initial_r

            if xy2 < 1e-9:
                initial_dtheta = 0.0
            else:
                initial_dtheta = (initial_z*(initial_x*dir_x + initial_y*dir_y) - xy2*dir_z) / (initial_r**2 * math.sqrt(xy2))

            if xy2 < 1e-9:
                initial_dphi = 0.0
            else:
                initial_dphi = (initial_x*dir_y - initial_y*dir_x) / xy2

            rs = bh.rs
            factor = 1.0 - (rs / initial_r)

            spatial_part = (initial_dr**2 / factor) + \
                           (initial_r**2 * initial_dtheta**2) + \
                           (initial_r**2 * (math.sin(initial_theta)**2) * initial_dphi**2)
        
            initial_dt = math.sqrt(spatial_part / factor)

            ray = np.array([
            initial_t, initial_r, initial_theta, initial_phi, 
            initial_dt, initial_dr, initial_dtheta, initial_dphi
            ])

            # YENİ EKLENEN KISIM: RK4 Solver Başlat
            solver = SchwarzschildGeodesic(rs)

            while λ <= maxλ:
                current_r = ray[1]

                #Çarpışma kontrolü
                current_pos_cartesian = spherical_to_cartesian([ray[1], ray[2], ray[3]])
                for obj in self.scene.objects:
                    if isinstance(obj, Sphere):
                    # Nokta ile Küre Merkezi arasındaki mesafe
                        dist = np.linalg.norm(current_pos_cartesian - obj.center)
                    if dist < obj.radius:
                        return obj.color 

                # Olay Ufku Kontrolü
                if current_r < rs * 1.01:
                    return np.array([0.0, 0.0, 0.0]) # SİYAH
            
                # Sonsuza Gidiş Kontrolü
                if current_r > 50.0:
                    return bg.get_background_color(ray[1:4]) # UZAY RENGİ

                # YENİ EKLENEN KISIM: Adım Hesaplama
                ray = solver.rk4_step(ray, dλ)
                λ += dλ

            return np.array([5.0, 2.0, 0.0]) #Döngü tamamlansa da hala çarpışmayanlar için hata kontrolü kırmızı renk
