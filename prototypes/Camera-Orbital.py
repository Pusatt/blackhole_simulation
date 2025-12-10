import numpy as np
import math, pygame

def spherical_to_cartesian(vector):
    vector = np.array(vector, dtype=float)
    
    r = vector[0]
    theta_rad = math.radians(vector[1])
    phi_rad = math.radians(vector[2])

    x = r * math.sin(theta_rad) * math.cos(phi_rad)
    y = r * math.sin(theta_rad) * math.sin(phi_rad)
    z = r * math.cos(theta_rad)
    
    return np.array([x, y, z], dtype=float)

def cartesian_to_spherical(vector):
    vector = np.array(vector, dtype=float)

    x = vector[0]
    y = vector[1]
    z = vector[2]

    r = math.sqrt(x**2 + y**2 + z**2)
    
    if r == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    
    theta_rad = math.atan2(math.sqrt(x**2 + y**2), z)
    phi_rad = math.atan2(y, x)

    theta_deg = math.degrees(theta_rad)
    phi_deg = math.degrees(phi_rad)

    return np.array([r, theta_deg, phi_deg], dtype=float)

    
#Center cartesian koordinatlarda olacak
#real position vector is in cartesian coordinates will hold real position
#relative position is in spherical coordinates will hold relative position to center
#direction vector is in spherical coordinates with r = 1
#Bu kamera Orbit camera olacak şekilde tasarlanacak
class Camera:
    
    def __init__(self, center, width, height, fov):
        self.center = np.array(center, dtype=float)
        self.real_position = self.center + np.array([10.0, 0.0, -5.0], dtype=float)
        self.relative_position = cartesian_to_spherical(self.real_position-self.center)
        self.direction = self.find_camera_direction()
        self.width = width
        self.height = height
        self.aspect_ratio = self.width/self.height
        self.fov = fov
        self.view = np.zeros((self.height, self.width, 3))

        # Mouse hassasiyeti ve hareket hızı
        self.mouse_sensitivity = 0.2
        self.zoom_speed = 0.5

    def find_camera_direction(self):
        vector = np.array(self.center - self.real_position)
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
        height_increment = self.fov/self.height
        width_increment = (self.fov*self.aspect_ratio)/self.width
        starting_theta = (self.direction[1]-self.fov/2)+(height_increment/2)
        starting_phi = (self.direction[2]-(self.fov*self.aspect_ratio/2))+(width_increment/2)
        new_theta = starting_theta + (height_index*height_increment)
        new_phi = starting_phi + (width_index*width_increment)
        return np.array([self.direction[0], new_theta, new_phi], dtype=float)

    def ray_trace(self):
        self.view.fill(0)

        light_dir = np.array([0.1, 0.1, 0.1])
        # Işık vektörünü normalize et (boyunu 1 yap)
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        for h in range(self.height):
            for w in range(self.width):
                spherical_ray_direction = self.find_ray_direction(h, w)
                cartesian_ray_direction = spherical_to_cartesian(spherical_ray_direction)
                t_latest = float('inf')
                pixel_color = [0,0,0]
                for obj in scene:
                    t = obj.check_intersect(self.real_position, cartesian_ray_direction) #ışın = başlangıç pozisyonu + t*ışının birim vektörü burda hangi t değeri için küre ile kesişir kontrol ediyoruz
                    if t>0 and t<t_latest:
                        hit_point = self.real_position + (cartesian_ray_direction * t)
                        normal = hit_point - obj.center
                        normal = normal / np.linalg.norm(normal)
                        intensity = np.dot(normal, light_dir)
                        intensity = max(0.1, intensity)
                        pixel_color = obj.color * intensity
                        t_latest = t
                        
                self.view[h][w] = pixel_color

    def handle_input(self):
        # Mouse hareketi (x farkı, y farkı)
        dx, dy = pygame.mouse.get_rel()
        
        if dx != 0 or dy != 0:
            # Mouse sağ/sol -> Phi açısını değiştirir (Yörüngede dönme)
            self.relative_position[2] += dx * self.mouse_sensitivity
            
            # Mouse yukarı/aşağı -> Theta açısını değiştirir (Yükseklik)
            self.relative_position[1] += dy * self.mouse_sensitivity

            # Theta'yı kısıtla (Kameranın takla atmasını önlemek için 1-179 derece arası)
            if self.relative_position[1] < 10.0:
                self.relative_position[1] = 10.0
            elif self.relative_position[1] > 170.0:
                self.relative_position[1] = 170.0

        # Klavye Girişleri (Zoom)
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_w]: # Yaklaş
            self.relative_position[0] -= self.zoom_speed
        if keys[pygame.K_s]: # Uzaklaş
            self.relative_position[0] += self.zoom_speed

        # Yarıçapın (r) negatif veya sıfır olmasını engelle
        if self.relative_position[0] < 2.0:
            self.relative_position[0] = 2.0

        # Değişiklikleri uygula
        self.update_real_position()
    
class Sphere:

    def __init__(self, center, radius, color):
        self.center = np.array(center) 
        self.radius = radius           
        self.color = np.array(color)
        scene.append(self)

    #küre denklemi = (xp-x)^2 + (yp-y)^2 + (zp-z)^2 = r^2
    #biz şimdi kameradan yolladığımız ışın bu denklemi çözer mi onu kontrol edeceğiz
    #kameradan yolladığımız ışının denklemi kameranın başlangıç pozisyonu + t*ray direction olacaktır
    def check_intersect(self, ray_origin, ray_direction):
        distance = ray_origin - self.center
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(distance, ray_direction)
        c = np.dot(distance, distance) - (self.radius ** 2)
        delta = (b * b) - (4 * a * c)

        if delta < 0:
            return -1.0  

        else:
            t = (-b - math.sqrt(delta)) / (2.0 * a)
            return t



scene = list()

width = 200
height = 100

screen_scale = 5
window_width = width*screen_scale
window_height = height*screen_scale
pygame.init()
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Ray Tracing Motoru")
pygame.mouse.set_visible(False) 
pygame.event.set_grab(True)
clock = pygame.time.Clock()

center = [3, 0, 0]

# Kamera FOV ayarı
cam = Camera(center, width, height, 90)

# Sahnede objeler
Sphere([0, 0, 0], 1.0, [255, 0, 0]) # Merkezde Kırmızı Küre
Sphere([3, 0, 0], 1.0, [0, 255, 0]) # Yanda Yeşil Küre

running = True

# Pygame mouse'un ilk hareketini "delta" olarak algılamasın diye başta bir kere oku
pygame.mouse.get_rel()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            
    cam.handle_input()
    cam.ray_trace()
    
    # Görüntüyü Pygame Surface'e çevir
    # self.view (H, W, 3) formatında, Pygame (W, H, 3) ister, o yüzden transpose
    pygame_screen = np.transpose(cam.view, (1, 0, 2))
    surface = pygame.surfarray.make_surface(pygame_screen)
    scaled_surface = pygame.transform.scale(surface, (window_width, window_height))
    
    window.blit(scaled_surface, (0, 0))
    pygame.display.flip()
    
    clock.tick(60) 
    pygame.display.set_caption(f"FPS: {clock.get_fps():.2f}")

pygame.quit()

