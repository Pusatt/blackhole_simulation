import numpy as np
import math, pygame

render_width = 100
render_height = 50

screen_scale = 10

window_width = render_width*screen_scale
window_height = render_height*screen_scale

aspect_ratio = render_width/render_height

screen = np.zeros((render_height, render_width, 3))
scene = list()

def spherical_to_cartesian(vector):
    vector = np.array(vector)
    
    r = vector[0]
    theta_rad = math.radians(vector[1])
    phi_rad = math.radians(vector[2])

    x = r * math.sin(theta_rad) * math.cos(phi_rad)
    y = r * math.sin(theta_rad) * math.sin(phi_rad)
    z = r * math.cos(theta_rad)
    
    return np.array([x, y, z])

#position vector is in cartesian coordinates
#direction vector is in spherical coordinates with r = 1    
class Camera:
    def __init__(self, position, direction, fov):
        self.position = np.array(position, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.fov = fov
        self.speed = 0.5
        self.mouse_sensitivity = 0.2

    def find_ray_direction(self, height_index, width_index):
        height_increment = self.fov/render_height
        width_increment = (self.fov*aspect_ratio)/render_width
        starting_theta = (self.direction[1]-self.fov/2)+(height_increment/2)
        starting_phi = (self.direction[2]-(self.fov*aspect_ratio/2))+(width_increment/2)
        new_theta = starting_theta + (height_index*height_increment)
        new_phi = starting_phi + (width_index*width_increment)
        return np.array([self.direction[0], new_theta, new_phi])

    def ray_trace(self):
        screen.fill(0)

        light_dir = np.array([0.1, 0.1, 0.1])
        # Işık vektörünü normalize et (boyunu 1 yap)
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        for h in range(render_height):
            for w in range(render_width):
                spherical_ray_direction = self.find_ray_direction(h, w)
                cartesian_ray_direction = spherical_to_cartesian(spherical_ray_direction)
                t_latest = float('inf')
                pixel_color = [0,0,0]
                for obj in scene:
                    t = obj.check_intersect(self.position, cartesian_ray_direction) #ışın = başlangıç pozisyonu + t*ışının birim vektörü burda hangi t değeri için küre ile kesişir kontrol ediyoruz
                    if t>0 and t<t_latest:
                        hit_point = self.position + (cartesian_ray_direction * t)
                        normal = hit_point - obj.center
                        normal = normal / np.linalg.norm(normal)
                        intensity = np.dot(normal, light_dir)
                        intensity = max(0.1, intensity)
                        pixel_color = obj.color * intensity
                        t_latest = t
                        
                screen[h][w] = pixel_color

                
    def handle_input(self):
        keys = pygame.key.get_pressed()
        
        dx, dy = pygame.mouse.get_rel()
        
        # Phi (Sağ/Sol) -> Mouse X ekseni (dx)
        self.direction[2] += dx * self.mouse_sensitivity
        
        # Theta (Yukarı/Aşağı) -> Mouse Y ekseni (dy)
        # Pygame'de yukarı gitmek eksi (-) değer verir.
        # Bizim koordinatlarda 0 derece tepe olduğu için, yukarı bakmak açıyı azaltmalı.
        self.direction[1] += dy * self.mouse_sensitivity
        
        # --- KORUMA (CLAMPING) ---
        # Theta asla tam 0 veya 180 olmamalı (Görüntü bozulur)
        if self.direction[1] < 1.0:
            self.direction[1] = 1.0
        elif self.direction[1] > 179.0:
            self.direction[1] = 179.0
            
        # 2. HAREKET (WASD)
        # Hareket için baktığımız yönün (r, theta, phi) XYZ karşılığını bulmalıyız
        # Sadece yatayda (Phi) hareket etmek istiyoruz, Theta'yı 90 (Ufuk) kabul edelim.
        
        # İleri Vektörü (Forward)
        forward = spherical_to_cartesian([1, 90, self.direction[2]])
        
        # Sağ Vektörü (Right) - Phi açısını 90 derece döndürürsek sağı buluruz
        right = spherical_to_cartesian([1, 90, self.direction[2] + 90])

        if keys[pygame.K_w]: # İleri
            self.position += forward * self.speed
        if keys[pygame.K_s]: # Geri
            self.position -= forward * self.speed
        if keys[pygame.K_d]: # Sağa
            self.position += right * self.speed
        if keys[pygame.K_a]: # Sola
            self.position -= right * self.speed
            
        if keys[pygame.K_SPACE]: # Yüksel (Z ekseni)
            self.position[2] += self.speed
        if keys[pygame.K_LSHIFT]: # Alçal
            self.position[2] -= self.speed
            
        if keys[pygame.K_ESCAPE]:
            pygame.event.post(pygame.event.Event(pygame.QUIT))

 
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



pygame.init()
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Ray Tracing Motoru")
pygame.mouse.set_visible(False) 
pygame.event.set_grab(True)
clock = pygame.time.Clock()

cam = Camera([0, 0, 0], [1, 90, 0], 90)
Sphere([5, 0, 0], 1.5, [255, 0, 0])
Sphere([5, 3, 0], 1.0, [0, 255, 0])

running = True

while running:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    
    cam.handle_input()
    cam.ray_trace()
    pygame_screen = np.transpose(screen, (1, 0, 2))
    surface = pygame.surfarray.make_surface(pygame_screen)
    scaled_surface = pygame.transform.scale(surface, (window_width, window_height))
    window.blit(scaled_surface, (0, 0))
    pygame.display.flip()
    clock.tick(60) 
    pygame.display.set_caption(f"FPS: {clock.get_fps():.2f}")

pygame.quit()








        
