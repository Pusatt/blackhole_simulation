import numpy as np
import math
from utils import spherical_to_cartesian, cartesian_to_spherical


class Camera:
    def __init__(self, scene, width, height, fov, dλ, maxλ):
        self.scene = scene

        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self.fov = fov

        self.dλ = dλ
        self.maxλ = maxλ

        # Kamera konumu
        self.center = np.array(scene.black_hole.position, dtype=float)
        self.real_position = self.center + np.array([5.0, 5.0, 5.0])
        self.relative_position = cartesian_to_spherical(
            self.real_position - self.center
        )

        self.direction = self.find_camera_direction()

    # --------------------------------------------------
    # Kamera yönü
    # --------------------------------------------------
    def find_camera_direction(self):
        v = self.center - self.real_position
        v = v / np.linalg.norm(v)
        sph = cartesian_to_spherical(v)
        sph[0] = 1.0
        return sph

    # --------------------------------------------------
    # Piksele göre ışın yönü (CPU)
    # --------------------------------------------------
    def find_ray_direction(self, h, w):
        forward = spherical_to_cartesian(self.direction)
        forward /= np.linalg.norm(forward)

        global_up = np.array([0, 0, 1.0])
        if abs(np.dot(forward, global_up)) > 0.99:
            global_up = np.array([0, 1.0, 0])

        right = np.cross(forward, global_up)
        right /= np.linalg.norm(right)

        up = np.cross(right, forward)

        scale = math.tan(math.radians(self.fov) / 2.0)

        px = (2 * (w + 0.5) / self.width - 1) * scale * self.aspect_ratio
        py = (1 - 2 * (h + 0.5) / self.height) * scale

        d = forward + px * right + py * up
        return d / np.linalg.norm(d)

    # --------------------------------------------------
    # GPU için başlangıç ray'i (TEK piksel)
    # --------------------------------------------------
    def _initial_ray_for_pixel(self, h, w):
        bh = self.scene.black_hole

        t0 = 0.0
        r0, th0, ph0 = self.relative_position

        x, y, z = spherical_to_cartesian(self.relative_position)
        d = self.find_ray_direction(h, w)

        dr = (x * d[0] + y * d[1] + z * d[2]) / r0

        xy2 = x * x + y * y
        if xy2 < 1e-9:
            dth = 0.0
            dph = 0.0
        else:
            dth = (
                z * (x * d[0] + y * d[1]) - xy2 * d[2]
            ) / (r0 * r0 * math.sqrt(xy2))
            dph = (x * d[1] - y * d[0]) / xy2

        rs = bh.rs
        factor = 1.0 - rs / r0

        spatial = (
            (dr * dr) / factor +
            r0 * r0 * dth * dth +
            r0 * r0 * (math.sin(th0) ** 2) * dph * dph
        )

        dt = math.sqrt(spatial / factor)

        return np.array(
            [t0, r0, th0, ph0, dt, dr, dth, dph],
            dtype=np.float32
        )

    # --------------------------------------------------
    # GPU: TÜM PİKSELLER İÇİN BAŞLANGIÇ RAY'LERİ
    # --------------------------------------------------
    def prepare_initial_rays(self):
        rays = np.zeros(
            (self.height, self.width, 8),
            dtype=np.float32
        )

        for h in range(self.height):
            for w in range(self.width):
                rays[h, w] = self._initial_ray_for_pixel(h, w)

        return rays
    
    def _update_relative_position(self):
        return cartesian_to_spherical(self.real_position - self.center)
