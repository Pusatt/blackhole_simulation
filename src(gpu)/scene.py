import numpy as np
import math
from shaders import Background
class Scene:
    def __init__(self):
        self.objects = []
        self.black_hole = None
        self.back_ground = Background(self)

    def add(self, obj):
        if isinstance(obj, BlackHole):
            self.black_hole = obj
        else:
            self.objects.append(obj)

class BlackHole:
    def __init__(self, scene, position, rs):
        scene.add(self)
        self.position = np.array(position, dtype=float)
        self.rs = rs

    def geodesic(self, ray, dλ):
        t, r, theta, phi, dt, dr, dtheta, dphi = ray
        rs = self.rs

        epsilon = 1e-5
        theta = min(max(theta, epsilon), math.pi - epsilon)

        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        cot_t = cos_t / sin_t

        ddt = -((rs / (r*(r-rs))) * dr * dt)
        ddr = -(((rs*(r-rs))/(2*r**3))*dt**2
                - ((rs/(2*r*(r-rs)))*dr**2)
                - ((r-rs)*dtheta**2)
                - ((r-rs)*(sin_t**2)*(dphi**2)))
        ddtheta = -(((2/r)*dr*dtheta) - (sin_t*cos_t*(dphi**2)))
        ddphi = -((2/r)*dr*dphi + 2*cot_t*dtheta*dphi)

        dt += ddt*dλ
        dr += ddr*dλ
        dtheta += ddtheta*dλ
        dphi += ddphi*dλ

        t += dt*dλ
        r += dr*dλ
        theta += dtheta*dλ
        phi += dphi*dλ

        return np.array([t, r, theta, phi, dt, dr, dtheta, dphi])

class Sphere:
    def __init__(self, scene, center, radius, color):
        scene.add(self)
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.color = np.array(color, dtype=float)

def extract_spheres(scene):
    centers, radii, colors = [], [], []
    for obj in scene.objects:
        if isinstance(obj, Sphere):
            centers.append(obj.center)
            radii.append(obj.radius)
            colors.append(obj.color)
    return (
        np.array(centers, dtype=np.float32),
        np.array(radii, dtype=np.float32),
        np.array(colors, dtype=np.float32)
    )
