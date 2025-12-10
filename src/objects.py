import numpy as np
import math

class Scene:

    def __init__(self):
        self.objects = list()
        self.black_hole = None

    def add(self, obj):
        if isinstance(obj, BlackHole):
            self.black_hole = obj
            
        else:
            self.objects.append(obj)

class BlackHole():

    def __init__(self, scene, position, rs):
        scene.add(self)
        self.position = np.array(position, dtype=float)
        self.rs = rs

    def geodesic(self, ray, dλ):
        dλ = dλ
        
        #ray = [t, r, theta, phi, dt, dr, dtheta, dphi]
        t = ray[0]
        r = ray[1]
        theta = ray[2]
        phi = ray[3]

        dt = ray[4]
        dr = ray[5]
        dtheta = ray[6]
        dphi = ray[7]

        rs = self.rs

        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        if abs(sin_theta) < 1e-6: 
            cot_theta = 0 # Kutup noktası yaması
        else:
            cot_theta = cos_theta / sin_theta

        ddt = -1 * ((rs / (r*(r-rs))) * dr * dt)
        ddr = -1 * ((((rs*(r-rs))/(2*r**3))*dt**2) - ((rs/(2*r*(r-rs)))*dr**2) - ((r-rs)*dtheta**2) - ((r-rs)*(sin_theta**2)*(dphi**2)))
        ddtheta = -1 * (((2/r)*dr*dtheta) - (sin_theta*cos_theta*(dphi**2)))
        ddphi = -1 * ((2/r)*dr*dphi + 2*cot_theta*dtheta*dphi)

        dt = dt + ddt*dλ
        dr = dr + ddr*dλ
        dtheta = dtheta + ddtheta*dλ
        dphi = dphi + ddphi*dλ

        t = t + dt*dλ
        r = r + dr*dλ
        theta = theta + dtheta*dλ
        phi = phi + dphi*dλ

        return np.array([t, r, theta, phi, dt, dr, dtheta, dphi])


class Sphere:

    def __init__(self, scene, center, radius, color):
        scene.add(self)
        self.center = np.array(center, dtype=float) 
        self.radius = radius           
        self.color = np.array(color, dtype=float)


