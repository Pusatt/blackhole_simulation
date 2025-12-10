import numpy as np
import math

def spherical_to_cartesian(vector):
    vector = np.array(vector, dtype=float)
    
    r = vector[0]
    theta = vector[1]
    phi = vector[2]

    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    
    return np.array([x, y, z], dtype=float)

def cartesian_to_spherical(vector):
    vector = np.array(vector, dtype=float)

    x = vector[0]
    y = vector[1]
    z = vector[2]

    r = math.sqrt(x**2 + y**2 + z**2)
    
    if r == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    
    theta = math.atan2(math.sqrt(x**2 + y**2), z)
    phi = math.atan2(y, x)

    return np.array([r, theta, phi], dtype=float)
