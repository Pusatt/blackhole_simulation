import numpy as np
from numba import jit, prange
import math

# Constants
PI = np.pi

@jit(nopython=True)
def kelvin_to_rgb_numba(temp):
    temp = temp / 100.0
    
    r = 0.0
    g = 0.0
    b = 0.0
    
    # Red
    if temp <= 66:
        r = 255.0
    else:
        r = temp - 60.0
        r = 329.698727446 * (r ** -0.1332047592)
        if r < 0: r = 0.0
        if r > 255: r = 255.0
        
    # Green
    if temp <= 66:
        r_temp = temp
        g = 99.4708025861 * (math.log(r_temp)) - 161.1195681661
        if g < 0: g = 0.0
        if g > 255: g = 255.0
    else:
        r_temp = temp - 60.0
        g = 288.1221695283 * (r_temp ** -0.0755148492)
        if g < 0: g = 0.0
        if g > 255: g = 255.0
        
    # Blue
    if temp >= 66:
        b = 255.0
    else:
        if temp <= 19:
            b = 0.0
        else:
            b = temp - 10.0
            b = 138.5177312231 * (math.log(b)) - 305.0447927307
            if b < 0: b = 0.0
            if b > 255: b = 255.0
            
    return np.array([r/255.0, g/255.0, b/255.0])

@jit(nopython=True)
def spherical_to_cartesian_numba(r, theta, phi):
    sin_theta = math.sin(theta)
    x = r * sin_theta * math.cos(phi)
    y = r * sin_theta * math.sin(phi)
    z = r * math.cos(theta)
    return np.array([x, y, z])

@jit(nopython=True)
def get_disk_temperature(r, inner_radius):
    if r <= inner_radius: return 0.0
    factor = (r ** -0.75) * ((1.0 - math.sqrt(inner_radius / r)) ** 0.25)
    T_max = 50000.0
    return T_max * factor * 5.0

@jit(nopython=True)
def get_disk_velocity_vector(x, y, z):
    r = math.sqrt(x*x + y*y)
    if r == 0: return np.array([0.0, 0.0, 0.0])
    # Tangent vector (-y, x, 0) / r
    return np.array([-y/r, x/r, 0.0])

@jit(nopython=True)
def get_disk_velocity_magnitude(r, rs):
    val = (rs / 2.0) / (r - rs)
    if val < 0: return 0.0
    return math.sqrt(val)

@jit(nopython=True)
def geodesic_derivs(state, rs):
    # state: [t, r, theta, phi, ut, ur, utheta, uphi]
    t, r, theta, phi, ut, ur, utheta, uphi = state 
    
    if r <= rs:
        return np.zeros(8)

    r_minus_rs = r - rs
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    
    # a_t
    a_t = - (rs / (r * r_minus_rs)) * ur * ut
    
    # a_r
    term1 = (rs * r_minus_rs) / (2 * r**3) * (ut**2)
    term2 = rs / (2 * r * r_minus_rs) * (ur**2)
    term3 = r_minus_rs * (utheta**2)
    term4 = r_minus_rs * (sin_theta**2) * (uphi**2)
    a_r = - (term1 - term2 - term3 - term4)
    
    # a_theta
    a_theta = - ((2.0 / r) * ur * utheta - sin_theta * cos_theta * (uphi**2))
    
    # a_phi
    cot_theta = cos_theta / sin_theta
    a_phi = - ((2.0 / r) * ur * uphi + 2.0 * cot_theta * utheta * uphi)
    
    return np.array([ut, ur, utheta, uphi, a_t, a_r, a_theta, a_phi])

@jit(nopython=True)
def rk4_step(state, h, rs):
    k1 = h * geodesic_derivs(state, rs)
    k2 = h * geodesic_derivs(state + 0.5 * k1, rs)
    k3 = h * geodesic_derivs(state + 0.5 * k2, rs)
    k4 = h * geodesic_derivs(state + k3, rs)
    return state + (k1 + 2*k2 + 2*k3 + k4) / 6.0

@jit(nopython=True)
def solve_ray(ray_init, rs, bg_height, bg_width, bg_data, disk_inner, disk_outer, disk_color, dlam, maxlam):
    # ray_init: [t, r, theta, phi, dt, dr, dtheta, dphi]
    ray = ray_init.copy()
    lam = 0.0
    
    prev_theta = ray[2]
    
    # Background logic helpers
    height_increment = bg_height/PI
    width_increment = bg_width/(2*PI)
    
    while lam <= maxlam:
        prev_theta = ray[2]
        
        # Step
        ray = rk4_step(ray, dlam, rs)
        lam += dlam
        
        current_r = ray[1]
        current_theta = ray[2]
        current_phi = ray[3]
        
        # Event Horizon
        if current_r < rs * 1.01:
            return np.array([0.0, 0.0, 0.0]) # Black
        
        # Infinity / Background
        if current_r > 50.0:
            # Map theta, phi to bg_data
            # Normalize angles
            theta_bg = current_theta % PI # Approximation
            phi_bg = current_phi % (2*PI)
            if phi_bg < 0: phi_bg += 2*PI
            
            x = phi_bg * width_increment
            y = theta_bg * height_increment
            
            x0 = int(x) % bg_width
            y0 = int(y)
            y0 = min(y0, bg_height - 1)
            
            # Simple nearest or bilinear - let's do simple for speed first
            # bg_data is uint8 array [h, w, 3]
            col = bg_data[y0, x0]
            return np.array([col[0]/255.0, col[1]/255.0, col[2]/255.0])
        
        # Accretion Disk Check
        # Check crossing PI/2
        crosses_plane = False
        if (prev_theta < PI/2 and current_theta >= PI/2) or (prev_theta > PI/2 and current_theta <= PI/2):
            crosses_plane = True
            
        if crosses_plane:
            r_cross = current_r
            if disk_inner <= r_cross <= disk_outer:
                 # PHYSICS COLORING
                temp = get_disk_temperature(r_cross, disk_inner)
                base_color = kelvin_to_rgb_numba(temp)
                
                # Velocity
                pos = spherical_to_cartesian_numba(r_cross, PI/2, current_phi)
                v_tan_dir = get_disk_velocity_vector(pos[0], pos[1], pos[2])
                v_mag = get_disk_velocity_magnitude(r_cross, rs)
                
                # Ray Direction (Cartesian) - needing derivatives
                # ray: [t, r, theta, phi, dt, dr, dtheta, dphi]
                dr = ray[5]
                dth = ray[6]
                dph = ray[7]
                
                sin_th = 1.0 # at pi/2
                cos_th = 0.0
                sin_ph = math.sin(current_phi)
                cos_ph = math.cos(current_phi)
                
                # At theta=pi/2:
                # vx = dr * cos_ph - r * sin_ph * dph
                # vy = dr * sin_ph + r * cos_ph * dph
                # vz = - r * dth
                
                vx = dr * cos_ph - r_cross * sin_ph * dph
                vy = dr * sin_ph + r_cross * cos_ph * dph
                vz = - r_cross * dth
                
                ray_dir = np.array([vx, vy, vz])
                norm_val = math.sqrt(vx*vx + vy*vy + vz*vz)
                if norm_val > 0:
                    ray_dir = ray_dir / norm_val
                
                photon_dir = -1.0 * ray_dir
                
                # Dot product
                cos_xi = photon_dir[0]*v_tan_dir[0] + photon_dir[1]*v_tan_dir[1] + photon_dir[2]*v_tan_dir[2]
                
                v = v_mag
                if v >= 0.99: v = 0.99
                
                g_doppler = math.sqrt(1.0 - v*v) / (1.0 - v * cos_xi)
                
                g_grav = 0.0
                if r_cross > rs:
                    g_grav = math.sqrt(1.0 - rs / r_cross)
                    
                g = g_grav * g_doppler
                
                observed_temp = temp * g
                obs_color = kelvin_to_rgb_numba(observed_temp)
                
                intensity = g**4
                final_col = obs_color * intensity
                
                # Clip
                for k in range(3):
                    if final_col[k] > 1.0: final_col[k] = 1.0
                    if final_col[k] < 0.0: final_col[k] = 0.0
                    
                return final_col
                
    return np.array([0.0, 0.0, 0.0]) # Failed/Timeout

@jit(nopython=True)
def cartesian_to_spherical_numba(x, y, z):
    r = math.sqrt(x*x + y*y + z*z)
    if r == 0: return np.array([0.0, 0.0, 0.0])
    theta = math.atan2(math.sqrt(x*x + y*y), z)
    phi = math.atan2(y, x)
    return np.array([r, theta, phi])

@jit(nopython=True)
def generate_ray(h_idx, w_idx, height, width, cam_pos, cam_dir, cam_right, cam_up, fov, aspect_ratio, rs):
    # cam_pos is cartesian [x, y, z] relative to BH center implies center is 0,0,0
    
    # 1. Screen coord
    scale = math.tan(math.radians(fov) / 2.0)
    
    # x
    px = (2 * (w_idx + 0.5) / width - 1) * scale * aspect_ratio
    # y
    py = (1 - 2 * (h_idx + 0.5) / height) * scale
    
    # 2. Ray Direction (Cartesian)
    # dir = forward + px*right + py*up
    # All vectors assumed normalized
    ray_dir_x = cam_dir[0] + px*cam_right[0] + py*cam_up[0]
    ray_dir_y = cam_dir[1] + px*cam_right[1] + py*cam_up[1]
    ray_dir_z = cam_dir[2] + px*cam_right[2] + py*cam_up[2]
    
    # Normalize
    norm = math.sqrt(ray_dir_x**2 + ray_dir_y**2 + ray_dir_z**2)
    ray_dir_x /= norm
    ray_dir_y /= norm
    ray_dir_z /= norm
    
    # 3. Initial State for Geodesic
    # ray = [t, r, theta, phi, dt, dr, dtheta, dphi]
    
    # Relative pos (spherical)
    sph = cartesian_to_spherical_numba(cam_pos[0], cam_pos[1], cam_pos[2])
    initial_r = sph[0]
    initial_theta = sph[1]
    initial_phi = sph[2]
    
    initial_x = cam_pos[0]
    initial_y = cam_pos[1]
    initial_z = cam_pos[2]
    
    # Ray dir is [dx, dy, dz]
    dir_x = ray_dir_x
    dir_y = ray_dir_y
    dir_z = ray_dir_z
    
    # Convert velocity to spherical (dr, dtheta, dphi)
    # dr = (x*dx + y*dy + z*dz) / r
    initial_dr = (initial_x*dir_x + initial_y*dir_y + initial_z*dir_z) / initial_r
    
    xy2 = initial_x**2 + initial_y**2
    sqrt_xy2 = math.sqrt(xy2)
    
    if xy2 < 1e-9:
        initial_dtheta = 0.0
        initial_dphi = 0.0
    else:
        # dtheta = (z*(x*dx + y*dy) - (x^2+y^2)*dz) / (r^2 * sqrt(x^2+y^2))
        num_theta = initial_z * (initial_x*dir_x + initial_y*dir_y) - xy2*dir_z
        den_theta = (initial_r**2) * sqrt_xy2
        initial_dtheta = num_theta / den_theta
        
        # dphi = (x*dy - y*dx) / (x^2 + y^2)
        initial_dphi = (initial_x*dir_y - initial_y*dir_x) / xy2
        
    # dt calculation
    # factor = 1 - rs/r
    factor = 1.0 - (rs / initial_r)
    
    spatial = (initial_dr**2 / factor) + \
              (initial_r**2 * initial_dtheta**2) + \
              (initial_r**2 * math.sin(initial_theta)**2 * initial_dphi**2)
              
    initial_dt = math.sqrt(spatial / factor)
    
    return np.array([0.0, initial_r, initial_theta, initial_phi, initial_dt, initial_dr, initial_dtheta, initial_dphi])

@jit(nopython=True, parallel=True)
def render_numba(width, height, fov, cam_pos, cam_dir, cam_right, cam_up, rs, bg_data, disk_inner, disk_outer, disk_color, dlam, maxlam):
    # Output image
    image = np.zeros((height, width, 3), dtype=np.float32)
    aspect_ratio = width / height
    
    bg_h = bg_data.shape[0]
    bg_w = bg_data.shape[1]
    
    for i in prange(height):
        for j in range(width):
            # Generate Ray
            ray = generate_ray(i, j, height, width, cam_pos, cam_dir, cam_right, cam_up, fov, aspect_ratio, rs)
            
            # Solve Ray
            col = solve_ray(ray, rs, bg_h, bg_w, bg_data, disk_inner, disk_outer, disk_color, dlam, maxlam)
            
            image[i, j, 0] = col[0]
            image[i, j, 1] = col[1]
            image[i, j, 2] = col[2]
            
    return image
