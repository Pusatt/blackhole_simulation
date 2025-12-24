import numpy as np
from numba import cuda
import math

# Constants
PI = 3.141592653589793

@cuda.jit(device=True)
def sample_noise_bilinear(noise_tex, u, v):
    h = noise_tex.shape[0]
    w = noise_tex.shape[1]
    
    # Wrap u, v
    u = u - math.floor(u)
    v = v - math.floor(v)
    
    px = u * w
    py = v * h
    
    x0 = int(px)
    y0 = int(py)
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    
    dx = px - x0
    dy = py - y0
    
    val00 = noise_tex[y0, x0]
    val10 = noise_tex[y0, x1]
    val01 = noise_tex[y1, x0]
    val11 = noise_tex[y1, x1]
    
    i1 = val00 * (1 - dx) + val10 * dx
    i2 = val01 * (1 - dx) + val11 * dx
    
    return i1 * (1 - dy) + i2 * dy

@cuda.jit(device=True)
def kelvin_to_rgb_cuda(temp):
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
            
    return r/255.0, g/255.0, b/255.0

@cuda.jit(device=True)
def spherical_to_cartesian_cuda(r, theta, phi):
    sin_theta = math.sin(theta)
    x = r * sin_theta * math.cos(phi)
    y = r * sin_theta * math.sin(phi)
    z = r * math.cos(theta)
    return x, y, z

@cuda.jit(device=True)
def cartesian_to_spherical_cuda(x, y, z):
    r = math.sqrt(x*x + y*y + z*z)
    if r == 0: return 0.0, 0.0, 0.0
    theta = math.atan2(math.sqrt(x*x + y*y), z)
    phi = math.atan2(y, x)
    return r, theta, phi

@cuda.jit(device=True)
def get_disk_temperature(r, inner_radius):
    if r <= inner_radius: return 0.0
    factor = (r ** -0.75) * ((1.0 - math.sqrt(inner_radius / r)) ** 0.25)
    T_max = 50000.0
    return T_max * factor * 5.0

@cuda.jit(device=True)
def get_disk_velocity_vector(x, y, z):
    r = math.sqrt(x*x + y*y)
    if r == 0: return 0.0, 0.0, 0.0
    return -y/r, x/r, 0.0

@cuda.jit(device=True)
def get_disk_velocity_magnitude(r, rs):
    val = (rs / 2.0) / (r - rs)
    if val < 0: return 0.0
    return math.sqrt(val)

@cuda.jit(device=True)
def geodesic_derivs(t, r, theta, phi, ut, ur, utheta, uphi, rs):
    if r <= rs:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    r_minus_rs = r - rs
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    
    a_t = - (rs / (r * r_minus_rs)) * ur * ut
    
    term1 = (rs * r_minus_rs) / (2 * r**3) * (ut**2)
    term2 = rs / (2 * r * r_minus_rs) * (ur**2)
    term3 = r_minus_rs * (utheta**2)
    term4 = r_minus_rs * (sin_theta**2) * (uphi**2)
    a_r = - (term1 - term2 - term3 - term4)
    
    a_theta = - ((2.0 / r) * ur * utheta - sin_theta * cos_theta * (uphi**2))
    
    cot_theta = cos_theta / sin_theta
    a_phi = - ((2.0 / r) * ur * uphi + 2.0 * cot_theta * utheta * uphi)
    
    return ut, ur, utheta, uphi, a_t, a_r, a_theta, a_phi

@cuda.jit(device=True)
def rk4_step(t, r, th, ph, dt, dr, dth, dph, h, rs):
    d1_t, d1_r, d1_th, d1_ph, d1_dt, d1_dr, d1_dth, d1_dph = geodesic_derivs(
        t, r, th, ph, dt, dr, dth, dph, rs
    )
    
    d2_t, d2_r, d2_th, d2_ph, d2_dt, d2_dr, d2_dth, d2_dph = geodesic_derivs(
        t + 0.5*h*d1_t, r + 0.5*h*d1_r, th + 0.5*h*d1_th, ph + 0.5*h*d1_ph,
        dt + 0.5*h*d1_dt, dr + 0.5*h*d1_dr, dth + 0.5*h*d1_dth, dph + 0.5*h*d1_dph, rs
    )

    d3_t, d3_r, d3_th, d3_ph, d3_dt, d3_dr, d3_dth, d3_dph = geodesic_derivs(
        t + 0.5*h*d2_t, r + 0.5*h*d2_r, th + 0.5*h*d2_th, ph + 0.5*h*d2_ph,
        dt + 0.5*h*d2_dt, dr + 0.5*h*d2_dr, dth + 0.5*h*d2_dth, dph + 0.5*h*d2_dph, rs
    )
    
    d4_t, d4_r, d4_th, d4_ph, d4_dt, d4_dr, d4_dth, d4_dph = geodesic_derivs(
        t + h*d3_t, r + h*d3_r, th + h*d3_th, ph + h*d3_ph,
        dt + h*d3_dt, dr + h*d3_dr, dth + h*d3_dth, dph + h*d3_dph, rs
    )

    nt = t + (h/6.0)*(d1_t + 2*d2_t + 2*d3_t + d4_t)
    nr = r + (h/6.0)*(d1_r + 2*d2_r + 2*d3_r + d4_r)
    nth = th + (h/6.0)*(d1_th + 2*d2_th + 2*d3_th + d4_th)
    nph = ph + (h/6.0)*(d1_ph + 2*d2_ph + 2*d3_ph + d4_ph)
    
    ndt = dt + (h/6.0)*(d1_dt + 2*d2_dt + 2*d3_dt + d4_dt)
    ndr = dr + (h/6.0)*(d1_dr + 2*d2_dr + 2*d3_dr + d4_dr)
    ndth = dth + (h/6.0)*(d1_dth + 2*d2_dth + 2*d3_dth + d4_dth)
    ndph = dph + (h/6.0)*(d1_dph + 2*d2_dph + 2*d3_dph + d4_dph)
    
    return nt, nr, nth, nph, ndt, ndr, ndth, ndph

# --- BURASI DEĞİŞTİRİLDİ (1D -> 2D) ---
@cuda.jit
def render_kernel(results, width, height, fov, cam_pos, cam_dir, cam_right, cam_up, rs, bg_data, noise_tex, disk_inner, disk_outer, dlam, maxlam):
    # 2D Grid Koordinatları
    x, y = cuda.grid(2)
    
    # Sınır Kontrolü
    if x >= width or y >= height:
        return
        
    # Doğrusal Index (Sonuç arrayine yazmak için)
    idx = y * width + x
    
    # J ve I değerleri artık doğrudan X ve Y'dir
    j = x
    i = y
    
    # Aspect Ratio
    aspect_ratio = width / height
    scale = math.tan(math.radians(fov) / 2.0)
    
    # Screen Coords
    px = (2 * (j + 0.5) / width - 1) * scale * aspect_ratio
    py = (1 - 2 * (i + 0.5) / height) * scale
    
    # Ray Direction Calculation
    ray_dir_x = cam_dir[0] + px*cam_right[0] + py*cam_up[0]
    ray_dir_y = cam_dir[1] + px*cam_right[1] + py*cam_up[1]
    ray_dir_z = cam_dir[2] + px*cam_right[2] + py*cam_up[2]
    
    norm = math.sqrt(ray_dir_x**2 + ray_dir_y**2 + ray_dir_z**2)
    dir_x = ray_dir_x / norm
    dir_y = ray_dir_y / norm
    dir_z = ray_dir_z / norm
    
    # Initial State
    ir, ith, iph = cartesian_to_spherical_cuda(cam_pos[0], cam_pos[1], cam_pos[2])
    
    ix = cam_pos[0]
    iy = cam_pos[1]
    iz = cam_pos[2]
    
    idr = (ix*dir_x + iy*dir_y + iz*dir_z) / ir
    
    xy2 = ix**2 + iy**2
    sqrt_xy2 = math.sqrt(xy2)
    
    idth = 0.0
    idph = 0.0
    
    if xy2 > 1e-9:
        num_theta = iz * (ix*dir_x + iy*dir_y) - xy2*dir_z
        den_theta = (ir**2) * sqrt_xy2
        idth = num_theta / den_theta
        
        idph = (ix*dir_y - iy*dir_x) / xy2
        
    factor = 1.0 - (rs / ir)
    spatial = (idr**2 / factor) + (ir**2 * idth**2) + (ir**2 * math.sin(ith)**2 * idph**2)
    idt = math.sqrt(spatial / factor)
    
    # Current State Variables
    t, r, th, ph = 0.0, ir, ith, iph
    dt, dr, dth, dph = idt, idr, idth, idph
    
    lam = 0.0
    
    # Background Params
    bg_h = bg_data.shape[0]
    bg_w = bg_data.shape[1]
    height_increment = bg_h / PI
    width_increment = bg_w / (2.0 * PI)
    
    # Loop
    while lam <= maxlam:
        prev_th = th
        
        # RK4 Step
        t, r, th, ph, dt, dr, dth, dph = rk4_step(t, r, th, ph, dt, dr, dth, dph, dlam, rs)
        lam += dlam
        
        # Event Horizon
        if r < rs * 1.01:
            results[idx, 0] = 0.0
            results[idx, 1] = 0.0
            results[idx, 2] = 0.0
            return
            
        # Infinity / Background
        if r > 50.0:
            theta_bg = th % PI
            phi_bg = ph % (2*PI)
            if phi_bg < 0: phi_bg += 2*PI
            
            bx = phi_bg * width_increment
            by = theta_bg * height_increment
            
            bx0 = int(bx) % bg_w
            by0 = int(by)
            if by0 >= bg_h: by0 = bg_h - 1
            
            results[idx, 0] = bg_data[by0, bx0, 0] / 255.0
            results[idx, 1] = bg_data[by0, bx0, 1] / 255.0
            results[idx, 2] = bg_data[by0, bx0, 2] / 255.0
            return

        # Accretion Disk Physics
        crosses_plane = False
        if (prev_th < PI/2.0 and th >= PI/2.0) or (prev_th > PI/2.0 and th <= PI/2.0):
            crosses_plane = True
            
        if crosses_plane and (disk_inner <= r <= disk_outer):
            temp = get_disk_temperature(r, disk_inner)
            cr, cg, cb = kelvin_to_rgb_cuda(temp)
            
            u_noise = (r - disk_inner) / (disk_outer - disk_inner)
            v_noise = (ph % (2*PI)) / (2*PI)
            v_noise += u_noise * 1.5
            
            noise_val = sample_noise_bilinear(noise_tex, u_noise * 1.0, v_noise * 4.0) 
            noise_val = noise_val ** 1.5
            
            cr *= noise_val
            cg *= noise_val
            cb *= noise_val

            px, py, pz = spherical_to_cartesian_cuda(r, PI/2.0, ph)
            vx, vy, vz = get_disk_velocity_vector(px, py, pz)
            v_mag = get_disk_velocity_magnitude(r, rs)
            
            sin_ph = math.sin(ph)
            cos_ph = math.cos(ph)
            
            rvx = dr * cos_ph - r * sin_ph * dph
            rvy = dr * sin_ph + r * cos_ph * dph
            rvz = -r * dth
            
            rv_norm = math.sqrt(rvx*rvx + rvy*rvy + rvz*rvz)
            if rv_norm > 0:
                rvx /= rv_norm
                rvy /= rv_norm
                rvz /= rv_norm
                
            pdx = -rvx
            pdy = -rvy
            pdz = -rvz
            
            cos_xi = pdx*vx + pdy*vy + pdz*vz
            
            v = v_mag
            if v >= 0.99: v = 0.99
            
            g_doppler = 1.0 # Basitleştirilmiş
            g_grav = 0.0
            if r > rs:
                g_grav = math.sqrt(1.0 - rs / r)
                
            g = g_grav * g_doppler
            
            obs_temp = temp * g
            ocr, ocg, ocb = kelvin_to_rgb_cuda(obs_temp)
            
            intensity = g**4 * noise_val * 2.5
            
            f_r = ocr * intensity
            f_g = ocg * intensity
            f_b = ocb * intensity
            
            if f_r > 1.0: f_r = 1.0
            if f_g > 1.0: f_g = 1.0
            if f_b > 1.0: f_b = 1.0
            
            results[idx, 0] = f_r
            results[idx, 1] = f_g
            results[idx, 2] = f_b
            return
            
    results[idx, 0] = 0.0
    results[idx, 1] = 0.0
    results[idx, 2] = 0.0

# --- BURASI DEĞİŞTİRİLDİ (Konfigürasyon) ---
def render_cuda(width, height, fov, cam_pos, cam_dir, cam_right, cam_up, rs, bg_device, noise_device, disk_inner, disk_outer, dlam, maxlam):
    size = width * height
    results_device = cuda.device_array((size, 3), dtype=np.float32)
    
    # 16x16 = 256 Thread. 
    # Ama 2D blok yapısı GPU'nun daha verimli çalışmasını sağlar.
    threadsperblock = (16, 16)
    
    blockspergrid_x = int(math.ceil(width / 16))
    blockspergrid_y = int(math.ceil(height / 16))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    render_kernel[blockspergrid, threadsperblock](
        results_device, width, height, fov, 
        cam_pos, cam_dir, cam_right, cam_up, 
        rs, bg_device, noise_device, disk_inner, disk_outer, dlam, maxlam
    )
    
    # Sonucu Host'a (CPU) çek
    results = results_device.copy_to_host()
    return results.reshape((height, width, 3))