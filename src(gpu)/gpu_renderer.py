import numpy as np
from numba import cuda
import math


# =========================
# GPU: GEODESIC STEP
# =========================
@cuda.jit(device=True)
def geodesic_step(ray, dlam, rs):
    # ray = [t, r, th, ph, dt, dr, dth, dph]
    t = ray[0]
    r = ray[1]
    th = ray[2]
    ph = ray[3]

    dt = ray[4]
    dr = ray[5]
    dth = ray[6]
    dph = ray[7]

    eps = 1e-5
    if th < eps:
        th = eps
    elif th > math.pi - eps:
        th = math.pi - eps

    sin_t = math.sin(th)
    cos_t = math.cos(th)
    cot_t = cos_t / sin_t

    ddt = -((rs / (r * (r - rs))) * dr * dt)

    ddr = -(
        ((rs * (r - rs)) / (2 * r**3)) * dt * dt
        - ((rs / (2 * r * (r - rs))) * dr * dr)
        - ((r - rs) * dth * dth)
        - ((r - rs) * (sin_t * sin_t) * (dph * dph))
    )

    ddth = -(((2 / r) * dr * dth) - (sin_t * cos_t * (dph * dph)))
    ddph = -((2 / r) * dr * dph + 2 * cot_t * dth * dph)

    dt += ddt * dlam
    dr += ddr * dlam
    dth += ddth * dlam
    dph += ddph * dlam

    ray[0] = t + dt * dlam
    ray[1] = r + dr * dlam
    ray[2] = th + dth * dlam
    ray[3] = ph + dph * dlam
    ray[4] = dt
    ray[5] = dr
    ray[6] = dth
    ray[7] = dph


# =========================
# GPU: RAY MARCH KERNEL
# =========================
@cuda.jit
def ray_march_kernel(rays, results, rs, dlam, maxlam):
    idx = cuda.grid(1)
    if idx >= rays.shape[0]:
        return

    ray = rays[idx]
    lam = 0.0

    while lam < maxlam:
        # Event horizon
        if ray[1] < rs * 1.01:
            results[idx, 0] = -1.0  # swallowed
            return

        # Escaped to infinity
        if ray[1] > 50.0:
            results[idx, 0] = ray[1]
            results[idx, 1] = ray[2]
            results[idx, 2] = ray[3]
            return

        geodesic_step(ray, dlam, rs)
        lam += dlam

    # fallback
    results[idx, 0] = -2.0


# =========================
# CPU API
# =========================
def render_gpu(scene, camera):
    """
    GPU sadece ray marching yapar.
    CPU background renklendirme yapar.
    """

    # 1) CPU: initial rays
    rays_cpu = camera.prepare_initial_rays()
    h, w, _ = rays_cpu.shape
    rays_cpu = rays_cpu.reshape((-1, 8))

    # 2) GPU buffers
    rays_gpu = cuda.to_device(rays_cpu)
    results_gpu = cuda.device_array((rays_cpu.shape[0], 3), dtype=np.float32)

    # 3) launch
    threads = 256
    blocks = (rays_cpu.shape[0] + threads - 1) // threads

    ray_march_kernel[blocks, threads](
        rays_gpu,
        results_gpu,
        scene.black_hole.rs,
        camera.dλ,
        camera.maxλ
    )

    # 4) back to CPU
    results = results_gpu.copy_to_host()

    # 5) CPU: color
    image = np.zeros((h * w, 3), dtype=np.float32)

    for i in range(h * w):
        r = results[i, 0]

        if r < 0:  # swallowed or error
            image[i] = np.array([0.0, 0.0, 0.0])
        else:
            image[i] = scene.back_ground.get_background_color(
                results[i]
            )

    return image.reshape((h, w, 3))
