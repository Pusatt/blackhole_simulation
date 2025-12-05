import numpy as np
import matplotlib.pyplot as plt

# --- SABİTLER ---
# (Standart olarak G=1, M=1, c=1)
G = 1.0
M = 1.0  # Karadeliğin kütlesi
C = 1.0  # Işık hızı
SCHWARZSCHILD_RADIUS = 2 * G * M / (C**2) # Olay Ufku Yarıçapı
# True = RK4 kullan, False = Euler kullan
USE_RK4 = False

# =============================================================================
# BÖLÜM 1: FİZİK MOTORU (Ömer ve Enes)
# =============================================================================
def calculate_acceleration(position, velocity):
    """
    Giriş: Konum (x,y,z), Hız (vx,vy,vz)
    Çıkış: İvme (ax,ay,az)
    Not: Gerçek projede burada Christoffel sembolleri ile Geodesic denklemi olacak.
    """
    x, y, z = position
    r_sq = x**2 + y**2 + z**2
    # koruma: r==0 durumundan kaçın
    if r_sq == 0.0:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    r = np.sqrt(r_sq)

    # --- TEST FİZİĞİ (Newtonian) ---
    factor = -1.5 * G * M / (r_sq * r)
    ax = factor * x
    ay = factor * y
    az = factor * z

    return np.array([ax, ay, az], dtype=float)

# =============================================================================
# BÖLÜM 2: ENTEGRATÖRLER
# =============================================================================
def euler_step(position, velocity, dt):
    a = calculate_acceleration(position, velocity)
    new_pos = position + velocity * dt
    new_vel = velocity + a * dt
    return new_pos, new_vel

def rk4_step(position, velocity, dt):
    """
    Runge-Kutta 4. Derece entegrasyonu ile bir sonraki adımı hesaplar.
    """
    k1_v = velocity
    k1_a = calculate_acceleration(position, velocity)

    p2 = position + k1_v * (dt / 2)
    v2 = velocity + k1_a * (dt / 2)
    k2_v = v2
    k2_a = calculate_acceleration(p2, v2)

    p3 = position + k2_v * (dt / 2)
    v3 = velocity + k2_a * (dt / 2)
    k3_v = v3
    k3_a = calculate_acceleration(p3, v3)

    p4 = position + k3_v * dt
    v4 = velocity + k3_a * dt
    k4_v = v4
    k4_a = calculate_acceleration(p4, v4)

    new_position = position + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    new_velocity = velocity + (dt / 6.0) * (k1_a + 2*k2_a + 2*k3_a + k4_a)

    return new_position, new_velocity

# =============================================================================
# BÖLÜM 3: SİMÜLASYON YÖNETİCİSİ (Main Loop)
# =============================================================================
def trace_ray(start_pos, start_vel, max_steps=5000, dt=0.05):
    """
    Bir ışın için tüm yörüngeyi hesaplar.
    RK4 geçici olarak kapatılabilir (USE_RK4 sabiti).
    """
    history = [start_pos]
    current_pos = np.array(start_pos, dtype=float)
    current_vel = np.array(start_vel, dtype=float)

    status = "escaped" # Varsayılan durum: kaçtı
    
    if USE_RK4:
        print("Integrator: RK4 (aktif)")
    else:
        print("Integrator: Euler (RK4 geçici olarak kapalı)")

    for step in range(max_steps):
        # adım at (USE_RK4 kontrolü)
        if USE_RK4:
            current_pos, current_vel = rk4_step(current_pos, current_vel, dt)
        else:
            current_pos, current_vel = euler_step(current_pos, current_vel, dt)

        history.append(current_pos)

        # Olay Ufku Kontrolü (Karadeliğe düştü mü?)
        dist_from_center = np.linalg.norm(current_pos)
        if dist_from_center < SCHWARZSCHILD_RADIUS:
            status = "captured"
            break

        # Çok uzaklaştı mı? (Simülasyon dışına çıktı mı?)
        if dist_from_center > 50.0:
            status = "escaped"
            break

    return np.array(history), status

# =============================================================================
# TEST VE GÖRSELLEŞTİRME
# =============================================================================
if __name__ == "__main__":
    # Örnek: Uzaktan geçen bir ışın (bükülüp yoluna devam etmeli)
    start_p = np.array([10.0, 3.0, 0.0])
    start_v = np.array([-1.0, 0.0, 0.0])

    path, result = trace_ray(start_p, start_v)

    print(f"Simülasyon Bitti! Sonuç: {result}")
    print(f"Toplam Adım: {len(path)}")

    # 3D Çizim
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Karadeliği Temsilen Siyah Küre
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_bh = SCHWARZSCHILD_RADIUS * np.cos(u) * np.sin(v)
    y_bh = SCHWARZSCHILD_RADIUS * np.sin(u) * np.sin(v)
    z_bh = SCHWARZSCHILD_RADIUS * np.cos(v)
    ax.plot_surface(x_bh, y_bh, z_bh, color='black', alpha=0.8)

    # Işığın Yolu
    ax.plot(path[:,0], path[:,1], path[:,2], color='cyan', linewidth=2, label='Foton Yolu')

    # Başlangıç noktası
    ax.scatter(start_p[0], start_p[1], start_p[2], color='red', s=50, label='Başlangıç')

    # Ayarlar
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Integrator={"RK4" if USE_RK4 else "Euler"} - Işın Durumu: {result}')
    ax.legend()

    # Eşit ölçekleme (Görüntü bozulmasın diye)
    limit = 12
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])

    plt.show()
