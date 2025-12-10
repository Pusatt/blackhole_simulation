import numpy as np
import matplotlib.pyplot as plt

class SchwarzschildGeodesic:
    def __init__(self, rs):
        # rs: Schwarzschild yarıçapı (2GM/c^2)
        self.rs = rs # kara delik yarıçapı

    def get_derivatives(self, state):
        # PDF'teki ivme denklemleri bu kısımda 
        # Girdi: [t, r, theta, phi, dt_dlam, dr_dlam, dtheta_dlam, dphi_dlam]
        # Çıktı: [dt, dr, dtheta, dphi, d2t, d2r, d2theta, d2phi]
        t, r, theta, phi, ut, ur, utheta, uphi = state 
        rs = self.rs
        
        # Singularity kontrolü. Eğer ışık olay ufkuna çarparsa durur.
        if r <= rs:
            return np.zeros_like(state) # Olay ufkuna çarptı

        # Ortak terimler
        r_minus_rs = r - rs
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # ******************** İVME DENKLEMLERİ ********************
        
        # A) Zaman İvmesi (d2t)
        # PDF Formülü: -(rs / (r * (r - rs))) * dr * dt
        # Bu kısım karadeliğin zamanı nasıl büktüğünü hesaplar. PDF'teki d2t denklemi
        a_t = - (rs / (r * r_minus_rs)) * ur * ut

        # B) Radyal İvme (d2r)
        # Bu en karmaşık denklem
        term1 = (rs * r_minus_rs) / (2 * r**3) * (ut**2)
        term2 = rs / (2 * r * r_minus_rs) * (ur**2)
        # Term1 ve Term2 ışığın kütleçekim ile çekilmesini hesaplar
        term3 = r_minus_rs * (utheta**2)
        term4 = r_minus_rs * (sin_theta**2) * (uphi**2)
        # Term4 merkezkaç etkisidir 
        
        # PDF'teki formül: -(term1 - term2 - term3 - term4)
        a_r = - (term1 - term2 - term3 - term4)
        # Bu denklemde a_r negatif çıkarsa ışık içeri çekilir, pozitif çıkarsa dışarı itilir.

        # C) Polar Açı İvmesi (d2theta)
        # PDF Formülü: -( (2/r)*dr*dtheta - sin*cos*dphi^2 )
        # Parantez içindeki eksi işareti dağılınca + olur. Açısal Momentumun korunumu.
        a_theta = - ((2.0 / r) * ur * utheta - sin_theta * cos_theta * (uphi**2))

        # D) Azimutal Açı İvmesi (d2phi)
        # PDF Formülü: -( (2/r)*dr*dphi + 2*cot*dtheta*dphi )
        cot_theta = cos_theta / sin_theta
        a_phi = - ((2.0 / r) * ur * uphi + 2.0 * cot_theta * utheta * uphi)

        # Türevleri döndüren kısım, buradaki çıktı bize bir sonraki adımı söyler.
        return np.array([ut, ur, utheta, uphi, a_t, a_r, a_theta, a_phi])

    def rk4_step(self, state, h):
        # h: Adım büyüklüğü (step size)
        k1 = h * self.get_derivatives(state)
        k2 = h * self.get_derivatives(state + 0.5 * k1)
        k3 = h * self.get_derivatives(state + 0.5 * k2)
        k4 = h * self.get_derivatives(state + k3)
        
        new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        # Bu kısım kabaca durumlar içinden ağırlıklı ortalama alır ve yeni durumu hesaplar
        return new_state

# --- YENİ EKLENEN FİZİK HESAPLAYICI ---
def get_precise_initial_state(b, r_start, rs):
    """
    Verilen impact parameter (b) için tam fiziksel hızları hesaplar.
    Bu fonksiyon, "elle" hız girmek yerine korunum yasalarını kullanır.
    """
    # 1. Enerjiyi (E) 1 birim kabul edelim (Foton için ölçek fark etmez)
    E = 1.0
    
    # 2. Açısal Momentum (L) = b * E
    L = b * E
    
    # 3. Konumlar
    theta = np.pi / 2
    phi = 0.0
    
    # 4. Schwarzschild Metriğinden türetilen hızlar
    # Zaman hızı (dt/dlambda)
    ut = E / (1.0 - rs / r_start)
    
    # Açısal hız (dphi/dlambda)
    uphi = L / (r_start**2)
    
    # Radyal hız (dr/dlambda) - Null Geodesic Şartından (ds^2=0 olmalı)
    # -(1-rs/r)ut^2 + (1/(1-rs/r))ur^2 + r^2*uphi^2 = 0 denkleminin çözümü:
    metric_factor = 1.0 - rs / r_start
    term_t = -metric_factor * (ut**2)
    term_phi = (r_start**2) * (uphi**2)
    
    ur_squared = -metric_factor * (term_t + term_phi)
    
    if ur_squared < 0:
        return None # Fiziksel olarak imkansız durum
        
    ur = -np.sqrt(ur_squared) # Eksi çünkü merkeze doğru gidiyor
    
    return np.array([0.0, r_start, theta, phi, ut, ur, 0.0, uphi])

# --- TEST KISMI ---

def run_test():
    # 1. Ayarlar *********************
    rs = 1.0  # Schwarzschild yarıçapını 1 birim kabul ettik 
    solver = SchwarzschildGeodesic(rs)
    
    # 2. Başlangıç Koşulları *********************
    # Işığı kara deliğe doğru ama biraz yandan gönderdik
    r_start = 20.0 * rs
    
    # Impact parameter (b) ayarı: Işının kara delikten ne kadar yandan geçeceğini ile ilgili 
    # Yani açısal momentumu belirler. Eğer b < 2.6 rs ise düşer, b > 2.6 rs ise sapar.
    # 2.6 rastgele bir değer değildir. Schwarzschild metrikte ışığın karadeliğe düşmemesi için gereken minimum yarıçaptır
    impact_parameter = 3 * rs  
    
    current_state = get_precise_initial_state(impact_parameter, r_start, rs)
    if current_state is None:
        print("Hata: Başlangıç koşulları fiziksel değil.")
        return

    # 3. Simülasyon Döngüsü *********************
    steps = 10000 # Adım sayısı artırıldı
    h = 0.05      # DAHA HASSAS HESAP İÇİN ADIM BÜYÜKLÜĞÜNÜ KÜÇÜLT
    
    trajectory_x = []
    trajectory_y = []

    print(f"\nSimülasyon b={impact_parameter/rs} Rs için başladı...")
    for i in range(steps):
        # Ekrana çizdirmek için koordinatları dönüştürüyoruz
        r_val = current_state[1]
        phi_val = current_state[3]
        
        x = r_val * np.cos(phi_val)
        y = r_val * np.sin(phi_val)
        
        trajectory_x.append(x)
        trajectory_y.append(y)
        
        # RK4 Adımı
        current_state = solver.rk4_step(current_state, h)
        
        # Olay ufkuna düştü mü kontrolü
        if current_state[1] < rs * 1.01:
            print(f"Işık {i}. adımda olay ufkuna düştü! (Simülasyon Başarılı)")
            break
            
        # Çok uzaklaştı mı?
        if current_state[1] > r_start * 1.5:
            print(f"Işık {i}. adımda sistemden kaçtı.")
            break

    # 4. Görselleştirme *********************
    plt.figure(figsize=(10, 6))
    
    # Olay Ufku
    circle = plt.Circle((0, 0), rs, color='black', label='Olay Ufku ($R_s$)')
    plt.gca().add_patch(circle)
    
    # Foton Yörüngesi (Photon Sphere - 1.5 Rs)
    circle_ph = plt.Circle((0, 0), 1.5*rs, color='red', fill=False, linestyle='--', label='Foton Yörüngesi')
    plt.gca().add_patch(circle_ph)
    
    # Işığın Yolu
    plt.plot(trajectory_x, trajectory_y, label='Işık Yolu', color='blue')
    
    # Kaynak
    plt.scatter(trajectory_x[0], trajectory_y[0], color='orange', label='Kaynak')
    
    plt.title(f"Test Simülasyonu (b={impact_parameter/rs} Rs)")
    plt.xlabel("x / Rs")
    plt.ylabel("y / Rs")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    run_test()
