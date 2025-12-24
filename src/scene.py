from shaders import Background

import numpy as np
import math

class Scene:

    def __init__(self):
        self.objects = list()
        self.black_hole = None
        self.back_ground = None

    def add(self, obj):
        if isinstance(obj, BlackHole):
            self.black_hole = obj

        elif isinstance(obj, Background):
            self.back_ground = obj
            
        else:
            self.objects.append(obj)

class BlackHole():

    def __init__(self, scene, position, rs):
        scene.add(self)
        self.position = np.array(position, dtype=float)
        self.rs = rs
        self.dλ = 0.0


    def geodesic(self, ray):
        dλ = self.dλ
        
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

        # Olay Ufku Kontrolü (ışığın mesafesi olay ufkuna 3 adımdan az kaldıysa veya %5 ten daha yakınsa zamanı dondur türevleri 0 döndürerek)
        threshold = rs + max(3*dλ, rs*0.05)
        if (r <= threshold) or (abs(dt) > 1e12): 
            return np.zeros(8)

        epsilon = 1e-5
        if theta < epsilon:
            theta = epsilon
        elif theta > np.pi - epsilon:
            theta = np.pi - epsilon

        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        cot_theta = cos_theta / sin_theta

        ddt = -1 * ((rs / (r*(r-rs))) * dr * dt)
        ddr = -1 * ((((rs*(r-rs))/(2*r**3))*dt**2) - ((rs/(2*r*(r-rs)))*dr**2) - ((r-rs)*dtheta**2) - ((r-rs)*(sin_theta**2)*(dphi**2)))
        ddtheta = -1 * (((2/r)*dr*dtheta) - (sin_theta*cos_theta*(dphi**2)))
        ddphi = -1 * ((2/r)*dr*dphi + 2*cot_theta*dtheta*dphi)

        """
        dt = dt + ddt*dλ
        dr = dr + ddr*dλ
        dtheta = dtheta + ddtheta*dλ
        dphi = dphi + ddphi*dλ

        t = t + dt*dλ
        r = r + dr*dλ
        theta = theta + dtheta*dλ
        phi = phi + dphi*dλ
        """
        
        return np.array([dt, dr, dtheta, dphi, ddt, ddr, ddtheta, ddphi])

    #burada euler metodunun gelişmişini yapıyoruz eulerde yukarıda dλ ile çarpıp değerleri eskilerine ekliyorduk burada şu şekilde yaparız
    #ray = ray + türevler*dλ yani eulerdeki ile aynı mantıkta hesaplyıoruz biz eulerde mesela dt = dt + ddt*dλ diyorduk burada 
    #matrisi direkt döndürüp step size ile çarpıp asıl ışın matrisinin üzerine ekliyoruz yani eulerin adım adım yaptığımızın tek seferde yapılmış hali
    #mesela bu fonksiyonu kullanarak euler hesaplayacak olsaydık new_ray = ray + k1*dλ diyecektik bu zaten eulerin aynısı çünkü k1 yani geodesicin döndürdüğü matris
    #ilgili değerin türevini içeren matris bu işlemde örneğin 0. indexi incelersek ray[0] = ray[0] + k1[0]*dλ olacaktır k1[0] da zaten dt yani t = t + dt*dλ anlamına geliyor bu işlem
    #eulerin aynısı sadece tekte yapmak yerine direkt türev matrisini döndürüyoruz ve onu stepsize ile çarpıp asıl matrisin üstüne ekliyoruz
    def rk4(self, ray):
        ray = ray
        dλ = self.dλ
        
        k1 = self.geodesic(ray)
        #orjinal ışına (ray) onun türev matrisinden dönen değerleri (yani k1 i) 0.5*dλ ile çarpıp ekliyoruz yani bir nevi euler ile yarım adım hesapladık ve sonucu orjinal ışına ekledik (ray + 0.5*dλ*k1) 
        #daha sonra bu ışını kullanarak tekrar türev matrisi döndürdük k3 te kullanmak için bu şekilde 4 kez ilerisi için hesap yaptık ve en son bunların ortalamasını döndüreceğiz
        k2 = self.geodesic(ray + 0.5*dλ*k1) 
        k3 = self.geodesic(ray + 0.5*dλ*k2) 
        k4 = self.geodesic(ray + dλ*k3)

        #en son ölçüm sonuçlarından 2 ve 3 ü 2 katıyla çarpıyoruz diğer 2 ölçüm sonucu 1, ağırlıklarının toplamı 6 ediyor o yüzden 6 ya bölüp step size ile çarpıyoruz ve asıl ışına euler ekleme işlemi gibi ekliyoruz
        #yani asıl ışına ağırlıklı türevlerini bulup stepsize ile çarpıp yine euler mantığıyla ekliyoruz t = t + dt*dλ gibi sadece dt miz 6 tane farklı adımdan bulduğumuz bir ortalama
        return ray + (dλ/6.0) * (k1 + 2*k2 + 2*k3 + k4)


class Sphere:

    def __init__(self, scene, center, radius, color):
        scene.add(self)
        self.center = np.array(center, dtype=float) 
        self.radius = radius           
        self.color = np.array(color, dtype=float)


