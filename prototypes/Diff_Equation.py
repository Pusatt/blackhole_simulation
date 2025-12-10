import matplotlib.pyplot as plt

t = 0
dt = 0.01

x0 = 0
v0 = 5
ax = 9.8

xnow = x0
xlast = x0 #başlangıçta son konum değerimiz direkt x0 anı olur
dxlast = v0 #başlangıçta daha ivme etki etmediği için türevi başlangıç hızı olacak o da v0
dxnow = v0

t_list = list()
x_list = list()

def calculate_x(x0, v0, ax, t):
    x = x0 + v0*t + (1/2)*ax*t**2
    return x

def calculate_dx(xnow, xlast, dt):
    dx = (xnow - xlast)/dt
    return dx

def calculate_ddx(dxnow, dxlast, dt):
    ddx = (dxnow - dxlast)/dt
    return ddx

while xnow<2000:
    t_list.append(t)
    x_list.append(xnow)
    
    t += dt
    xnow = calculate_x(x0, v0, ax, t)
    dxnow = calculate_dx(xnow, xlast, dt)
    ddxnow = calculate_ddx(dxnow, dxlast, dt)
    xlast = xnow
    dxlast = dxnow
    print(xnow)

plt.figure(figsize=(10, 6))        # Grafik boyutunu ayarla (Opsiyonel)
plt.plot(t_list, x_list, label='Konum (x)', color='blue', linewidth=2) 

# Grafik Süslemeleri
plt.title('Sabit İvmeli Hareket Konum-Zaman (x-t) Grafiği')
plt.xlabel('Zaman (s)')
plt.ylabel('Konum (m)')
plt.grid(True)                     # Arkaya ızgara (kareli yapı) ekler
plt.legend()                       # "Konum (x)" etiketini gösterir
plt.show()
