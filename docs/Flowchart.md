# Proje Akış Şeması (Flowchart)

```mermaid
graph TD
    Start([Başlangıç: main.py]) --> Init[Scene, BlackHole, Background ve Camera Nesnelerini Oluştur]
    Init --> Canvas[Boş Görüntü Matrisi Oluştur H x W]
    
    subgraph Render_Loop [Render Döngüsü]
        CheckPixel{"Tüm Pikseller<br/>İşlendi mi?"}
        CheckPixel -- Hayır --> GetCoord["Sıradaki Piksel Koordinatını Al (h, w)"]
        GetCoord --> RayDir["Işın Yönünü Hesapla<br/>(spherical_to_cartesian)"]
        RayDir --> InitRay["Işın Başlangıç Değerlerini Ata<br/>(t, r, theta, phi)"]
        
        subgraph Ray_Marching [Fizik Döngüsü / Ray Tracing]
            CheckSteps{"Adım Sayısı < Max?"}
            CheckSteps -- Hayır (Zaman Aşımı) --> ErrColor[Hata Rengi Döndür Kırmızı]
            CheckSteps -- Evet --> ColSphere{"Küre ile<br/>Çarpışma Var mı?"}
            
            ColSphere -- Evet --> RetSphere[Küre Rengini Döndür]
            ColSphere -- Hayır --> ColEH{"Olay Ufkuna<br/>(Event Horizon)<br/>Girdi mi?"}
            
            ColEH -- Evet (r <= rs) --> RetBlack["Siyah Renk Döndür (0,0,0)"]
            ColEH -- Hayır --> ColInf{"Sonsuza Gitti mi?<br/>(r > 50)"}
            
            ColInf -- Evet --> MapBG["Arka Plan Rengini Hesapla<br/>(shaders.py: get_background_color)"]
            MapBG --> RetBG[Arka Plan Rengini Döndür]
            
            ColInf -- Hayır --> RK4["RK4 Entegrasyonu ile<br/>Yeni Konumu Hesapla"]
            RK4 --> Geodesic["Geodesic Denklemleri<br/>(Işığın Bükülmesi)"]
            Geodesic --> UpdateRay["Işın Verilerini Güncelle<br/>(Konum ve Türevler)"]
            UpdateRay --> StepUp[Adım Sayısını Artır]
            StepUp --> CheckSteps
        end
        
        RetSphere --> SetPixel[Piksel Rengini Kaydet]
        RetBlack --> SetPixel
        RetBG --> SetPixel
        ErrColor --> SetPixel
        SetPixel --> CheckPixel
    end
    
    CheckPixel -- Evet --> ShowImg["Görüntüyü Oluştur (PIL)"]
    ShowImg --> Finish([Bitiş: Resmi Göster])

    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style Finish fill:#f9f,stroke:#333,stroke-width:2px
    style Ray_Marching fill:#e1f5fe,stroke:#01579b,stroke-width:2px,stroke-dasharray: 5 5
    style RK4 fill:#fff9c4,stroke:#fbc02d
    style Geodesic fill:#fff9c4,stroke:#fbc02d
