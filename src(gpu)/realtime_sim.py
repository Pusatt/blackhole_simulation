import numpy as np
import cv2
import time
from gpu_renderer import render_gpu
from camera import Camera


def run_realtime(scene):
    w, h = 160, 90

    cam = Camera(
        scene,
        w, h,
        fov=80,
        dλ=0.01,
        maxλ=30
    )

    angle = 0.0
    radius = 6.0

    print("FPS simülatör başlıyor (ESC ile çık)")

    last_time = time.time()

    while True:
        # --- Kamera orbit hareketi ---
        cam.real_position = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            2.0
        ])
        cam.relative_position = cam._update_relative_position()
        cam.direction = cam.find_camera_direction()

        angle += 0.01  # dönüş hızı

        # --- Render ---
        image = render_gpu(scene, cam)

        # --- FPS hesabı ---
        now = time.time()
        fps = 1.0 / (now - last_time)
        last_time = now

        img = (image * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.putText(
            img,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Black Hole Simulator", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
