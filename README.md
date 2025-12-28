Black Hole Simulation

EN
This project aims to simulate the general behavior and gravitational lensing around black holes using Python. The solutions to the black hole equations and their explanations can be found in the docs folder.
To generate a single-frame output using the CPU, run the main.py file located in the src directory. For higher resolution and more precise results, increase the width / height values (recommended: at least 400x200) and decrease the step_size value (recommended: at most 0.01).
For GPU acceleration, you can utilize NVIDIA's CUDA toolkit to perform parallel processing instead of serial processing, which significantly reduces render times. Sample images and video related to the project are located in the docs directory. The example image provided here was rendered using CUDA.

Project Owners: 
Cem Kutay NANÇIN 
Enes SÖKMEN 
Furkan KART 
Muhammed Pusat ÖZÇELİK 
Ömer Faruk KOLAYCA

****************************************************************

TR
Bu proje, karadeliklerin genel davranışını ve kütleçekimsel merceklenmeyi Python ortamında simüle etmeyi amaçlar. Karadelik denklemlerinin çözümleri ve açıklamaları docs klasöründe belirtilmiştir. CPU kullanarak tek frame çıktı almak için src klasöründeki main.py dosyasını çalıştırın. Daha yüksek çözünürlüklü ve daha hassas çözümler için width / height değerlerini arttırın (tavsiye edilen: en az 400x200) ve step_size değerini azaltın. (tavsiye edilen: en fazla 0.01)
GPU kullanımı için NVIDIA'nın sunduğu CUDA yazılımını kullanarak seri işlemler yerine paralel işlemler yapabilirsiniz ve render süresini ciddi manada düşürebilirsiniz. Proje ile ilgili örnek görseller ve video docs dizininde yer almaktadır. Tüm görseller CUDA kullanılarak render alınmıştır.

Proje sahipleri:
Cem Kutay NANÇIN
Enes SÖKMEN
Furkan KART
Muhammed Pusat ÖZÇELİK
Ömer Faruk KOLAYCA

## 🚀 How to Run? (CPU Version)

This project performs a black hole simulation using the Ray Tracing method on the processor (CPU) with Python.

### 📦 Required Libraries
Before running the project, ensure the following Python libraries are installed:

    pip install numpy pillow

### 1. Installation and Setup
Correct folder structure is critical for the project to run without errors:
1.  Download the **`src`** (source code) and **`assets`** (background images) folders to your computer.
2.  **Important:** Ensure that the `src` and `assets` folders are located side-by-side **within the same parent directory**. The program uses this structure to locate the space background.
   * Example Structure:
        * 📁 `My_Project/`
            * 📁 `src/`
            * 📁 `assets/`

### 2. Running
After installing the required libraries, navigate into the `src` folder via file explorer and run the **`main.py`** file.

✅ **Result:** Once the render is complete, the generated black hole image will **automatically open in your computer's default image viewer.**

### ⚙️ Configuration and Settings
To change simulation settings, open the `main.py` file with a text editor and modify the variables below:

* **`width` / `height`:** Resolution of the output image. (Higher values increase render time).
* **`fov`:** The camera's Field of View.
* **`step_size`:** Ray marching precision. (Smaller values result in a smoother and more accurate image but increase wait time).
* **`bg_index`:** Selects which background from the `assets` folder to use (e.g., `0` selects `bg0.jpg`).

---

### ⚠️ Important Notes

#### 🔸 About the Accretion Disk
In this version, the accretion disk around the black hole is **not included**. Since gravitational lensing calculations on the CPU already require significant processing power, adding the disk would extend render times to days.

#### ⚡ GPU Version (NVIDIA)
If you have an **NVIDIA** graphics card, we highly recommend using the GPU version to complete simulations in minutes, achieve much higher resolutions, and experience the full simulation including **Accretion Disk** effects:

## ⚡ GPU Acceleration & Installation Guide (NVIDIA)

Follow these steps to unlock the full potential of the simulation (Accretion Disk, Bloom Effect, High Resolution) and reduce render times from days to minutes using your GPU.

### 🛠️ 1. Prerequisites
To run this mode, your computer must have an **NVIDIA** graphics card.

1.  **Install CUDA Toolkit:**
    Ensure your graphics card drivers are up to date. Then, download and install the **CUDA Toolkit** version compatible with your operating system:
    👉 [Download NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

2.  **Install Anaconda:**
    Install Anaconda or Miniconda to manage the Python environment and packages:
    👉 [Download Anaconda](https://www.anaconda.com/download/success)

### 📦 2. Installing Libraries
Once the prerequisites are installed, you need to set up the required Python libraries.

1.  Open the **Anaconda Prompt** application (search for "Anaconda Prompt" in your Start menu).

    <img width="412" height="798" alt="image" src="https://github.com/user-attachments/assets/e6bcd5cc-8cd9-4011-b7d1-98bea66749c2" />
    
2.  Run the following command to install the core GPU computing libraries (type `y` and hit Enter if asked for confirmation):

    ```bash
    conda install numba cudatoolkit scipy
    ```
    <img width="1112" height="621" alt="image" src="https://github.com/user-attachments/assets/c532c4ef-3414-41e0-943e-3df722e92110" />

3.  Run the following command to install the image processing and utility libraries:

    ```bash
    pip install opencv-python pillow numpy
    ```

    <img width="1113" height="627" alt="image" src="https://github.com/user-attachments/assets/b0b8f361-dbc3-4406-9808-81bfaecaebed" />

### 📂 3. File Structure
For the simulation to run correctly, the folder structure is critical:

1.  Download the **`src_gpu`** (GPU source code) and **`assets`** (background images) folders from the repository.
2.  **Important:** Ensure that `src_gpu` and `assets` are located side-by-side **within the same parent directory**.
    * Example Structure:
        * 📁 `My_Project/`
            * 📁 `src_gpu/`
            * 📁 `assets/`

### 🚀 4. Running the Simulation

1. Open the Anaconda prompt by simply typing anaconda on your search bar and opening the application


2.  In **Anaconda Prompt**, use the `cd` command to navigate into the `src_gpu` folder. (Adjust the path below to match your computer):

    ```bash
    cd C:\Users\YourUsername\Desktop\My_Project\src_gpu
    ```

3.  To render a single high-quality frame, run:

    ```bash
    python render_frame.py
    ```

✅ **Result:** Once the process is complete, the render time will be displayed in the terminal, and the generated **`render_output.png`** file will appear in the same folder.
