import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from sympy.polys.numberfields.galois_resolvents import s_vars

from affine_transformation import warp_with_triangulation
from bg import bg_blur
from dayan import dayan
from meibai import segment_face, apply_whitening_and_blend
from shoulian import shoulian
from test import smooth_and_sharpen_skin
from test_speed import enhance_face_detail

mp_face_mesh = mp.solutions.face_mesh

DAYAN_DEFAULT = 1
SHOULIAN_DEFAULT = 1
H_DEFAULT = 0
S_DEFAULT = 0
V_DEFAULT = 0
MOPI_DEFAULT = 0
bg_blur_enabled = False

def get_points_all(image, landmarks):
    points = [(landmarks[i].x, landmarks[i].y) for i in range(len(landmarks))]
    points_pixels = [(int(p[0] * image.shape[1]), int(p[1] * image.shape[0])) for p in points]
    return points_pixels

def process_image(image, DAYAN, SHOULIAN, H, S, V, MOPI, DETAIL,apply_bg_blur=False):
    if image is None:
        print("Error: Input image is None.")
        return None

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            print("No face landmarks detected.")
            return image

        src_points = []
        dst_points = []

        for face_landmarks in results.multi_face_landmarks:
            all_point = get_points_all(image, face_landmarks.landmark)
            lower_hsv, upper_hsv = segment_face(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), all_point)
            dayan(image, face_landmarks, DAYAN, src_points, dst_points)
            shoulian(image, face_landmarks, SHOULIAN, src_points, dst_points)
            processed_image = warp_with_triangulation(image, src_points, dst_points)
            if processed_image is None:
                print("Error: Warp with triangulation failed.")
                return image

            processed_image = apply_whitening_and_blend(
                processed_image,
                cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV),
                lower_hsv,
                upper_hsv,
                H, S, V, MOPI
            )
            processed_image = enhance_face_detail(processed_image, strength_pct=DETAIL)
            if apply_bg_blur:
                processed_image = bg_blur(processed_image)

        return processed_image


def update_image():
    global image_path, bg_blur_enabled
    image = cv2.imread(image_path)
    if image is None:
        error_label.config(text=f"Error: Could not load image at {image_path}")
        return
    processed_image = process_image(
        image,
        DAYAN_var.get(),
        SHOULIAN_var.get(),
        H_DEFAULT,
        S_var.get(),
        V_var.get(),
        MOPI_var.get(),
        DETAIL_var.get(),
        apply_bg_blur=bg_blur_enabled
    )
    if processed_image is None:
        error_label.config(text="Error: Image processing failed.")
        return

    display_image = processed_image.copy()
    # max_size = (500, 500)
    # h, w = display_image.shape[:2]
    # scale = min(max_size[0] / w, max_size[1] / h)
    # if scale < 1:
    #     new_w, new_h = int(w * scale), int(h * scale)
    #     display_image = cv2.resize(display_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
    display_image_pil = Image.fromarray(display_image_rgb)
    display_image_tk = ImageTk.PhotoImage(display_image_pil)

    image_label.config(image=display_image_tk)
    image_label.image = display_image_tk
    error_label.config(text="")

    # Update button text
    blur_button.config(text="取消模糊背景" if bg_blur_enabled else "模糊背景")


# === Toggle Background Blur ===
def toggle_bg_blur():
    global bg_blur_enabled
    bg_blur_enabled = not bg_blur_enabled
    update_image()


def open_image():
    global image_path
    new_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if new_image_path:
        image_path = new_image_path
        update_image()


def save_image():
    global image_path, bg_blur_enabled
    image = cv2.imread(image_path)
    if image is None:
        error_label.config(text=f"Error: Could not load image at {image_path}")
        return

    processed_image = process_image(
        image,
        DAYAN_var.get(),
        SHOULIAN_var.get(),
        H_DEFAULT,
        S_var.get(),
        V_var.get(),
        MOPI_var.get(),
        apply_bg_blur=bg_blur_enabled
    )
    if processed_image is None:
        error_label.config(text="Error: Image processing failed.")
        return

    base_name = image_path.split('/')[-1].split('.')[0]
    blur_suffix = "_blur" if bg_blur_enabled else ""
    filename = f"{base_name}_D{DAYAN_var.get():.2f}_S{SHOULIAN_var.get():.2f}_S{S_var.get():.2f}_V{V_var.get():.2f}_M{MOPI_var.get():.1f}{blur_suffix}.jpg"

    cv2.imwrite(filename, processed_image)
    error_label.config(text=f"Image saved as {filename}")


root = tk.Tk()
root.title("美颜相机 + 模糊背景")

image_path = 'mopi.jpg'
image_frame = tk.Frame(root)
image_frame.pack(side=tk.LEFT, padx=10, pady=10)
image_label = tk.Label(image_frame)
image_label.pack()
error_label = tk.Label(image_frame, text="", fg="red")
error_label.pack()

control_frame = tk.Frame(root)
control_frame.pack(side=tk.RIGHT, padx=10, pady=10)

DAYAN_var = tk.DoubleVar(value=DAYAN_DEFAULT)
SHOULIAN_var = tk.DoubleVar(value=SHOULIAN_DEFAULT)
DETAIL_var = tk.DoubleVar(value=0)
# H_var = tk.DoubleVar(value=H_DEFAULT)
S_var = tk.DoubleVar(value=S_DEFAULT)
V_var = tk.DoubleVar(value=V_DEFAULT)
MOPI_var = tk.DoubleVar(value=MOPI_DEFAULT)

tk.Scale(control_frame, from_=1.0, to=1.2, resolution=0.01, orient=tk.HORIZONTAL, label="大眼", variable=DAYAN_var,
         command=lambda x: update_image()).pack(padx=5, pady=5)
tk.Scale(control_frame, from_=0.85, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, label="瘦脸", variable=SHOULIAN_var,
         command=lambda x: update_image()).pack(padx=5, pady=5)
# tk.Scale(control_frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, label="H", variable=H_var,
#          command=lambda x: update_image()).pack(padx=5, pady=5)
text = tk.StringVar(value="调整S，V来调整美白和冷暖\nS不宜超过0.20，V不宜超过0.10")
tk.Label(control_frame, textvariable=text, font=("Arial", 9)).pack(padx=5)
tk.Scale(control_frame, from_=0.0, to=0.4, resolution=0.01, orient=tk.HORIZONTAL, label="S", variable=S_var,
         command=lambda x: update_image()).pack(padx=5, pady=5)
tk.Scale(control_frame, from_=0.0, to=0.2, resolution=0.01, orient=tk.HORIZONTAL, label="V", variable=V_var,
         command=lambda x: update_image()).pack(padx=5, pady=5)
text = tk.StringVar(value="根据美图秀秀，其磨皮拉满约等于这\n里数值0.30，因此不宜超过0.30")
tk.Label(control_frame, textvariable=text, font=("Arial", 9)).pack(padx=5)
tk.Scale(control_frame, from_=0, to=0.6, resolution=0.01, orient=tk.HORIZONTAL, label="磨皮", variable=MOPI_var,
         command=lambda x: update_image()).pack(padx=5, pady=5)
tk.Scale(control_frame,
         from_=0, to=100, resolution=1,
         orient=tk.HORIZONTAL,
         label="细节增强",
         variable=DETAIL_var,
         command=lambda x: update_image()).pack(padx=5, pady=5)

open_button = tk.Button(control_frame, text="打开图片", command=open_image)
open_button.pack(pady=5)

blur_button = tk.Button(control_frame, text="模糊背景", command=toggle_bg_blur, bg="#4CAF50", fg="white",
                        font=("Arial", 10, "bold"))
blur_button.pack(pady=5, fill=tk.X)

save_button = tk.Button(control_frame, text="保存图片", command=save_image)
save_button.pack(pady=5)

update_image()
root.mainloop()