import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from affine_transformation import warp_with_triangulation
from anti_faling import nasolabial_folds_filter
from anti_black import black_filter
from bg import bg_blur
from dayan import dayan
from dazui import dazui
from forehead import forehead
from long_nose import longnose
from meibai import apply_whitening_and_blend
from renzhong import renzhong
from shoulian import shoulian
from quangu import quangu
from zhailian import zhailian
from biyi import biyi
from res import enhance_face_detail
from white_eye import white_eyes
from white_teeth import white_teeth
from lipstick import lipstick as apply_lipstick
from blush import apply_blush
from zuijiao import zuijiao

mp_face_mesh = mp.solutions.face_mesh
EYE_INDICES = [374, 380, 390, 373, 249, 385, 384, 263, 466, 387, 386, 381, 382, 398, 388, 362, 154, 155, 33, 7, 246, 161, 159, 158, 144, 145, 173, 133, 157, 163, 153, 160]
# BOUNDARY = [9,70,111,122,351,293,340]
# 355,358,327,289,392,309,459,458,250,290 鼻子边缘，如需要从4开始粘贴
BOUNDARY = [270, 409, 317, 402, 81, 82, 91, 181, 37, 0, 84, 17, 269, 321, 375, 318, 324, 312, 311, 415, 308, 314, 61,
            146, 78, 95, 267, 13, 405, 178, 87, 185, 14, 88, 40, 291, 191, 310, 39, 80, #LIPS AND TEETH

            4, #NOSETIP

            334, 296, 276, 283, 293, 295, 285,336, 282, 300,46, 53, 66, 107, 52, 65, 63, 105, 70, 55, #EYEBROWS

            374, 380,390, 373, 249, 385, 384, 263, 466, 387,386, 381, 382, 398, 388, 362, 154, 155, 33, 7, 246, 161,
            159, 158, 144, 145, 173, 133, 157, 163, 153, 160, #LEFT RIGHT EYE

            132,58,172, 136, 150, 149, 176, 148,361,288,397, 365, 379, 378, 400,377,152, #JAWLINE

            473,468, #IRIS

            116,123,345,352,103,67, #LEFT RIGHT QUANGU

            109,10,338,297,332,126,129,98,75,166,79,239,238,20,60,355,358,327,289,392,309,459,458,250,290,97,2,327,326,
            48,115,220,45,275,440,344,278,280,50,389,9,162]
#BOUNDARY = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
# 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,172, 58, 132, 93,234, 127, 162, 21, 54, 103, 67, 109, 10]
DAYAN_DEFAULT = 1
SHOULIAN_DEFAULT = 1
QUANGU_DEFAULT = 1
ZHAILIAN_DEFAULT = 1
RENZHONG_DEFAILT = 0
BIYI_DEFAULT = 1
FOREHEAD_DEFAULT = 0
H_DEFAULT = 0
S_DEFAULT = 0
V_DEFAULT = 0
MOPI_DEFAULT = 0
FALING_DEFAULT = 0
BLACK_DEFAULT = 0
WHITE_EYE_DEFAULT = 1
WHITE_TEETH_DEFAULT = 1
LIPSTICK_DEFAULT = 0.0
BLUSH_DEFAULT = 0.0
LONGNOSE_DEFAULT = 0.0
ZUIJIAO_DEFAULT = 0.0
DAZUI_DEFAULT = 1
bg_blur_enabled = False

def get_points_all(image, landmarks):
    points = [(landmarks[i].x, landmarks[i].y, landmarks[i].z+1) for i in range(len(landmarks))]
    points_pixels = [transform_to_3d(int(p[0] * image.shape[1]), int(p[1] * image.shape[0]), p[2]) for p in points]
    return points_pixels
def transform_to_3d(x, y, z):
    return (x * z, y * z, z)
def get_points_2D(image, landmarks):
    points = [(landmarks[i].x, landmarks[i].y) for i in range(len(landmarks))]
    points_pixels = [(int(p[0] * image.shape[1]), int(p[1] * image.shape[0])) for p in points]
    return points_pixels

def process_image(image, DAYAN, SHOULIAN, QUANGU, BIYI, LONGNOSE, RENZHONG, ZHAILIAN, FOREHEAD,ZUIJIAO, DAZUI,
                  H, S, V, MOPI, DETAIL, FALING, BLACK, WHITE_EYE, WHITE_TEETH, LIPSTICK, BLUSH, apply_bg_blur=False):
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

        for face_landmarks in results.multi_face_landmarks:
            all_point = get_points_all(image, face_landmarks.landmark)
            all_2d = get_points_2D(image, face_landmarks.landmark)
            dst_points = {}
            src_points = {}
            for pts in BOUNDARY:
                pt = all_point[pts]
                src_points[pts] = pt
                dst_points[pts] = pt
            # print(dst_points)
            dayan(DAYAN, dst_points)
            shoulian(SHOULIAN, dst_points)
            quangu(QUANGU, dst_points)
            biyi(BIYI, dst_points)
            longnose(LONGNOSE, dst_points)
            zhailian(ZHAILIAN, dst_points)
            renzhong(RENZHONG, dst_points)
            forehead(FOREHEAD, dst_points)
            zuijiao(ZUIJIAO, dst_points)
            dazui(DAZUI, dst_points)
            processed_image = warp_with_triangulation(image, src_points, dst_points)
            # print(dst_points)


            if processed_image is None:
                print("Error: Warp with triangulation failed.")
                return image

            processed_image = apply_whitening_and_blend(
                processed_image,
                all_2d,
                cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV),
                H, S, V, MOPI,

            )
            processed_image = enhance_face_detail(processed_image, strength_pct=DETAIL)
            processed_image = nasolabial_folds_filter(processed_image,all_2d, FALING)
            processed_image = black_filter(processed_image,all_2d, BLACK)
            processed_image = white_eyes(processed_image, all_2d,WHITE_EYE)
            processed_image = white_teeth(processed_image,all_2d, WHITE_TEETH)
            processed_image = apply_lipstick(processed_image, all_2d, LIPSTICK)
            processed_image = apply_blush(processed_image,all_2d, color=(157, 107, 255), intensity=BLUSH)

            if apply_bg_blur:
                processed_image = bg_blur(processed_image)

        return processed_image


def update_image():
    global image_path, bg_blur_enabled
    image = cv2.imread(image_path)
    if image is None:
        error_label.config(text=f"Error: Could not load image at {image_path}")
        return


    h, w = image.shape[:2]
    # print(h,w)
    # image = cv2.resize(image, (int(w*0.8), int(h*0.8)))
    if w >= 1920 or h >= 1080:
        image = cv2.resize(image, (w//2, h//2), interpolation=cv2.INTER_AREA)
    processed_image = process_image(
        image,
        DAYAN_var.get(),
        SHOULIAN_var.get(),
        QUANGU_var.get(),
        BIYI_var.get(),
        LONGNOSE_var.get(),
        RENZHONG_var.get(),
        ZHAILIAN_var.get(),
        FOREHEAD_var.get(),
        ZUIJIAO_var.get(),
        DAZUI_var.get(),
        H_DEFAULT,
        S_var.get(),
        V_var.get(),
        MOPI_var.get(),
        DETAIL_var.get(),
        FALING_var.get(),
        BLACK_var.get(),
        WHITE_EYE_var.get(),
        WHITE_TEETH_var.get(),
        LIPSTICK_var.get(),
        BLUSH_var.get(),
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
        QUANGU_var.get(),
        BIYI_var.get(),
        RENZHONG_var.get(),
        ZHAILIAN_var.get(),
        FOREHEAD_var.get(),
        H_DEFAULT,
        S_var.get(),
        V_var.get(),
        MOPI_var.get(),
        DETAIL_var.get(),
        FALING_var.get(),
        BLACK_var.get(),
        WHITE_EYE_var.get(),
        WHITE_TEETH_var.get(),
        LIPSTICK_var.get(),
        BLUSH_var.get(),
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
root.title("美颜算法")

image_path = 'imgs/face4.jpg'
image_frame = tk.Frame(root)
image_frame.pack(side=tk.LEFT, padx=10, pady=10)
image_label = tk.Label(image_frame)
image_label.pack()
error_label = tk.Label(image_frame, text="", fg="red")
error_label.pack()

control_frame = tk.Frame(root)
control_frame.pack(side=tk.RIGHT, padx=10, pady=10)
control_frame.grid_columnconfigure(0, weight=1)
control_frame.grid_columnconfigure(1, weight=1)
control_frame.grid_columnconfigure(2, weight=1)

DAYAN_var = tk.DoubleVar(value=DAYAN_DEFAULT)
SHOULIAN_var = tk.DoubleVar(value=SHOULIAN_DEFAULT)
ZHAILIAN_var = tk.DoubleVar(value=ZHAILIAN_DEFAULT)
FOREHEAD_var = tk.DoubleVar(value=FOREHEAD_DEFAULT)
RENZHONG_var = tk.DoubleVar(value=RENZHONG_DEFAILT)
QUANGU_var = tk.DoubleVar(value=QUANGU_DEFAULT)
BIYI_var = tk.DoubleVar(value=BIYI_DEFAULT)
LONGNOSE_var = tk.DoubleVar(value=LONGNOSE_DEFAULT)
DETAIL_var = tk.DoubleVar(value=0)
# H_var = tk.DoubleVar(value=H_DEFAULT)
S_var = tk.DoubleVar(value=S_DEFAULT)
V_var = tk.DoubleVar(value=V_DEFAULT)
MOPI_var = tk.DoubleVar(value=MOPI_DEFAULT)
FALING_var = tk.DoubleVar(value=FALING_DEFAULT)
BLACK_var = tk.DoubleVar(value=BLACK_DEFAULT)
WHITE_EYE_var = tk.DoubleVar(value=WHITE_EYE_DEFAULT)
WHITE_TEETH_var = tk.DoubleVar(value=WHITE_TEETH_DEFAULT)
LIPSTICK_var = tk.DoubleVar(value=LIPSTICK_DEFAULT)
BLUSH_var = tk.DoubleVar(value=BLUSH_DEFAULT)
ZUIJIAO_var = tk.DoubleVar(value=ZUIJIAO_DEFAULT)
DAZUI_var = tk.DoubleVar(value=DAZUI_DEFAULT)


tk.Scale(control_frame, from_=1.0, to=1.1, resolution=0.01, orient=tk.HORIZONTAL, label="大眼", variable=DAYAN_var,
         command=lambda x: update_image()).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
tk.Scale(control_frame, from_=0.95, to=1.0, resolution=0.005, orient=tk.HORIZONTAL, label="瘦脸", variable=SHOULIAN_var,
         command=lambda x: update_image()).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
tk.Scale(control_frame, from_=0.95, to=1.0, resolution=0.005, orient=tk.HORIZONTAL, label="颧骨", variable=QUANGU_var,
         command=lambda x: update_image()).grid(row=0, column=2, padx=5, pady=5, sticky="ew")
tk.Scale(control_frame, from_=-3.0, to=3.0, resolution=0.005, orient=tk.HORIZONTAL, label="人中", variable=RENZHONG_var,
         command=lambda x: update_image()).grid(row=1, column=2, padx=5, pady=5, sticky="ew")
tk.Scale(control_frame, from_=-5.0, to=5.0, resolution=0.005, orient=tk.HORIZONTAL, label="额头", variable=FOREHEAD_var,
         command=lambda x: update_image()).grid(row=2, column=2, padx=5, pady=5, sticky="ew")
tk.Scale(control_frame, from_=0.95, to=1.0, resolution=0.005, orient=tk.HORIZONTAL, label="窄脸", variable=ZHAILIAN_var,
         command=lambda x: update_image()).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
tk.Scale(control_frame, from_=0.9, to=1.1, resolution=0.005, orient=tk.HORIZONTAL, label="鼻翼", variable=BIYI_var,
         command=lambda x: update_image()).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
tk.Scale(control_frame, from_=-2.0, to=2.0, resolution=0.005, orient=tk.HORIZONTAL, label="长鼻", variable=LONGNOSE_var,
         command=lambda x: update_image()).grid(row=2, column=1, padx=5, pady=5, sticky="ew")
tk.Scale(control_frame, from_=0.0, to=3.0, resolution=0.005, orient=tk.HORIZONTAL, label="嘴角", variable=ZUIJIAO_var,
         command=lambda x: update_image()).grid(row=2, column=0, padx=5, pady=5, sticky="ew")
tk.Scale(control_frame, from_=0.9, to=1.1, resolution=0.005, orient=tk.HORIZONTAL, label="大嘴", variable=DAZUI_var,
         command=lambda x: update_image()).grid(row=3, column=2, padx=5, pady=5, sticky="ew")
# tk.Scale(control_frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, label="H", variable=H_var,
#          command=lambda x: update_image()).pack(padx=5, pady=5)
# text = tk.StringVar(value="调整S，V来调整美白和冷暖\nS不宜超过0.20，V不宜超过0.10")
# tk.Label(control_frame, textvariable=text, font=("Arial", 9)).grid(row=2, column=0, columnspan=2, padx=5, sticky="w")
tk.Scale(control_frame, from_=0.0, to=0.08, resolution=0.002, orient=tk.HORIZONTAL, label="S", variable=S_var,
         command=lambda x: update_image()).grid(row=3, column=0, padx=5, pady=5, sticky="ew")
tk.Scale(control_frame, from_=0.0, to=0.03, resolution=0.002, orient=tk.HORIZONTAL, label="V", variable=V_var,
         command=lambda x: update_image()).grid(row=3, column=1, padx=5, pady=5, sticky="ew")
# text = tk.StringVar(value="根据美图秀秀，其磨皮拉满约等于这\n里数值0.30，因此不宜超过0.30")
# tk.Label(control_frame, textvariable=text, font=("Arial", 9)).grid(row=4, column=0, columnspan=2, padx=5, sticky="w")
tk.Scale(control_frame, from_=0, to=0.5, resolution=0.01, orient=tk.HORIZONTAL, label="磨皮", variable=MOPI_var,
         command=lambda x: update_image()).grid(row=5, column=0, padx=5, pady=5, sticky="ew")
tk.Scale(control_frame,
         from_=0, to=50, resolution=1,
         orient=tk.HORIZONTAL,
         label="细节增强",
         variable=DETAIL_var,
         command=lambda x: update_image()).grid(row=5, column=1, padx=5, pady=5, sticky="ew")
tk.Scale(control_frame,
         from_=0, to=1, resolution=0.02,
         orient=tk.HORIZONTAL,
         label="去法令纹",
         variable=FALING_var,
         command=lambda x: update_image()).grid(row=6, column=0, padx=5, pady=5, sticky="ew")
tk.Scale(control_frame,
         from_=0, to=1, resolution=0.02,
         orient=tk.HORIZONTAL,
         label="去黑眼圈",
         variable=BLACK_var,
         command=lambda x: update_image()).grid(row=6, column=1, padx=5, pady=5, sticky="ew")
tk.Scale(control_frame,
         from_=1, to=1.2, resolution=0.02,
         orient=tk.HORIZONTAL,
         label="亮眼",
         variable=WHITE_EYE_var,
         command=lambda x: update_image()).grid(row=7, column=0, padx=5, pady=5, sticky="ew")
tk.Scale(control_frame,
         from_=1, to=1.2, resolution=0.01,
         orient=tk.HORIZONTAL,
         label="亮牙",
         variable=WHITE_TEETH_var,
         command=lambda x: update_image()).grid(row=7, column=1, padx=5, pady=5, sticky="ew")

tk.Scale(control_frame,
         from_=0.0, to=0.3, resolution=0.01,
         orient=tk.HORIZONTAL,
         label="口红",
         variable=LIPSTICK_var,
         command=lambda x: update_image()).grid(row=8, column=0, padx=5, pady=5, sticky="ew")

tk.Scale(control_frame,
         from_=0.0, to=1, resolution=0.05,
         orient=tk.HORIZONTAL,
         label="腮红",
         variable=BLUSH_var,
         command=lambda x: update_image()).grid(row=8, column=1, padx=5, pady=5, sticky="ew")

open_button = tk.Button(control_frame, text="打开图片", command=open_image)
open_button.grid(row=9, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

blur_button = tk.Button(control_frame, text="模糊背景", command=toggle_bg_blur, bg="#4CAF50", fg="white",
                        font=("Arial", 10, "bold"))
blur_button.grid(row=10, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

save_button = tk.Button(control_frame, text="保存图片", command=save_image)
save_button.grid(row=11, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

update_image()
root.mainloop()
