import platform
import sys
import time
import torch
import shutil  # 对文件的一些操作

import core.globals
import glob
import argparse
import multiprocessing as mp
import os
from pathlib import Path  # 和路径有关
import tkinter as tk  # 窗口GUI设计
from tkinter import filedialog
from tkinter.filedialog import asksaveasfilename
from core.processor import process_video, process_img
from core.utils import is_img, detect_fps, set_fps, create_video, add_audio, extract_frames, rreplace
from core.config import get_face
import webbrowser
import psutil
import cv2
import threading
from PIL import Image, ImageTk

pool = None  # 用来设置进程池
args = {}

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--face', help='use this face', dest='source_img')
parser.add_argument('-t', '--target', help='replace this face', dest='target_path')
parser.add_argument('-o', '--output', help='save output to this file', dest='output_file')
parser.add_argument('--gpu', help='use gpu', dest='gpu', action='store_true', default=False)
parser.add_argument('--keep-fps', help='maintain original fps', dest='keep_fps', action='store_true', default=False)
parser.add_argument('--keep-frames', help='keep frames directory', dest='keep_frames', action='store_true', default=False)
parser.add_argument('--max-memory', help='set max memory', default=4, type=int)
parser.add_argument('--max-cores', help='number of cores to use', dest='cores_count', type=int, default=max(psutil.cpu_count() - 2, 2))

for name, value in vars(parser.parse_args()).items():
    args[name] = value

sep = "/"
# 判断目前正在使用的平台  Windows 返回 'nt' Linux/mac 返回'posix'
if os.name == "nt":
    sep = "\\"


# 内存配置
def limit_resources():
    if args['max_memory'] >= 1:
        memory = args['max_memory'] * 1024 * 1024 * 1024
        if str(platform.system()).lower() == 'windows':  # 判断是否为windows操作系统
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:  # 其他操作系统
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


# 程序运行前的一些环境检查
def pre_check():
    # python需要>=3.8
    # if sys.version_info < (3, 8):
    #     quit(f'Python version is not supported - please upgrade to 3.8 or higher')
    # 判断ffmpeg是否安装
    if not shutil.which('ffmpeg'):
        quit('ffmpeg is not installed!')
    # 预权重路径
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'inswapper_128.onnx')
    # 判断权重是否存在
    if not os.path.isfile(model_path):
        quit('File "inswapper_128.onnx" does not exist!')
    # 是否使用GPU
    if '--gpu' in sys.argv:
        print(sys.argv)
        CUDA_VERSION = torch.version.cuda  # 获取cuda版本
        CUDNN_VERSION = torch.backends.cudnn.version()  # cudnn版本
        NVIDIA_PROVIDERS = ['CUDAExecutionProvider', 'TensorrtExecutionProvider']
        if len(list(set(core.globals.providers) - set(NVIDIA_PROVIDERS))) > 1:
            # 判断是否支持gpu
            if not torch.cuda.is_available() or not CUDA_VERSION:
                quit("You are using --gpu flag but CUDA isn't available or properly installed on your system.")
            # cuda要求 11.4≤ cuda ≤11.8
            # if CUDA_VERSION > '11.8':
            #     quit(f"CUDA version {CUDA_VERSION} is not supported - please downgrade to 11.8.")
            # if CUDA_VERSION < '11.4':
            #     quit(f"CUDA version {CUDA_VERSION} is not supported - please upgrade to 11.8")
            # # cudnn要求 8.2.2≤ cuda ≤ 8.9.1
            # if CUDNN_VERSION < 8220:
            #     quit(f"CUDNN version {CUDNN_VERSION} is not supported - please upgrade to 8.9.1")
            # if CUDNN_VERSION > 8910:
            #     quit(f"CUDNN version {CUDNN_VERSION} is not supported - please downgrade to 8.9.1")
            core.globals.providers = ['CUDAExecutionProvider']
    # cpu运行
    else:
        core.globals.providers = ['CPUExecutionProvider']


# 开始进程 视频人脸交换
def start_processing():
    # 记录进程启动时间
    start_time = time.time()
    # 当为gpu运行时
    if args['gpu']:
        print('gpu 推理')
        # source_img:图像路径，frame_paths:图像序列路径。
        # frame_paths是目标人脸，将source_img替换到frame_paths人脸
        process_video(args['source_img'], args["frame_paths"])
        # 记录结束时间
        end_time = time.time()
        print(flush=True)
        # 打印人脸检测进程运行时间
        print(f"Processing time: {end_time - start_time:.2f} seconds", flush=True)
        return
    # 获取图像序列路径
    frame_paths = args["frame_paths"]
    n = len(frame_paths)//(args['cores_count'])
    processes = []
    for i in range(0, len(frame_paths), n):
        # pool.apply_async异步非阻塞 不用等待当前进程执行完毕，随时根据系统调度来进行进程切换
        p = pool.apply_async(process_video, args=(args['source_img'], frame_paths[i:i+n],))
        processes.append(p)
    for p in processes:
        p.get()
    # 进程关闭
    pool.close()
    # 加入主进程
    pool.join()
    end_time = time.time()
    print(flush=True)
    print(f"Processing time: {end_time - start_time:.2f} seconds", flush=True)


# 图像的预处理
def preview_image(image_path):
    img = Image.open(image_path)
    # 图像大小的resize
    img = img.resize((180, 180), Image.ANTIALIAS)
    photo_img = ImageTk.PhotoImage(img)
    # GUI显示
    left_frame = tk.Frame(window)
    left_frame.place(x=60, y=100)
    img_label = tk.Label(left_frame, image=photo_img)
    img_label.image = photo_img
    img_label.pack()


# 视频的预处理
def preview_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # opencv2PIL
        img = Image.fromarray(frame)
        img = img.resize((180, 180), Image.ANTIALIAS)
        # 仅需读取一帧并在GUI上显示即可
        photo_img = ImageTk.PhotoImage(img)
        right_frame = tk.Frame(window)
        right_frame.place(x=360, y=100)
        img_label = tk.Label(right_frame, image=photo_img)
        img_label.image = photo_img
        img_label.pack()

    cap.release()


# 选择source人脸
def select_face():
    args['source_img'] = filedialog.askopenfilename(title="Select a face")  # 选择打开什么文件，返回的是文件名
    # 对source人脸图像进行预处理
    preview_image(args['source_img'])


# 选择target人脸
def select_target():
    args['target_path'] = filedialog.askopenfilename(title="Select a target")  # 获得target人脸文件名
    # 对视频进行预处理并开启线程
    threading.Thread(target=preview_video, args=(args['target_path'],)).start()


# 保持FPS
def toggle_fps_limit():
    args['keep_fps'] = limit_fps.get() != True


# 保持frames
def toggle_keep_frames():
    args['keep_frames'] = keep_frames.get() != True


# 保存文件
def save_file():
    filename, ext = 'output.mp4', '.mp4'
    # 判断目标文件名是否为图片，"png", "jpg", "jpeg", "bmp"，否则保存为视频格式
    if is_img(args['target_path']):
        # 如果是图片则保存为图片格式
        filename, ext = 'output.png', '.png'
    args['output_file'] = asksaveasfilename(initialfile=filename, defaultextension=ext, filetypes=[("All Files","*.*"),("Videos","*.mp4")])


def status(string):
    if 'cli_mode' in args:
        print("Status: " + string)
    else:
        status_label["text"] = "Status: " + string
        window.update()


def start():
    print("DON'T WORRY. IT'S NOT STUCK/CRASHED.\n" * 5)  # 这并不是卡住了，而是在等待！
    # 判断输入图像路径是否正确
    if not args['source_img'] or not os.path.isfile(args['source_img']):
        print("\n[WARNING] Please select an image containing a face.")
        return
    # # 判断输入图像路径是否正确
    elif not args['target_path'] or not os.path.isfile(args['target_path']):
        print("\n[WARNING] Please select a video/image to swap face in.")
        return
    if not args['output_file']:
        args['output_file'] = rreplace(args['target_path'], "/", "/swapped-", 1) if "/" in target_path else "swapped-"+target_path
    global pool
    # 进程池
    pool = mp.Pool(args['cores_count'])
    target_path = args['target_path']
    # 人脸检测
    test_face = get_face(cv2.imread(args['source_img']))
    if not test_face:  # 如果没有检测到人脸
        print("\n[WARNING] No face detected in source image. Please try with another one.\n")
        return
    # 判断是否为图像，"png", "jpg", "jpeg", "bmp"
    if is_img(target_path):
        # 人脸交换
        process_img(args['source_img'], target_path, args['output_file'])
        status("swap successful!")
        return

    # 视频
    video_name = os.path.basename(target_path)  # 读取视频名字，xx.mp4，eg:test.mp4
    video_name = os.path.splitext(video_name)[0]  # 去除后缀  eg:test
    output_dir = os.path.join(os.path.dirname(target_path), video_name)  # 输出路径
    Path(output_dir).mkdir(exist_ok=True)

    status("detecting video's FPS...")
    fps = detect_fps(target_path)  # 检测帧率
    # 如果keep_fps=False，并且 fps>30，则进行FPS处理
    if not args['keep_fps'] and fps > 30:
        this_path = output_dir + "/" + video_name + ".mp4"
        set_fps(target_path, this_path, 30)
        target_path, fps = this_path, 30  # 获取视频路径和帧率
    else:
        shutil.copy(target_path, output_dir)
    status("extracting frames...")
    #
    extract_frames(target_path, output_dir)
    args['frame_paths'] = tuple(sorted(
        glob.glob(output_dir + f"/*.png"),
        key=lambda x: int(x.split(sep)[-1].replace(".png", ""))
    ))
    status("swapping in progress...")
    start_processing()
    status("creating video...")
    create_video(video_name, fps, output_dir)
    status("adding audio...")
    add_audio(output_dir, target_path, args['keep_frames'], args['output_file'])
    save_path = args['output_file'] if args['output_file'] else output_dir + "/" + video_name + ".mp4"
    print("\n\nVideo saved as:", save_path, "\n\n")
    status("swap successful!")


if __name__ == "__main__":
    global status_label, window

    # 预权重，python环境的检查
    pre_check()
    # 内存检测
    limit_resources()
    # 状态触发
    if args['source_img']:
        args['cli_mode'] = True
        start()
        quit()
    # UI设置
    window = tk.Tk()
    window.geometry("600x700")
    window.title("roop")
    window.configure(bg="#2d3436")
    window.resizable(width=False, height=False)

    # Contact information
    support_link = tk.Label(window, text="Donate to project <3", fg="#fd79a8", bg="#2d3436", cursor="hand2", font=("Arial", 8))
    support_link.place(x=180,y=20,width=250,height=30)
    support_link.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/sponsors/s0md3v"))

    # Select a face button
    face_button = tk.Button(window, text="Select a face", command=select_face, bg="#2d3436", fg="#74b9ff", highlightthickness=4, relief="flat", highlightbackground="#74b9ff", activebackground="#74b9ff", borderwidth=4)
    face_button.place(x=60,y=320,width=180,height=80)

    # Select a target button
    target_button = tk.Button(window, text="Select a target", command=select_target, bg="#2d3436", fg="#74b9ff", highlightthickness=4, relief="flat", highlightbackground="#74b9ff", activebackground="#74b9ff", borderwidth=4)
    target_button.place(x=360,y=320,width=180,height=80)

    # FPS limit checkbox
    limit_fps = tk.IntVar()
    fps_checkbox = tk.Checkbutton(window, relief="groove", activebackground="#2d3436", activeforeground="#74b9ff", selectcolor="black", text="Limit FPS to 30", fg="#dfe6e9", borderwidth=0, highlightthickness=0, bg="#2d3436", variable=limit_fps, command=toggle_fps_limit)
    fps_checkbox.place(x=30,y=500,width=240,height=31)
    fps_checkbox.select()

    # Keep frames checkbox
    keep_frames = tk.IntVar()
    frames_checkbox = tk.Checkbutton(window, relief="groove", activebackground="#2d3436", activeforeground="#74b9ff", selectcolor="black", text="Keep frames dir", fg="#dfe6e9", borderwidth=0, highlightthickness=0, bg="#2d3436", variable=keep_frames, command=toggle_keep_frames)
    frames_checkbox.place(x=37,y=450,width=240,height=31)

    # Start button
    start_button = tk.Button(window, text="Start", bg="#f1c40f", relief="flat", borderwidth=0, highlightthickness=0, command=lambda: [save_file(), start()])
    start_button.place(x=240,y=560,width=120,height=49)

    # Status label
    status_label = tk.Label(window, width=580, justify="center", text="Status: waiting for input...", fg="#2ecc71", bg="#2d3436")
    status_label.place(x=10,y=640,width=580,height=30)
    
    window.mainloop()
