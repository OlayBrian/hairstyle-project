import cv2
import mediapipe as mp
import time
import tkinter as tk
from PIL import Image, ImageTk
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as mpImage
from mediapipe import ImageFormat

print("Imports done")

# Download the face landmarker model
import urllib.request
import os

model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )
    print("Model downloaded!")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

print("Model loaded, opening camera...")

cap = cv2.VideoCapture(0)
time.sleep(2)

root = tk.Tk()
root.title("Hairstyle AI - Face Landmarks")
root.geometry("800x600")
root.lift()
root.focus_force()

label = tk.Label(root)
label.pack()

def update_frame():
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mpImage(image_format=ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        if results.face_landmarks:
            h, w = frame.shape[:2]
            for landmark in results.face_landmarks[0]:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(rgb_frame, (x, y), 1, (0, 255, 0), -1)

        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()