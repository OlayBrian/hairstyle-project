import cv2
import mediapipe as mp
import time
import tkinter as tk
from PIL import Image, ImageTk
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as mpImage
from mediapipe import ImageFormat
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
import os

print("Imports done")

model_path = "face_landmarker.task"

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

# Calculating facial geometry
def get_face_measurements(landmarks, w, h):
    def dist(p1, p2):
        x1, y1 = int(landmarks[p1].x * w), int(landmarks[p1].y * h)
        x2, y2 = int(landmarks[p2].x * w), int(landmarks[p2].y * h)
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    face_length = dist(10, 152)
    face_width = dist(234, 454)
    forehead_width = dist(103, 332)
    jaw_width = dist(172, 397)

    return {
        "face_length": round(face_length, 1),
        "face_width": round(face_width, 1),
        "forehead_width": round(forehead_width, 1),
        "jaw_width": round(jaw_width, 1)
    }

def get_face_shape(measurements):
    length = measurements["face_length"]
    width = measurements["face_width"]
    forehead = measurements["forehead_width"]
    jaw = measurements["jaw_width"]

    ratio = length / width if width != 0 else 0
    forehead_jaw_diff = forehead - jaw

    if ratio > 1.5:
        return "Oblong"
    elif ratio > 1.25:
        if forehead_jaw_diff > 20:
            return "Heart"
        else:
            return "Oval"
    elif ratio > 0.95:
        if abs(forehead - jaw) < 15:
            return "Square"
        elif forehead_jaw_diff > 20:
            return "Heart"
        else:
            return "Round"
    else:
        return "Round"
# This function draws the landmarks onto the face
def update_frame():
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mpImage(image_format=ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        # Draws dots
        if results.face_landmarks:
            h, w = frame.shape[:2]
            landmarks = results.face_landmarks[0]

            measurements = get_face_measurements(landmarks, w, h)
            y_pos = 30
            for key, value in measurements.items():
                cv2.putText(rgb_frame, f"{key}: {value}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_pos += 25

            face_shape = get_face_shape(measurements)
            cv2.putText(rgb_frame, f"Face Shape: {face_shape}", (10, y_pos + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            # Draws connections
            for connection in FACEMESH_TESSELATION:
                start, end = connection
                x1 = int(landmarks[start].x * w)
                y1 = int(landmarks[start].y * h)
                x2 = int(landmarks[end].x * w)
                y2 = int(landmarks[end].y * h)
                cv2.line(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()