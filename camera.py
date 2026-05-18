import cv2
import time
import tkinter as tk
from PIL import Image, ImageTk

def show_camera():
    cap = cv2.VideoCapture(0)
    time.sleep(2)

    root = tk.Tk()
    root.title("Hairstyle AI:  Press SPACE to capture")
    root.geometry("800x600") # Chooses size of the window
    root.lift() # Brings the camera window to the front
    root.attributes("-topmost", True)
    root.focus_force() # Focuses on the window, so instead of clicking on it, and pressing space, you just press space

    label = tk.Label(root)
    label.pack()
        
    captured = [False]

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)
        if not captured[0]:
            root.after(10, update_frame)

    def capture(event):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite("test_frame.jpg", frame)
            print("Frame captured!")
            captured[0] = True
            cap.release()
            root.destroy()

    root.bind("<space>", capture)
    update_frame()
    root.mainloop()

show_camera()