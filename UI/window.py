#!/usr/bin/env python3
"""
Graphic user interface implementation for Application with asynchronous video processing.
"""
import tkinter
import customtkinter
import mediapipe as mp
from PIL import Image, ImageTk
import cv2
import asyncio
import threading

customtkinter.set_appearance_mode("System")  # sets theme mode default: system, dark, light
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue")


class WindowUi(customtkinter.CTk):
    """
    Main User Gesture interface that inherits from customtkinter
    """

    def __init__(self):
        """Initializer at first call"""
        super().__init__()
        self.geometry(f"{1100}x{580}")
        self.title("Intelligent-System-For-Application-and-Appliance-Control")

        # Grid and responsiveness
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # OpenCV Video Capture
        self.cap = cv2.VideoCapture(0)

        # Create a Canvas to Display Video
        self.video_label = customtkinter.CTkLabel(self, text="")
        self.video_label.grid(row=0, column=1, padx=(2, 0), pady=(5, 0), sticky="nsew")

        self.sidebar_frame = customtkinter.CTkFrame(self, width=130, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="ACTIONS",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Start", command=self.start_frame)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Stop", command=self.stop_frame)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Exit", command=self.on_closing)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))

        self.is_running = False
        self.loop = asyncio.new_event_loop()

        # Start asyncio loop in a separate thread
        threading.Thread(target=self.run_event_loop, daemon=True).start()

    def run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start_frame(self):
        self.is_running = True
        asyncio.run_coroutine_threadsafe(self.update_video(), self.loop)

    def stop_frame(self):
        self.is_running = False

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    async def update_video(self):
        """Capture video frame and update the label asynchronously"""
        while self.is_running:
            ret, frame = await asyncio.to_thread(self.cap.read)
            if ret:
                # Flip frame horizontally for a mirror view
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (500, 400))

                # Convert the BGR frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame with MediaPipe
                results = await asyncio.to_thread(self.hands.process, rgb_frame)

                # Draw MediaPipe landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Convert the frame to a Tkinter-compatible image
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)

                # Update the GUI in the main thread
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            await asyncio.sleep(0.01)  # Small delay to prevent overloading the event loop

    def on_closing(self):
        """Handle application closing"""
        self.is_running = False
        self.cap.release()  # Release the camera
        self.loop.stop()  # Stop asyncio loop
        self.destroy()  # Close the window


if __name__ == '__main__':
    app = WindowUi()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)  # Ensure camera is released on close
    app.mainloop()