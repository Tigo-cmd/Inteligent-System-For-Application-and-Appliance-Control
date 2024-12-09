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
from HelperFunctions import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

import csv
import copy
import itertools
from collections import Counter
from collections import deque
import numpy as np

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

        # # keypoint and point classifiers
        # self.keypoint_classifier = KeyPointClassifier()
        # self.point_history_classifier = PointHistoryClassifier()
        #
        # # Read labels basically csv files ###########################################################
        # with open('model/keypoint_classifier/keypoint_classifier_label.csv',
        #           encoding='utf-8-sig') as f:
        #     self.keypoint_classifier_labels = csv.reader(f)
        #     self.keypoint_classifier_labels = [
        #         row[0] for row in self.keypoint_classifier_labels
        #     ]
        # with open(
        #         'model/point_history_classifier/point_history_classifier_label.csv',
        #         encoding='utf-8-sig') as f:
        #     self.point_history_classifier_labels = csv.reader(f)
        #     self.point_history_classifier_labels = [
        #         row[0] for row in self.point_history_classifier_labels
        #     ]
        #
        # # FPS Measurement ########################################################
        # self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        #
        # # Coordinate history #################################################################
        # self.history_length = 16
        # self.point_history = deque(maxlen=self.history_length)
        #
        # # Finger gesture history ################################################
        # self.finger_gesture_history = deque(maxlen=self.history_length)
        #
        # # mode initializer ########################################################################
        self.mode = 0

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
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

        # create menu for changing theme
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        # create menu for changing theme
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                               values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        self.main_button_1 = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2,
                                                     text_color=("gray10", "#DCE4EE"), text="TRAIN MODEL")
        self.main_button_1.grid(row=5, column=3, padx=10, pady=10, sticky="nsew")

        # change maximum number of hand to detect
        self.Max_hands_label = customtkinter.CTkLabel(self.sidebar_frame, text="Max No Of Hands:", anchor="w")
        self.Max_hands_label.grid(row=4, column=0, padx=20, pady=(0, 20))
        # create menu for changing hand detections
        self.Max_hands_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                 values=["1", "2", "3", "4", "5"],
                                                                 command=self.change_Max_hands)
        self.Max_hands_optionemenu.grid(row=4, column=0, padx=20, pady=(40, 0))
        self.Max_hands_optionemenu.grid(row=4, column=0, padx=20, pady=(40, 0))

        # detection confidence label
        self.detection_label = customtkinter.CTkLabel(self.sidebar_frame, text="Min detection Confidence", anchor="w")
        self.detection_label.grid(row=3, column=0, padx=20, pady=(0, 65))
        # create menu for changing hand detections
        self.detection_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                 values=["0.5", "0.6", "0.7", "0.8", "0.9"],
                                                                 command=self.change_detection_confidence)
        self.detection_optionemenu.grid(row=3, column=0, padx=20, pady=(10, 30))
        self.detection_optionemenu.grid(row=3, column=0, padx=20, pady=(10, 30))

        self.tracking_label = customtkinter.CTkLabel(self.sidebar_frame, text="Min Tracking Confidence", anchor="w")
        self.tracking_label.grid(row=3, column=0, padx=20, pady=(50, 0))
        # create menu for changing hand detections
        self.tracking_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                values=["0.5", "0.6", "0.7", "0.8", "0.9"],
                                                                command=self.change_tracking_confidence)
        self.tracking_optionemenu.grid(row=3, column=0, padx=20, pady=(100, 0))
        self.tracking_optionemenu.grid(row=3, column=0, padx=20, pady=(100, 0))

        # gesture ID selector
        self.gesture_label = customtkinter.CTkLabel(self.sidebar_frame, text="Gesture ID", anchor="w")
        self.gesture_label.grid(row=4, column=0, padx=20, pady=(100, 0))
        # create menu for changing hand detections
        self.gesture_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                               values=["0", "1", "2", "3", "4", "5", "6", "7", "8",
                                                                       "9"],
                                                               command=self.change_gesture_id)
        self.gesture_optionemenu.grid(row=4, column=0, padx=50, pady=(150, 0))
        self.gesture_optionemenu.grid(row=4, column=0, padx=50, pady=(150, 0))

        # create radiobutton frame
        self.radiobutton_frame = customtkinter.CTkFrame(self)
        self.radiobutton_frame.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.radio_var = tkinter.IntVar(value=0)
        self.label_radio_group = customtkinter.CTkLabel(master=self.radiobutton_frame, text="Data Recording Mode")
        self.label_radio_group.grid(row=0, column=2, columnspan=1, padx=10, pady=10, sticky="")
        self.radio_button_1 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var,
                                                           value=1, text="Static Gesture Mode   ",
                                                           command=self.mode_selector)
        self.radio_button_1.grid(row=3, column=2, pady=10, padx=20, sticky="n")
        self.radio_button_2 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var,
                                                           value=2, text="Dynamic Gesture Mode",
                                                           command=self.mode_selector)
        self.radio_button_2.grid(row=2, column=2, pady=10, padx=20, sticky="n")
        self.radio_button_3 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var,
                                                           value=0, text="Normal Gesture Mode   ",
                                                           command=self.mode_selector)
        self.radio_button_3.grid(row=1, column=2, pady=10, padx=20, sticky="n")

        self.numberbutton_frame = customtkinter.CTkFrame(self)
        self.numberbutton_frame.grid(row=0, column=4, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.number_var = tkinter.IntVar(value=0)

        # create checkbox and switch frame
        self.checkbox_slider_frame = customtkinter.CTkFrame(self)
        self.checkbox_slider_frame.grid(row=1, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.checkbox_1 = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame, text="Enable drawing",
                                                    command=self.enable_disable_drawing)
        self.checkbox_1.grid(row=1, column=0, pady=(20, 0), padx=20, sticky="n")
        self.checkbox_2 = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame, text="Show FPS")
        self.checkbox_2.grid(row=2, column=0, pady=(20, 0), padx=20, sticky="n")
        self.checkbox_3 = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame)
        self.checkbox_3.grid(row=3, column=0, pady=20, padx=20, sticky="n")

        # create a terminal like display frame
        self.termina_like_display = customtkinter.CTkTextbox(self, wrap="word", width=100, height=300)
        self.termina_like_display.grid(row=1, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.termina_like_display.grid_columnconfigure(0, weight=1)
        self.termina_like_display.grid_rowconfigure(4, weight=1)

        # create a terminal like display frame for landmark tracking
        self.tracking_like_display = customtkinter.CTkTextbox(self, wrap="word", width=100, height=300)
        self.tracking_like_display.grid(row=1, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tracking_like_display.grid_columnconfigure(0, weight=1)
        self.tracking_like_display.grid_rowconfigure(4, weight=1)

        # set default values
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        self.detection_optionemenu.set("0.5")
        self.tracking_optionemenu.set("0.5")
        self.Max_hands_optionemenu.set("1")
        self.gesture_optionemenu.set("0")

        self.is_running = False
        self.drawing = False
        self.loop = asyncio.new_event_loop()

        # Start asyncio loop in a separate thread
        threading.Thread(target=self.run_event_loop, daemon=True).start()

    def run_event_loop(self):
        self.log_to_terminal(f"Running event loop")
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        self.log_to_terminal(f"Success!")

    def log_to_terminal(self, message):
        """Append a message to the terminal-like display."""
        self.termina_like_display.insert("end", f"{message}\n")
        self.termina_like_display.see("end")  # Auto-scroll to the latest entry

    def change_gesture_id(self, value):
        """change Gesture Id for Classifications"""
        self.log_to_terminal(f"{value} ID selected")

    def mode_selector(self):
        """handler function for mode selection"""
        selected = self.radio_var.get()
        if selected == 0:
            self.log_to_terminal("Normal Mode selected.")
            self.mode = 0
        elif selected == 1:
            self.log_to_terminal("Static Gesture Mode selected.")
            self.mode = 1
        elif selected == 2:
            self.log_to_terminal("Dynamic Gesture Mode selected.")
            self.mode = 2

    def log_to_tracking(self, message):
        """Append tracking landmark details to the terminal-like display"""
        self.tracking_like_display.insert("end", f"{message}\n")
        self.tracking_like_display.see("end")  # Auto-scroll to the latest entry

    def change_scaling_event(self, new_scaling: str):
        self.log_to_terminal(f"Adjusted Scale Size to {new_scaling}")
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def start_frame(self):
        self.log_to_terminal(f"Started Capture and Processing")
        self.is_running = True
        asyncio.run_coroutine_threadsafe(self.update_video(), self.loop)

    def stop_frame(self):
        self.log_to_terminal(f"Stopped frame processing")
        self.is_running = False

    def enable_disable_drawing(self):
        """Enable or disable landmark drawing based on checkbox state."""
        if self.checkbox_1.get():  # Returns True if checked, False otherwise
            self.drawing = True
            self.log_to_terminal("Drawing enabled")
        else:
            self.drawing = False
            self.log_to_terminal("Drawing disabled")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        self.log_to_terminal(f"Changed Appearance Mode {new_appearance_mode} ")
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_Max_hands(self, value):
        self.log_to_terminal(f"Detecting {value} hand(s)")
        # Reinitialize MediaPipe Hands with updated parameters
        self.hands = mp.solutions.hands.Hands(max_num_hands=int(value))

    def change_detection_confidence(self, value):
        """ change detection confidence"""
        self.log_to_terminal(f"Minimum Detection confidence {value}")
        # Reinitialize MediaPipe detection confidence with updated parameters
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=int(value))

    def change_tracking_confidence(self, value):
        """ change tracking confidence """
        self.log_to_terminal(f"Tracking confidence {value}")
        self.hands = mp.solutions.hands.Hands(min_tracking_confidence=int(value))

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
                        self.log_to_tracking(f"{hand_landmarks}")
                        if self.drawing:
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
