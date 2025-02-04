#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graphic user interface implementation for Application with asynchronous video processing.

"""
import csv
import copy
import itertools
import time
from collections import Counter
from collections import deque
import tkinter
import customtkinter
import cv2
from PIL import Image, ImageTk
import asyncio
import threading
import pyautogui

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

customtkinter.set_appearance_mode("System")  # sets theme mode default: system, dark, light
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue")


class WindowUi(customtkinter.CTk):
    """
    Main User Gesture interface that inherits from customtkinter
    """

    prev_action = None

    def __init__(self):
        """Initializer at first call"""
        super().__init__()
        self.geometry(f"{1100}x{580}")
        self.title("Intelligent-System-For-Application-and-Appliance-Control")

        # Grid and responsiveness
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # # mode initializer ########################################################################
        self.mode = 0
        self.number = None

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # OpenCV Video Capture
        self.cap = cv.VideoCapture(0)

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

        # menu for seting appearance mode for light to dark and vice versa
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

        # create menu for changing theme
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        # create menu for changing theme
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)

        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        self.main_button_1 = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2,
                                                     text_color=("gray10", "#DCE4EE"), text="TRAIN MODEL",
                                                     command=self.train_model)
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

        # self.numberbutton_frame = customtkinter.CTkFrame(self)
        # self.numberbutton_frame.grid(row=0, column=4, padx=(20, 20), pady=(20, 0), sticky="nsew")
        # self.number_var = tkinter.IntVar(value=0)

        # create checkbox and switch frame
        self.checkbox_slider_frame = customtkinter.CTkFrame(self)
        self.checkbox_slider_frame.grid(row=1, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.checkbox_1 = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame, text="Enable drawing",
                                                    command=self.enable_disable_drawing)
        self.checkbox_1.grid(row=1, column=0, pady=(20, 0), padx=20, sticky="n")
        self.checkbox_2 = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame, text="Show FPS",
                                                    command=self.show_fps)
        self.checkbox_2.grid(row=2, column=0, pady=(20, 0), padx=20, sticky="n")
        self.checkbox_3 = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame, text="Hand Tracking blue",
                                                    command=self.show_blue)
        self.checkbox_3.grid(row=3, column=0, pady=20, padx=20, sticky="n")
        self.checkbox_4 = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame, text="Hand Tracking white",
                                                    command=self.show_white)
        self.checkbox_4.grid(row=4, column=0, pady=20, padx=20, sticky="n")

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
        self.fps = False
        self.blue = False
        self.white = False
        self.hand_sign_id = None
        self.prev_action = None
        # self.last_action_time = 0  # To track the time of the last action
        # self.cooldown = 0.5  # Cooldown in seconds (adjust as needed)
        self.loop = asyncio.new_event_loop()

        # Start asyncio loop in a separate thread
        threading.Thread(target=self.run_event_loop, daemon=True).start()

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

    async def update_video(self):
        """Capture video frame and update the label asynchronously"""
        keypoint_classifier = KeyPointClassifier()

        point_history_classifier = PointHistoryClassifier()

        use_brect = True

        # Read labels ###########################################################
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
        with open(
                'model/point_history_classifier/point_history_classifier_label.csv',
                encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            point_history_classifier_labels = [
                row[0] for row in point_history_classifier_labels
            ]

        # FPS Measurement ########################################################
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        # Coordinate history #################################################################
        history_length = 16
        point_history = deque(maxlen=history_length)

        # Finger gesture history ################################################
        finger_gesture_history = deque(maxlen=history_length)

        #  ########################################################################
        # mode = 0

        while self.is_running:
            ret, image = await asyncio.to_thread(self.cap.read)
            if ret:
                fps = cvFpsCalc.get()

                # Process Key (ESC: end) #################################################
                # key = cv.waitKey(10)
                # if key == 27:  # ESC
                #     break
                # self.number, self.mode = self.select_mode(key, self.mode)

                # Camera capture #####################################################
                ret, image = self.cap.read()
                if not ret:
                    break
                image = cv.flip(image, 1)  # Mirror display
                image = cv2.resize(image, (500, 400))

                # Detection implementation #############################################################
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                debug_image = copy.deepcopy(image)

                image.flags.writeable = False
                results = await asyncio.to_thread(self.hands.process, image)
                image.flags.writeable = True

                #  ####################################################################
                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                          results.multi_handedness):
                        self.log_to_tracking(f"{hand_landmarks}")
                        # Bounding box calculation
                        brect = calc_bounding_rect(debug_image, hand_landmarks)
                        # Landmark calculation
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list = pre_process_landmark(
                            landmark_list)
                        pre_processed_point_history_list = pre_process_point_history(
                            debug_image, point_history)

                        # Write to the dataset file
                        logging_csv(self.number, self.mode, pre_processed_landmark_list,
                                    pre_processed_point_history_list)
                        self.log_to_tracking(f"{self.number}, {self.mode}, {pre_processed_landmark_list}, "
                                             f"{pre_processed_point_history_list}")

                        # Hand sign classification
                        self.hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                        if self.hand_sign_id == 2:  # Point gesture
                            point_history.append(landmark_list[8])
                        else:
                            point_history.append([0, 0])

                        # Finger gesture classification
                        finger_gesture_id = 0
                        point_history_len = len(pre_processed_point_history_list)
                        if point_history_len == (history_length * 2):
                            finger_gesture_id = point_history_classifier(
                                pre_processed_point_history_list)

                        # Calculates the gesture IDs in the latest detection
                        finger_gesture_history.append(finger_gesture_id)
                        most_common_fg_id = Counter(
                            finger_gesture_history).most_common()

                        if self.drawing:
                            # Drawing part
                            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                            if self.white:
                                debug_image = draw_landmarks(debug_image, landmark_list)
                            if self.blue:
                                self.mp_drawing.draw_landmarks(debug_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                            debug_image = draw_info_text(
                                debug_image,
                                brect,
                                handedness,
                                keypoint_classifier_labels[self.hand_sign_id],
                                point_history_classifier_labels[most_common_fg_id[0][0]],
                            )
                else:
                    point_history.append([0, 0])

                debug_image = draw_point_history(debug_image, point_history)
                if self.fps:
                    debug_image = draw_info(debug_image, fps, self.mode, self.number)

                # Screen reflection #############################################################
                # cv.imshow('Application and Appliance Control System', debug_image)

                # Convert the frame to a Tkinter-compatible image
                img = Image.fromarray(debug_image)
                imgtk = ImageTk.PhotoImage(image=img)

                # Update the GUI in the main thread
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            await asyncio.sleep(0.01)  # Small delay to prevent overloading the event loop
            #
            current_time = time.time()
            if self.hand_sign_id == 8:  # Swipe left
                if self.prev_action != "left":
                    self.log_to_terminal("Swipe Left - Previous Slide")
                    pyautogui.press("left")
                    self.prev_action = "left"

            elif self.hand_sign_id == 3:  # Swipe right
                if self.prev_action != "right":
                    self.log_to_terminal("Swipe Right - Next Slide")
                    pyautogui.press("right")
                    self.prev_action = "right"

            else:
                self.prev_action = None

    def start_frame(self):
        self.log_to_terminal(f"Started Capture and Processing")
        self.is_running = True
        asyncio.run_coroutine_threadsafe(self.update_video(), self.loop)

    def change_gesture_id(self, value):
        """change Gesture ID for Classifications"""
        self.log_to_terminal(f"{value} ID selected")
        self.number = int(value)

    def train_model(self):
        """strikes the numbers gotten from self  numbers and press repeatedly to log to csv """
        pyautogui.press(str(self.number))
        self.log_to_terminal(f"Training gesture to id {self.number} ")

    # stops the frame
    def stop_frame(self):
        self.log_to_terminal(f"Stopped frame processing")
        self.is_running = False

    # starts the frame
    def run_event_loop(self):
        self.log_to_terminal(f"Running event loop")
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        self.log_to_terminal(f"Success!")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        self.log_to_terminal(f"Changed Appearance Mode {new_appearance_mode} ")
        customtkinter.set_appearance_mode(new_appearance_mode)

    # log messages to terminal
    def log_to_terminal(self, message):
        """Append a message to the terminal-like display."""
        self.termina_like_display.insert("end", f"{message}\n")
        self.termina_like_display.see("end")  # Auto-scroll to the latest entry

    # logs tracking messages to be viewed and displayed for the user to seee
    def log_to_tracking(self, message):
        """Append tracking landmark details to the terminal-like display"""
        self.tracking_like_display.insert("end", f"{message}\n")
        self.tracking_like_display.see("end")  # Auto-scroll to the latest entry

    # changes scales of the window size enabling zooming features
    def change_scaling_event(self, new_scaling: str):
        self.log_to_terminal(f"Adjusted Scale Size to {new_scaling}")
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def show_fps(self):
        """
        this shows the fps when user enables it
        """
        if self.checkbox_2.get():  # Returns True if checked, False otherwise
            self.fps = True
            self.log_to_terminal(" enabled FPS")
        else:
            self.fps = False
            self.log_to_terminal("disabled FPS")

    def show_white(self):
        """
        sets color of hand drawing
        :return:
        """
        if self.checkbox_4.get():  # Returns True if checked, False otherwise
            self.white = True
            self.log_to_terminal(" enabled Hand Tracing White")
        else:
            self.white = False
            self.log_to_terminal("disabled Hand Tracing White")

    def show_blue(self):
        """
        sets color of hand drawing
        :return:
        """
        if self.checkbox_3.get():  # Returns True if checked, False otherwise
            self.blue = True
            self.log_to_terminal(" enabled Hand Tracing blue")
        else:
            self.blue = False
            self.log_to_terminal("disabled Hand Tracing blue")

    def enable_disable_drawing(self):
        """Enable or disable landmark drawing based on checkbox state."""
        if self.checkbox_1.get():  # Returns True if checked, False otherwise
            self.drawing = True
            self.log_to_terminal("Drawing enabled")
        else:
            self.drawing = False
            self.log_to_terminal("Drawing disabled")

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

    # def select_mode(self, key, mode):
    #     number = -1
    #     if 48 <= key <= 57:  # 0 ~ 9
    #         number = key - 48
    #     if key == 110:  # n
    #         self.mode = 0
    #     if key == 107:  # k
    #         self.mode = 1
    #     if key == 104:  # h
    #         self.mode = 2
    #     return number, mode

    def on_closing(self):
        """Handle application closing"""
        self.is_running = False
        self.cap.release()  # Release the camera
        self.loop.stop()  # Stop asyncio loop
        self.destroy()  # Close the window


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 0, 5), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    app = WindowUi()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)  # Ensure camera is released on close
    app.mainloop()
