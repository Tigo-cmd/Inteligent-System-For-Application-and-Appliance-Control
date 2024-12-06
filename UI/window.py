#!/usr/bin/env python3
"""
Graphic user interface implementation
"""
import customtkinter


class WindowUi:
    """
    main User Gesture interface
    """
    def __init__(self):
        """ initializer at first call """
        self.app = customtkinter.CTk()
        self.app.geometry("700x700")
        self.app.title("Intelligent-System-For-Application-and-Appliance-Control")
        self.Video_frame =