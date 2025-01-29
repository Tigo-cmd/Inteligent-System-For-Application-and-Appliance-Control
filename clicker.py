import pyautogui
import time

# Define the target area
min_target_x, min_target_y = 916, 737
max_target_x, max_target_y = 1350, 750

while True:
    x, y = pyautogui.position()  # Get current mouse position
    if min_target_x <= x <= max_target_x and min_target_y <= y <= max_target_y:
        print("Mouse is at the target area!")
        # Perform an action (e.g., clicking)
        pyautogui.click()  # Click at the current position
        print(f"Clicking {x}, {y}")
    time.sleep(0.1)