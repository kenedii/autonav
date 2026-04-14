# xbox_calibrate.py
# After plugging in USB Xbox controller to Jetson Nano, this script will allow you to configure the controller to be used to control the RC Car. 
# XPAD or XBOXDRV drivers may be required for Pygame to recognize the XBOX controller on the Jetson Nano device.
"""
Xbox Controller Calibration for LaTrax RC Car on Jetson Nano

This script helps you map your Xbox controller axes to LaTrax PWM values
for steering and throttle. It prints the corresponding PWM value in microseconds
for the LaTrax ESC and servo.

Python 3.6 compatible
"""

import pygame
import time

# =================== STEP 1: Prepare your hardware ===================
print("\nSTEP 1: Hardware Setup")
print(" - Connect your Xbox controller to the Jetson Nano via USB")
print(" - Turn on the LaTrax car transmitter")
print(" - Ensure the LaTrax receiver is bound and working")
print(" - Make sure no other script is controlling the car yet\n")
input("Press ENTER when ready to continue...")

# =================== STEP 2: Initialize Pygame ===================
print("\nSTEP 2: Initialize Xbox Controller")
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise Exception("No joystick detected. Connect your Xbox controller and try again.")

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Joystick detected: {joystick.get_name()}")

print("\nSTEP 3: Identify Axes")
print(" - Move the left stick fully left and right to test horizontal axis (steering)")
print(" - Move the left stick fully forward and backward to test vertical axis (throttle)")
print(" - Observe printed PWM values\n")
input("Press ENTER to start axis calibration...")

# =================== STEP 3: Configure Axes ===================
# Default mapping (can be adjusted if needed)
STEERING_AXIS = 0  # Left stick horizontal
THROTTLE_AXIS = 1  # Left stick vertical

# Expected normalized range (-1.0 to 1.0)
STEERING_MIN = -1.0
STEERING_MAX = 1.0
THROTTLE_MIN = -1.0
THROTTLE_MAX = 1.0

# =================== STEP 4: Axis to PWM Conversion ===================
def normalize_axis(value, min_val, max_val):
    """
    Converts normalized axis (-1 to 1) to LaTrax PWM microseconds (1000-2000)
    1500us = center, 1000us = full left/back, 2000us = full right/forward
    """
    pwm = int(1500 + 500 * value)
    return max(1000, min(2000, pwm))

print("\nSTEP 5: Move your sticks to see PWM values update in real time.")
print("Press Ctrl+C to finish calibration.\n")

# =================== STEP 5: Read Axis Values ===================
try:
    while True:
        pygame.event.pump()  # Update controller state

        # Read axes
        steer = joystick.get_axis(STEERING_AXIS)
        throttle = joystick.get_axis(THROTTLE_AXIS)

        # Convert to PWM
        steer_us = normalize_axis(steer, STEERING_MIN, STEERING_MAX)
        throttle_us = normalize_axis(-throttle, THROTTLE_MIN, THROTTLE_MAX)  # invert Y if needed

        # Print results
        print(f"\rSteering: {steer:+.3f} -> {steer_us}us | "
              f"Throttle: {throttle:+.3f} -> {throttle_us}us", end="")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n\nCalibration finished!")
    print("You can now use these axes and PWM mapping in your data collection script.")
