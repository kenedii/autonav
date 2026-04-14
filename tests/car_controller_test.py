# car_controller_test.py
# After the XBOX controller is configured to control the RC Car with the configuration, 
# this Python script will allow you to control the car using the XBOX controller.
"""
Xbox controller test for LaTrax + PCA9685 using left stick for forward/reverse
Steering and throttle directions fixed.
"""

import pygame
import time
from smbus2 import SMBus

# ================= CONFIG =================
PCA_ADDR = 0x40
STEERING_CHANNEL = 0
THROTTLE_CHANNEL = 1
STEERING_AXIS = 0      # Left stick horizontal
THROTTLE_AXIS = 1      # Left stick vertical
PWM_FREQ = 50

STEERING_CENTER = 1500
STEERING_MIN = 1000
STEERING_MAX = 2000

THROTTLE_CENTER = 1500
THROTTLE_MIN = 1000
THROTTLE_MAX = 2000

# ================= PCA9685 HELPER =================
class PCA9685:
    def __init__(self, bus=1, address=0x40):
        self.bus = SMBus(bus)
        self.address = address
        self.set_pwm_freq(PWM_FREQ)

    def set_pwm_freq(self, freq_hz):
        prescaleval = 25000000.0
        prescaleval /= 4096.0
        prescaleval /= freq_hz
        prescaleval -= 1.0
        prescale = int(prescaleval + 0.5)
        self.bus.write_byte_data(self.address, 0x00, 0x10)  # sleep
        self.bus.write_byte_data(self.address, 0xFE, prescale)
        self.bus.write_byte_data(self.address, 0x00, 0x80)  # restart

    def set_pwm(self, channel, on, off):
        self.bus.write_byte_data(self.address, 0x06 + 4*channel, on & 0xFF)
        self.bus.write_byte_data(self.address, 0x07 + 4*channel, on >> 8)
        self.bus.write_byte_data(self.address, 0x08 + 4*channel, off & 0xFF)
        self.bus.write_byte_data(self.address, 0x09 + 4*channel, off >> 8)

    def set_us(self, channel, microseconds):
        pulse_length = 1000000 / PWM_FREQ / 4096
        pulse = int(microseconds / pulse_length)
        self.set_pwm(channel, 0, pulse)

# ================= SETUP =================
pca = PCA9685()
pca.set_us(STEERING_CHANNEL, STEERING_CENTER)
pca.set_us(THROTTLE_CHANNEL, THROTTLE_CENTER)

# ================= PYGAME =================
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise Exception("No joystick detected! Connect Xbox controller.")

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Joystick detected: {joystick.get_name()}")
print("\nXbox control with correct directions. Press Ctrl+C to quit.\n")

# ================= MAIN LOOP =================
try:
    while True:
        pygame.event.pump()

        # Steering: invert so right = positive PWM
        steer = -joystick.get_axis(STEERING_AXIS)
        steer_us = int(STEERING_CENTER + steer * (STEERING_MAX - STEERING_CENTER))
        pca.set_us(STEERING_CHANNEL, steer_us)

        # Throttle: invert so down = reverse (PWM < 1500)
        throttle = -joystick.get_axis(THROTTLE_AXIS)
        throttle_us = int(THROTTLE_CENTER + throttle * (THROTTLE_MAX - THROTTLE_CENTER))
        pca.set_us(THROTTLE_CHANNEL, throttle_us)

        print(f"\rSteer: {steer:+.3f}->{steer_us}us | "
              f"Throttle: {throttle:+.3f}->{throttle_us}us", end="")
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nStopping car control...")

finally:
    pca.set_us(STEERING_CHANNEL, STEERING_CENTER)
    pca.set_us(THROTTLE_CHANNEL, THROTTLE_CENTER)
    pygame.quit()
    print("Exited safely.")
