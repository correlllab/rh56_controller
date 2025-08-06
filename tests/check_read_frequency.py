import os
import sys
import time
sys.path.append("../")
from rh56_controller.rh56_hand import RH56Hand
import threading

# stream read angles over a user-specified period of time, then compute frequency of reading

if __name__ == "__main__":
    hand = RH56Hand(port="/dev/ttyUSB0", hand_id=1)

    while True:
        user_input = input("Enter an integer n to read for n seconds, or 'exit' to quit: ").strip().lower()
        if user_input in ['exit', 'quit', 'q']:
            print("Exiting...")
            break
        elif user_input.isdigit():
            try:
                read_angles = []
                duration = min(int(user_input), 10)  # Ensure at most 10 seconds
                start = time.time()
                while time.time() - start < duration:
                    angle = hand.angle_read()
                    if angle is not None:
                        print(angle)
                        read_angles.append(angle)
                frequency = len(read_angles) / duration if duration > 0 else 0
                print(f"Read {len(read_angles)} angles in {duration} seconds.\nFrequency: {frequency} Hz")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
        else:
            print("Invalid input. Please enter a valid integer or 'exit' to quit.")