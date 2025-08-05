import os
import sys
import time
sys.path.append("../")
from rh56_controller.rh56_hand import RH56Hand
import threading
from typing import List, Tuple, Optional
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def read_hand_status(hand) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
    """Read status for a single hand (no shared lock needed)"""
    angles = hand.angle_read()
    forces = hand.force_act()
    temps = hand.temp_read()
    return angles, forces, temps
# stream read angles over a user-specified period of time, then compute frequency of reading

if __name__ == "__main__":
    lefthand = RH56Hand(port="/dev/ttyUSB0", hand_id=1)
    righthand = RH56Hand(port="/dev/ttyUSB0", hand_id=2)

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
                    
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        right_future = executor.submit(read_hand_status, righthand)
                        left_future = executor.submit(read_hand_status, lefthand)
                        
                        try:
                            right_angles, right_forces, right_temps = right_future.result()
                        except Exception as e:
                            print(f"Error reading right hand status: {e}")
                        try:
                            left_angles, left_forces, left_temps = left_future.result()
                        except Exception as e:
                            print(f"Error reading left hand status: {e}")

                        angles = right_angles + left_angles
                        read_angles.append(angles)

                frequency = len(read_angles) / duration if duration > 0 else 0
                print(f"Read {len(read_angles)} angles in {duration} seconds.\nFrequency: {frequency} Hz")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
        else:
            print("Invalid input. Please enter a valid integer or 'exit' to quit.")