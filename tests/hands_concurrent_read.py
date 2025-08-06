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
    time.sleep(0.005)    
    forces = hand.force_act()
    time.sleep(0.005)
    temps = hand.temp_read()
    return angles
# stream read angles over a user-specified period of time, then compute frequency of reading

def read_all_hand_statuses_threading(hands: List[RH56Hand]) -> List[Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]]:
    results = [None] * len(hands)

    def worker(idx: int, hand: RH56Hand):
        results[idx] = read_hand_status(hand)

    threads = []
    for i, hand in enumerate(hands):
        t = threading.Thread(target=worker, args=(i, hand))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return results


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
                    
                    # Option 1: concurrent with ThreadPoolExecutor
                    # with ThreadPoolExecutor(max_workers=2) as executor:
                    #     right_future = executor.submit(read_hand_status, righthand)
                    #     left_future = executor.submit(read_hand_status, lefthand)
                        
                    #     right_angles, right_forces, right_temps = right_future.result()
                    #     left_angles, left_forces, left_temps = left_future.result()

                    #     angles = right_angles + left_angles
                    #     if len(angles) > 0:
                    #         read_angles.append(angles)

                    # Option 2: serialized
                    right_angles = righthand.angle_read()
                    right_forces = righthand.force_act()
                    # right_temps = righthand.temp_read()
                    left_angles = lefthand.angle_read()
                    left_forces = lefthand.force_act()
                    # left_temps = lefthand.temp_read()
                    angles = right_angles + left_angles
                    if angles is not None:
                        print(angles)
                        read_angles.append(angles)

                    # Option 3: concurrent with threading
                    # angles, forces, temps = read_all_hand_statuses_threading([righthand, lefthand])
                    # if angles[0] is not None and angles[1] is not None:
                    #     print(angles[0] + angles[1])
                    #     read_angles.append(angles[0] + angles[1])

                frequency = len(read_angles) / duration if duration > 0 else 0
                print(f"Read {len(read_angles)} angles in {duration} seconds.\nFrequency: {frequency} Hz")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
        else:
            print("Invalid input. Please enter a valid integer or 'exit' to quit.")