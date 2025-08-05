import os
import sys

sys.path.append("../")
from rh56_controller.rh56_hand import RH56Hand

# test moving both hands concurrently
if __name__ == "__main__":
    # Initialize the right hand
    right_hand = RH56Hand("/dev/ttyUSB0", 1)
    left_hand = RH56Hand("/dev/ttyUSB0", 2)

    # Move both hands concurrently
    close = [0] * 6
    open = [1000] * 6

    # create user input to accept either 'c' or 'close' for close or 'o' for open
    # only close user input when user enters in exit, quit, q, or Ctrl+C
    while True:
        user_input = input("Enter 'c' to close hands, 'o' to open hands, or 'exit' to quit: ").strip().lower()
        if user_input in ['exit', 'quit', 'q']:
            print("Exiting...")
            break
        elif user_input in ['c', 'close']:
            right_hand.angle_set(close)
            left_hand.angle_set(close)
            print("Hands closed.")
        elif user_input in ['o', 'open']:
            right_hand.angle_set(open)
            left_hand.angle_set(open)
            print("Hands opened.")
        else:
            print("Invalid input. Please try again.")
