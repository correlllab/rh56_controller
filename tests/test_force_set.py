import os
import sys
import time
sys.path.append("../")
from rh56_controller.rh56_hand import RH56Hand

def test_force_set_and_close(hand, force_value, speed_value):
    """
    Test force_set functionality by setting force threshold and closing hand
    
    Args:
        hand: RH56Hand instance
        force_value: Force threshold value (0-1000 grams)
    """
    try:
        # Set the same force threshold for all 6 fingers
        force_thresholds = [1000,1000,force_value,1000,1000,1000]
        print(f"Setting force thresholds to {force_value}g for all fingers...")
        
        # Set force thresholds
        response = hand.force_set(force_thresholds)
        if response is not None:
            print("Force thresholds set successfully!")
        else:
            print("Failed to set force thresholds!")
            return False
        
        time.sleep(0.1)  # Brief pause
        
        # Close hand (set all angles to 0)
        #close_angles = [1000,1000,450,1000,1000,1000]
        #close_speed = [1000,1000,1000,1000,1000,1000]
        #response = hand.speed_set(close_speed)
        #response = hand.angle_set(close_angles)
        #time.sleep(0.5)  # Wait for a moment before final close
        close_angles = [1000,1000,0,1000,1000,1000]
        close_speed = [1000,1000,speed_value,1000,1000,1000]
        response = hand.speed_set(close_speed)
        response = hand.angle_set(close_angles)

        if response is not None:
            print("Hand closed successfully!")
            print("The fingers should stop when they reach the force threshold.")
        else:
            print("Failed to close hand!")
            return False
        
        # Read force values 10 times, every 0.25 seconds
        print("\nReading force values ...")
        for i in range(10):
            time.sleep(0.5)
            actual_forces = hand.force_act()
            if actual_forces is not None:
                print(f"Reading {i+1}: {actual_forces}")
            else:
                print(f"Reading {i+1}: Failed to read force values")
            
        return True
        
    except Exception as e:
        print(f"Error during force test: {e}")
        return False

def main():
    """
    Main function to run force_set test with user input
    """
    try:
        hand = RH56Hand(port="/dev/ttyUSB0", hand_id=1)
        print("Hand initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize hand: {e}")
        return
    
    while True:
        try:
            user_input = input("\nEnter force threshold (0-1000g), 'open' to open hand, 'read' to check force, or 'exit' to quit: ").strip().lower()
            
            if user_input in ['exit', 'quit', 'q']:
                print("Exiting...")
                break
                
            elif user_input in ['status','s']:
                hand.print_status()
            elif user_input in ['clear','c']:
                hand.clear_errors()
            elif user_input in ['open', 'o']:
                # Open hand
                open_angles = [1000,1000,1000,1000,1000,1000]
                print("Opening hand...")
                response = hand.speed_set([1000,1000,1000,1000,1000,1000])
                response = hand.angle_set(open_angles)
                if response is not None:
                    print("Hand opened successfully!")
                else:
                    print("Failed to open hand!")
                    
            elif user_input in ['read', 'r']:
                # Read current force values
                actual_forces = hand.force_act()
                actual_angles = hand.angle_read()
                if actual_forces is not None:
                    print(f"Current force values: {actual_forces}")
                    print(f"Current angle values: {actual_angles}")
                else:
                    print("Failed to read force values!")
                    
            elif user_input.isdigit():
                force_value = int(user_input)
                
                if 0 <= force_value <= 1000:
                    user_input = input("\nEnter the speed you need: ")
                    speed_value = int(user_input)
                    print(f"\n--- Testing with force threshold: {force_value}g ---")
                    success = test_force_set_and_close(hand, force_value, speed_value)
                    
                    if success:
                        print(f"✓ Force test completed with threshold {force_value}g")
                    else:
                        print(f"✗ Force test failed with threshold {force_value}g")
                else:
                    print("Force value must be between 0 and 1000 grams!")
                    
            else:
                print("Invalid input! Please enter:")
                print("  - A number (0-1000) for force threshold")
                print("  - 'open' to open the hand")
                print("  - 'read' to check current force values")
                print("  - 'exit' to quit")
                
        except ValueError:
            print("Please enter a valid number or command!")
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
