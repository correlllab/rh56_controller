import os
import sys
import time
sys.path.append("../")
from rh56_controller.rh56_hand import RH56Hand

def test_angle_steps(hand, target_angle, step_size, finger_index, force_limit=1000, speed_limit=1000):
    """
    Test angle_set functionality by moving finger step by step to target angle
    
    Args:
        hand: RH56Hand instance
        target_angle: Target angle value (0-1000)
        step_size: Step size for each movement
        finger_index: Finger index (0-5) to control
        force_limit: Force threshold value (0-1000 grams)
        speed_limit: Speed limit value (0-1000)
    """
    try:
        # Initialize angles array - keep all fingers at current position except target finger
        print("Reading current angles...")
        current_angles = hand.angle_read()
        if current_angles is None:
            print("Failed to read current angles! Using default positions.")
            current_angles = [1000, 1000, 1000, 1000, 1000, 1000]  # Default open position
        
        print(f"Current angles: {current_angles}")
        
        # Set up force and speed limits
        force_thresholds = [1000] * 6
        speed_limits = [1000] * 6
        
        force_thresholds[finger_index] = force_limit
        speed_limits[finger_index] = speed_limit
        
        print(f"Setting force threshold to {force_limit}g for finger {finger_index}...")
        response = hand.force_set(force_thresholds)
        if response is None:
            print("Failed to set force thresholds!")
            return False
        
        print(f"Setting speed limit to {speed_limit} for finger {finger_index}...")
        response = hand.speed_set(speed_limits)
        if response is None:
            print("Failed to set speed limits!")
            return False
        
        time.sleep(0.1)  # Brief pause
        
        # Calculate movement steps for grasping (moving from open to close)
        start_angle = current_angles[finger_index]
        print(f"Grasping test: Moving finger {finger_index} from {start_angle} to {target_angle} with step size {step_size}")
        
        # For grasping, we expect target_angle to be smaller than start_angle
        if target_angle >= start_angle:
            print(f"Warning: Target angle ({target_angle}) should be smaller than start angle ({start_angle}) for grasping!")
            print("For grasping test: 1000=fully open, 0=fully closed")
            user_confirm = input("Do you want to continue anyway? (y/n): ").strip().lower()
            if user_confirm not in ['y', 'yes']:
                return False
        
        # Calculate steps for closing movement (decreasing angles)
        if target_angle < start_angle:
            # Moving towards lower angles (closing/grasping)
            steps = list(range(start_angle - step_size, target_angle - 1, -step_size))
            if steps[-1] != target_angle:
                steps.append(target_angle)
        else:
            # Moving towards higher angles (opening) - unusual for grasping test
            steps = list(range(start_angle + step_size, target_angle + 1, step_size))
            if steps[-1] != target_angle:
                steps.append(target_angle)
        
        print(f"Movement steps: {steps}")
        
        # Execute step-by-step movement
        for i, step_angle in enumerate(steps):
            # Prepare angle command
            angles = current_angles.copy()
            angles[finger_index] = step_angle
            
            print(f"Step {i+1}/{len(steps)}: Setting finger {finger_index} to angle {step_angle}")
            
            # Send angle command
            response = hand.angle_set(angles)
            if response is None:
                print(f"Failed to set angle {step_angle} at step {i+1}!")
                return False
            
            # Wait for movement to complete
            #time.sleep(0.2)  # Adjust this delay as needed
            
            # Read current position and force
            actual_angles = hand.angle_read()
            actual_forces = hand.force_act()
            
            if actual_angles is not None and actual_forces is not None:
                print(f"  Actual angle: {actual_angles[finger_index]}, Force: {actual_forces[finger_index]}g")
                
                # Check if force limit was reached
                if actual_forces[finger_index] >= force_limit:
                    print(f"  Force limit ({force_limit}g) reached! Stopping movement.")
                    break
            else:
                print("  Failed to read current status")
        
        print(f"\nMovement completed! Final position: {step_angle}")
        
        # Final status check
        time.sleep(0.5)
        final_angles = hand.angle_read()
        final_forces = hand.force_act()
        
        if final_angles is not None and final_forces is not None:
            print(f"Final status - Angle: {final_angles[finger_index]}, Force: {final_forces[finger_index]}g")
        
        return True
        
    except Exception as e:
        print(f"Error during angle step test: {e}")
        return False

def main():
    """
    Main function to run angle step test with user input
    """
    try:
        hand = RH56Hand(port="/dev/ttyUSB0", hand_id=1)
        print("Hand initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize hand: {e}")
        return
    
    print("\n=== RH56 Hand Grasping Test ===")
    print("\nFinger mapping:")
    print("0 - Pinky finger (小拇指)")
    print("1 - Ring finger (无名指)")
    print("2 - Middle finger (中指)")
    print("3 - Index finger (食指)")
    print("4 - Thumb bend (大拇指弯曲)")
    print("5 - Thumb rotation (大拇指旋转)")
    
    while True:
        try:
            print("\n" + "="*50)
            user_input = input("Enter command ('test' for angle test, 'open' to open hand, 'read' to check status, 'status' for hand status, 'clear' to clear errors, or 'exit' to quit): ").strip().lower()
            
            if user_input in ['exit', 'quit', 'q']:
                print("Exiting...")
                break
                
            elif user_input in ['status', 's']:
                hand.print_status()
                
            elif user_input in ['clear', 'c']:
                hand.clear_errors()
                print("Errors cleared!")
                
            elif user_input in ['open', 'o']:
                # Open hand (all fingers to maximum angle)
                open_angles = [1000, 1000, 1000, 1000, 1000, 1000]
                print("Opening hand...")
                response = hand.speed_set([1000, 1000, 1000, 1000, 1000, 1000])
                response = hand.angle_set(open_angles)
                if response is not None:
                    print("Hand opened successfully!")
                else:
                    print("Failed to open hand!")
                    
            elif user_input in ['read', 'r']:
                # Read current angles and forces
                actual_angles = hand.angle_read()
                actual_forces = hand.force_act()
                if actual_angles is not None and actual_forces is not None:
                    print(f"Current angles: {actual_angles}")
                    print(f"Current forces: {actual_forces}")
                else:
                    print("Failed to read current status!")
                    
            elif user_input in ['test', 't']:
                # Angle step test for grasping
                try:
                    finger_index = int(input("Enter finger index (0-5): "))
                    if not (0 <= finger_index <= 5):
                        print("Finger index must be between 0 and 5!")
                        continue
                        
                    target_angle = int(input("Enter target angle for grasping (0-1000, smaller than current for closing): "))
                    if not (0 <= target_angle <= 1000):
                        print("Target angle must be between 0 and 1000!")
                        continue
                        
                    step_size = int(input("Enter step size for gradual closing: "))
                    if step_size <= 0:
                        print("Step size must be positive!")
                        continue
                        
                    force_limit = int(input("Enter force limit (0-1000g, default 1000): ") or "1000")
                    if not (0 <= force_limit <= 1000):
                        print("Force limit must be between 0 and 1000!")
                        continue
                        
                    speed_limit = int(input("Enter speed limit (0-1000, default 1000): ") or "1000")
                    if not (0 <= speed_limit <= 1000):
                        print("Speed limit must be between 0 and 1000!")
                        continue
                    
                    print(f"\n--- Testing finger {finger_index} grasping movement ---")
                    print(f"Target angle: {target_angle} (closing direction)")
                    print(f"Step size: {step_size}")
                    print(f"Force limit: {force_limit}g")
                    print(f"Speed limit: {speed_limit}")
                    
                    success = test_angle_steps(hand, target_angle, step_size, finger_index, force_limit, speed_limit)
                    
                    if success:
                        print(f"✓ Grasping test completed successfully!")
                    else:
                        print(f"✗ Grasping test failed!")
                        
                except ValueError:
                    print("Please enter valid numbers!")
                    
            else:
                print("Invalid input! Available commands:")
                print("  - 'test' for angle step test")
                print("  - 'open' to open the hand")
                print("  - 'read' to check current status")
                print("  - 'status' to check hand status")
                print("  - 'clear' to clear errors")
                print("  - 'exit' to quit")
                
        except ValueError:
            print("Please enter valid input!")
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
