import os
import sys
import time
import threading
sys.path.append("../")
from rh56_controller.rh56_hand import RH56Hand

def test_middle_finger_force_monitor(hand, target_force, monitoring_frequency=165):
    """
    Test middle finger force monitoring at specified frequency
    
    Args:
        hand: RH56Hand instance
        target_force: Target force for middle finger (0-1000 grams)
        monitoring_frequency: Monitoring frequency in Hz (default 165)
    """
    try:
        print(f"Starting middle finger force monitoring test")
        print(f"Target force: {target_force}g")
        print(f"Monitoring frequency: {monitoring_frequency}Hz")
        
        # Set force thresholds - only middle finger (index 2) gets the target force
        # Other fingers set to high value (1000g) so they won't stop
        force_thresholds = [1000, 1000, target_force, 1000, 1000, 1000]
        print(f"Setting force thresholds: {force_thresholds}")
        
        # Set force thresholds
        response = hand.force_set(force_thresholds)
        if response is None:
            print("Failed to set force thresholds!")
            return False
        
        time.sleep(0.1)  # Brief pause
        
        speeds = [1000, 1000, 1000, 1000, 1000, 1000]
        response = hand.speed_set(speeds)
        if response is None:
            print("Failed to set speeds!")
            return False
        
        # Start closing movement - only middle finger moves to 0, others stay open
        target_angles = [1000, 1000, 0, 1000, 1000, 1000]
        print(f"Starting movement - middle finger closing to 0, others stay at 1000")
        response = hand.angle_set(target_angles)
        if response is None:
            print("Failed to start movement!")
            return False
        
        # Monitoring variables
        monitoring = True
        force_reached = False
        final_angle = None
        iteration_count = 0
        sleep_time = 1.0 / monitoring_frequency  # Calculate sleep time for target frequency
        start_time = time.time()
        
        print(f"Starting force monitoring at {monitoring_frequency}Hz...")
        print("Press Ctrl+C to stop monitoring manually\n")
        
        try:
            while monitoring:
                loop_start = time.time()
                
                # Read current forces and angles
                current_forces = hand.force_act()
                current_angles = hand.angle_read()
                
                if current_forces is None or current_angles is None:
                    print(f"Iteration {iteration_count}: Failed to read data")
                    time.sleep(sleep_time)
                    continue
                
                middle_finger_force = current_forces[2]  # Index 2 is middle finger
                middle_finger_angle = current_angles[2]
                
                # Check if target force reached
                if not force_reached and middle_finger_force >= target_force:
                    force_reached = True
                    final_angle = middle_finger_angle
                    
                    # Stop middle finger by setting its angle to current position
                    stop_angles = current_angles.copy()
                    stop_angles[2] = middle_finger_angle  # Keep current position
                    
                    response = hand.angle_set(stop_angles)
                    if response is not None:
                        print(f"\n*** TARGET FORCE REACHED! ***")
                        print(f"Force: {middle_finger_force}g (target: {target_force}g)")
                        print(f"Stopped at angle: {middle_finger_angle}")
                        print(f"Iterations: {iteration_count}")
                        print(f"Time: {time.time() - start_time:.2f}s")
                    else:
                        print(f"Force reached but failed to stop finger!")
                
                # Print status every 50 iterations (~0.3 seconds at 165Hz)
                if iteration_count % 50 == 0:
                    elapsed_time = time.time() - start_time
                    actual_freq = iteration_count / elapsed_time if elapsed_time > 0 else 0
                    status = "STOPPED" if force_reached else "MOVING"
                    print(f"Iter {iteration_count:4d} | Time: {elapsed_time:5.2f}s | "
                          f"Force: {middle_finger_force:3d}g | Angle: {middle_finger_angle:4d} | "
                          f"Freq: {actual_freq:5.1f}Hz | Status: {status}")
                
                # Check if finger reached target angle (close enough to 0) or force reached
                if force_reached or abs(middle_finger_angle - 0) <= 10:
                    if not force_reached:
                        print(f"\nMiddle finger reached target angle: {middle_finger_angle}")
                        final_angle = middle_finger_angle
                    monitoring = False
                    break
                
                iteration_count += 1
                
                # Maintain precise timing
                loop_elapsed = time.time() - loop_start
                sleep_time_adjusted = max(0, sleep_time - loop_elapsed)
                if sleep_time_adjusted > 0:
                    time.sleep(sleep_time_adjusted)
                    
        except KeyboardInterrupt:
            print(f"\nMonitoring stopped by user at iteration {iteration_count}")
            monitoring = False
        
        # Final status
        total_time = time.time() - start_time
        avg_frequency = iteration_count / total_time if total_time > 0 else 0
        
        print(f"\n=== MONITORING COMPLETE ===")
        print(f"Total iterations: {iteration_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average frequency: {avg_frequency:.1f} Hz")
        print(f"Force reached: {'YES' if force_reached else 'NO'}")
        if final_angle is not None:
            print(f"Final angle: {final_angle}")
        
        # Read final values
        final_forces = hand.force_act()
        final_angles = hand.angle_read()
        if final_forces and final_angles:
            print(f"Final middle finger force: {final_forces[2]}g")
            print(f"Final middle finger angle: {final_angles[2]}")
        
        return True
        
    except Exception as e:
        print(f"Error during force monitoring test: {e}")
        return False

def main():
    """
    Main function to run middle finger force monitoring test
    """
    try:
        hand = RH56Hand(port="/dev/ttyUSB0", hand_id=1)
        print("Hand initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize hand: {e}")
        return
    
    while True:
        try:
            print("\n" + "="*60)
            user_input = input("Enter command ('test' for force monitor, 'open' to open hand, 'read' to check status, 'status' for hand status, 'clear' to clear errors, or 'exit' to quit): ").strip().lower()
            
            if user_input in ['exit', 'quit', 'q']:
                print("Exiting...")
                break
                
            elif user_input in ['status', 's']:
                hand.print_status()
                
            elif user_input in ['clear', 'c']:
                hand.clear_errors()
                print("Errors cleared!")
                
            elif user_input in ['open', 'o']:
                # Open hand
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
                    print(f"Middle finger - Angle: {actual_angles[2]}, Force: {actual_forces[2]}g")
                else:
                    print("Failed to read current status!")
                    
            elif user_input in ['test', 't']:
                # Middle finger force monitoring test
                try:
                    target_force = int(input("Enter target force for middle finger (0-1000g): "))
                    
                    if not (0 <= target_force <= 1000):
                        print("Target force must be between 0 and 1000g!")
                        continue
                    
                    # Optional: ask for monitoring frequency
                    freq_input = input("Enter monitoring frequency in Hz (default 165): ").strip()
                    if freq_input:
                        monitoring_freq = int(freq_input)
                        if not (1 <= monitoring_freq <= 1000):
                            print("Frequency must be between 1 and 1000 Hz, using default 165Hz")
                            monitoring_freq = 165
                    else:
                        monitoring_freq = 165
                    
                    print(f"\n--- Middle Finger Force Monitoring Test ---")
                    print(f"Target force: {target_force}g")
                    print(f"Monitoring frequency: {monitoring_freq}Hz")
                    
                    success = test_middle_finger_force_monitor(hand, target_force, monitoring_freq)
                    
                    if success:
                        print("✓ Middle finger force monitoring test completed!")
                    else:
                        print("✗ Middle finger force monitoring test failed!")
                        
                except ValueError:
                    print("Please enter valid numbers!")
                    
            else:
                print("Invalid input! Available commands:")
                print("  - 'test' for middle finger force monitoring")
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
