# RH56 Controller (ROS 2)

This package provides a ROS 2 driver for the Inspire RH56DFX robotic hand, which wraps the serial interface with a Python API. Force-control (in Newtons) is WIP.


### Building
1.  Clone this repository into your ROS 2 workspace's `src` directory.
2.  Navigate to the root of your workspace.
3.  Build the package using `colcon`:
    ```bash
    colcon build --packages-select rh56_controller
    ```
4.  Source the workspace's `setup.bash` file:
    ```bash
    source install/setup.bash
    ```

## Usage

*Temporary Usage*: By default, the `inspire_hand.service` connects to the hands' serial bus and publishes/subscribes to the `inspire/state` and `inspire/cmd` topics. Stop this service with: `sudo systemctl stop inspire_hand.service`. 

Then for some reason, `pyserial` is denied access to the hands USB device. As a temporary fix, run `sudo chmod 666 /dev/ttyUSB0`. I will add this to the udev rules imminently if I can't fix this issue.

---

To run the driver node, use the provided launch file. You can specify the serial port, which is `/dev/ttyUSB0` by default if not specified.

```bash
ros2 launch rh56_controller rh56_controller.launch.py serial_port:=/dev/ttyUSB0
```



## ROS API

### Published Topics

*   **`/hands/state`** (`custom_ros_messages/msg/MotorStates`)
    *   A not-quite-mirror of the original `inspire/state` topic. Publishes the current angle of each finger joint in radians, not the 0-1000 range, in a 12-element array (`[right[6] + left[6]]`). Subject to change if this is annoying.

### Subscribed Topics

*   **`/hands/cmd`** (`custom_ros_messages/msg/MotorCmds`)
    *   A not-quite-mirror of the original `inspire/cmd` topic. Subscribes to the 12-element array of commanded angles (`[right[6] + left[6]]`) for each finger joint in radians, not the 0-1000 range. Subject to change if this is annoying.

### Services
*   **`/hands/set_angles`** (`custom_ros_messages/srv/SetHandAngles`)
    *   Input: 6-element float array, hand specification ('left', 'right', 'both')
    * This command opens both hands.
    * ```bash
         ros2 service call /hands/set_angles custom_ros_messages/srv/SetHandAngles "{angles: [1000, 1000, 1000, 1000, 1000, 1000], hand: 'both'}"
*   **`/hands/calibrate_force_sensors`** (`std_srvs/srv/Trigger`)
    *   Initiates the hardware's force sensor calibration routine. This process takes approximately 15 seconds.
*   **`/hands/save_parameters`** (`std_srvs/srv/Trigger`)
    *   Saves the current configuration to the hand's non-volatile memory.
*   **TODO: `/hands/adaptive_force_control`** (`rh56_controller/srv/AdaptiveForce`)
    *   Executes the advanced adaptive force control routine. **DOES NOT WORK YET**
* Preliminary feature: named gestures `{name: String: angles: List[Int]}` in the [`rh56_driver.py:self._gesture_library`](https://github.com/correlllab/rh56_controller/blob/dcda3061751199523323d6b24221c99eade7b0a5/rh56_controller/rh56_driver.py#L78) class dictionary will autogenerate `/hands/<gesture>`, `/hands/left/<gesture>`, and `/hands/right/<gesture>` services. Run `ros2 service list | grep '^/hands/'` to see the full list of generated services. These are generic `Trigger` services and can be run with: 
    * ```bash
        ros2 service call /hands/<gesture> std_srvs/srv/Trigger
        ros2 service call /hands/left/<gesture> std_srvs/srv/Trigger
        ros2 service call /hands/right/<gesture> std_srvs/srv/Trigger
    * ```bash
        ros2 service list | grep '^/hands/'
        /hands/close
        /hands/left/close
        /hands/left/open
        /hands/left/pinch
        /hands/left/point
        /hands/open
        /hands/pinch
        /hands/point
        /hands/right/close
        /hands/right/open
        /hands/right/pinch
        /hands/right/point

## Examples

### Close the Index Finger
The joints on the hand are ordered as such: `[pinky, ring, middle, index, thumb_bend (pitch), thumb_rotate (yaw)]`. This command closes the index finger while opening the other joints.

```bash
 ros2 service call /hands/set_angles custom_ros_messages/srv/SetHandAngles "{angles: [1000, 1000, 1000, 0, 1000, 1000], hand: 'both'}"
```

### Calibrate the Sensors

```bash
ros2 service call /calibrate_force_sensors std_srvs/srv/Trigger
```

### Use Adaptive Force Control

Call the service to move fingers to specific angles while trying to achieve target forces.

```bash
ros2 service call /adaptive_force_control rh56_controller/srv/AdaptiveForce '{
    "target_forces": [100, 100, 100, 500, 100, 100],
    "target_angles": [1000, 1000, 1000, 0, 1000, 1000],
    "step_size": 50,
    "max_iterations": 20
}'
```

---

<details>
<summary><b>Legacy Python Script Documentation (Pre-ROS)</b></summary>

## RH56 Advanced Hand Controller (Legacy)
## 1. Project Overview

This project contains a Python script (`controller.py`) for controlling the RH56 dexterous hand via a serial port. The script encapsulates the low-level communication protocol and provides a high-level `RH56Hand` Python class, making it easier for developers to implement complex control logic.

Currently, the project features a **Adaptive Force Control** function that can adjust force thresholds while dynamically changing finger positions to achieve a preset contact force.

## 2. Core Features

- **Basic Control**:
  - Set/read the angle for all six degrees of freedom (five fingers + thumb rotation).
  - Set/read the movement speed for each finger.
  - Set/read the force control threshold for each finger (unit: grams).
- **Sensor Reading**:
  - Real-time reading of the pressure sensor for each finger (unit: grams).
- **Force Sensor Calibration**:
  - Provides an interactive calibration routine to calibrate the force sensors before precise control operations.
- **Advanced Control Logic**:
  - **Adaptive Force Control (`adaptive_force_control`)**: This is an advanced control mode with the following characteristics:
    1. **Position-Force Coordinated Control**: Can simultaneously move fingers to a target **angle** and have them reach a target **contact force**.
    2. **Step-wise Adjustment**: Gradually moves fingers to the target position instead of all at once, making the control process smoother and more stable.
    3. **Intelligent Force Adjustment**: During movement, it dynamically adjusts the force control threshold based on the difference between the current force reading and the original target.

## 3. Setup and Installation

### Hardware
- RH56 Dexterous Hand
- USB-to-Serial adapter to connect the hand to the computer

### Software
- Python 3
- `pyserial` library
- `numpy` library


## 4. Configuration

Before running the script, you need to modify two key parameters at the **bottom** of the `controller.py` file, inside the `if __name__ == "__main__":` block, according to your setup:

1.  **Serial Port (`port`)**:
    -   Find the line `hand = RH56Hand(...)`.
    -   Change the `port` parameter to the actual serial port recognized by your computer.
        -   **Windows**: e.g., `COM3`, `COM4`
        -   **macOS/Linux**: e.g., `/dev/tty.usbserial-xxxx` or `/dev/ttyUSB0`

2.  **Hand ID (`hand_id`)**:
    -   In the same line, modify the `hand_id` parameter.
        -   **1**: Right Hand
        -   **2**: Left Hand

**Example:**
```python
if __name__ == "__main__":
    # Modify the parameters here based on your hardware connection
    hand = RH56Hand(port="/dev/tty.usbserial-1130", hand_id=1) 
    ...
```

## 5. Usage

The script can be run directly to start the pre-configured **Adaptive Force Control** demonstration.

### Steps to Run
1.  **Connect Hardware**: Ensure the dexterous hand is correctly connected to the computer and powered on.
2.  **Modify Configuration**: Correctly configure the serial port and hand ID as described in the previous section.
3.  **Execute Script**: Run the following command in your terminal:
    ```bash
    python controller.py
    ```
4.  **Start Calibration (Optional)**:
    -   By default, the script first runs `demonstrate_force_calibration`.
    -   You will see the prompt `Press Enter to start calibration...`. Press Enter to begin. The calibration process takes about 15 seconds.
    -   If you do not need to calibrate, you can comment out the `demonstrate_force_calibration(...)` line in the `__main__` block.
5.  **Observe Adaptive Force Control**:
    -   After calibration, the script will automatically start the `adaptive_force_control` routine.
    -   You will see real-time output in the terminal showing each finger's **current angle**, **current force reading**, **original target force**, and the **action taken** for each iteration.
    -   The program will finish after reaching the targets or the maximum number of iterations and will print a final summary report.

## 6. Key Methods (API)


---
`force_set(thresholds: List[int])`
- **Function**: Directly sets the force control thresholds for the 6 fingers.
- **Parameters**: `thresholds` - A list of 6 integers, with each value ranging from 0-1000g.

---
`angle_set(angles: List[int])`
- **Function**: Sets the target angles for the 6 fingers.
- **Parameters**: `angles` - A list of 6 integers, with each value ranging from 0-1000.

---
`force_act() -> Optional[List[int]]`
- **Function**: Reads and returns the current force sensor readings for the 6 fingers (unit: grams).
- **Returns**: A list of 6 integers, or `None` if the read fails.

---
`angle_read() -> Optional[List[int]]`
- **Function**: Reads and returns the current angle positions for the 6 fingers.
- **Returns**: A list of 6 integers, or `None` if the read fails.

---
`adaptive_force_control(target_forces: List[int], target_angles: List[int], step_size: int = 50, max_iterations: int = 20)`
- **Function**: Executes the advanced adaptive force control routine.
- **Parameters**:
  - `target_forces`: List of target contact forces (unit: grams).
  - `target_angles`: List of target angles.
  - `step_size`: The angle step for each iteration.
  - `max_iterations`: The maximum number of iterations.
- **Returns**: A dictionary containing detailed results and history.

---
`demonstrate_force_calibration(port: str, hand_id: int)`
- **Function**: Starts an interactive force sensor calibration routine. It is recommended to run this before performing precision tasks.

## 7. Known Issues and Limitations

- **Controller Precision and Response**: The precision and response speed of the finger controllers are currently limited.
- **Force Control Overshoot**: Even at the slowest movement speeds, the force control can overshoot the preset target values by 50-100 grams.
- **High-Speed Behavior**: When moving at high speeds, the fingers tend to "ignore" the preset maximum force thresholds and move directly to their peak force.
- **Testing Status**: All features have currently only undergone light and informal testing. 

</details>
