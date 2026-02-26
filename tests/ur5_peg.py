import time, csv, threading
import numpy as np

from rh56_controller.rh56_hand import RH56Hand

ROBOT_IP = "192.168.0.5"
HAND_PORT = "/dev/ttyUSB0"
HAND_ID = 1

OUTPUT_CSV = f"ur5_task_data_{int(time.time())}.csv"

WRIST_CONTACT_SPIKE_N = 2.0  # phase0: f_norm - baseline > this => contact
WRIST_STABLE_THRESH_N = 3.0  # phase1: f_norm > this for stable time => stable
WRIST_STABLE_TIME_S = 1.0

WRIST_WINDOW_S = 0.3  # phase2: window size
WRIST_SPIKE_N = 1.0  # phase2: avg delta >= spike => spike_detected True
WRIST_DROP_N = 0.8  # phase2: after spike, avg delta <= -drop => open

BASELINE_ALPHA = 0.05


DEFAULT_OPEN = [1000, 1000, 1000, 1000, 1000, 0]
CLOSE_ANGLES = [1000, 1000, 1000, 0, 600, 150]

IDX_FINGER_INDEX = 3
IDX_FINGER_THUMB = 4


def apply_angles(hand, angles):
    hand.angle_set(list(angles))


def apply_speed(hand, speed):
    hand.speed_set([speed] * 6)


data_log = []
recording_active = False
pose_peg_prep = np.array(
    [
        [-0.999, -0.019, -0.041, -0.450],
        [0.026, 0.499, -0.866, -0.443],
        [0.037, -0.867, -0.498, 0.243],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

pose_peg_go = np.array(
    [
        [-0.999, -0.019, -0.041, -0.450],
        [0.026, 0.499, -0.866, -0.485],
        [0.037, -0.867, -0.498, 0.243],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

pose_peg_n = np.array(
    [
        [-0.999, -0.019, -0.041, -0.450],
        [0.026, 0.499, -0.866, -0.480],
        [0.037, -0.867, -0.498, 0.243],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

pose_peg_lift = np.array(
    [
        [-0.999, -0.019, -0.041, -0.450],
        [0.026, 0.499, -0.866, -0.480],
        [0.037, -0.867, -0.498, 0.290],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

pose_peg_put = np.array(
    [
        [-0.999, -0.019, -0.041, -0.450],
        [0.026, 0.499, -0.866, -0.450],
        [0.037, -0.867, -0.498, 0.270],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

hand_phase = 0
hand_done_event = threading.Event()


def background_recorder_with_hand(robot, hand, sample_period_s=0.01):
    """
    Background loop:
      - read wrist FT (for control + logging)
      - (optional) read hand forces (for logging)
      - run wrist-driven phase state machine to command the hand
    """
    global data_log, recording_active, hand_phase

    baseline = None
    stable_start = None

    window_samples = []
    window_start = time.monotonic()
    prev_avg = None
    spike_detected = False

    apply_speed(hand, 1000)
    apply_angles(hand, CLOSE_ANGLES)
    time.sleep(1)

    print("Background recording + hand control started...")

    while recording_active:
        t0 = time.time()  # epoch for logging sync
        tm = time.monotonic()  # monotonic for timers

        # -------- wrist FT --------
        wrist = robot.get_ft_data()  # expected [Fx,Fy,Fz,Tx,Ty,Tz]
        if wrist and len(wrist) >= 6:
            fx, fy, fz, tx, ty, tz = wrist[:6]
            f_norm = float(np.linalg.norm([fx, fy, fz]))
        elif wrist and len(wrist) >= 3:
            fx, fy, fz = wrist[:3]
            tx = ty = tz = 0.0
            f_norm = float(np.linalg.norm([fx, fy, fz]))
        else:
            fx = fy = fz = tx = ty = tz = 0.0
            f_norm = 0.0

        # -------- hand force (log only) --------
        try:
            hd = hand.force_act()
            raw_index = hd[IDX_FINGER_INDEX]
            raw_thumb = hd[IDX_FINGER_THUMB]
            f_index_N = (raw_index * 0.007478) - 0.414
            f_thumb_N = (raw_thumb * 0.012547) + 0.384
        except Exception:
            raw_index = raw_thumb = 0
            f_index_N = f_thumb_N = 0.0

        # -------- wrist-based state machine --------
        if hand_phase == 0:
            if baseline is None:
                baseline = f_norm
            if f_norm < baseline:
                baseline = f_norm
            else:
                baseline += (f_norm - baseline) * BASELINE_ALPHA

            if (f_norm - baseline) > WRIST_CONTACT_SPIKE_N:
                print(">>> [Phase 0->1] Wrist contact detected. Closing hand.")
                apply_speed(hand, 25)
                apply_angles(hand, CLOSE_ANGLES)
                time.sleep(10)
                hand_phase = 1
                stable_start = None

        elif hand_phase == 1:
            hand_phase = 2
            window_samples = []
            window_start = tm
            prev_avg = None
            spike_detected = False

        elif hand_phase == 2:
            window_samples.append(f_norm)

            if (tm - window_start) >= WRIST_WINDOW_S:
                if window_samples:
                    avg_force = float(sum(window_samples) / len(window_samples))
                    if prev_avg is not None:
                        delta = avg_force - prev_avg
                        if not spike_detected:
                            if delta >= WRIST_SPIKE_N:
                                spike_detected = True
                                print(f"    [Phase 2] Spike (+{delta:.3f} N).")
                        else:
                            if delta <= -WRIST_DROP_N:
                                print(
                                    f">>> [Phase 2->3] Drop ({delta:.3f} N). Opening hand."
                                )
                                apply_speed(hand, 1000)
                                apply_angles(hand, DEFAULT_OPEN)
                                hand_phase = 3
                                hand_done_event.set()
                    prev_avg = avg_force

                window_samples = []
                window_start = tm
        elif hand_phase == 3:
            apply_speed(hand, 1000)
            for _ in range(5):
                apply_angles(hand, DEFAULT_OPEN)
                time.sleep(0.5)
            hand_phase = 4  # done, just keep trying to open in case of misses

        # -------- logging --------
        data_log.append(
            [t0, fx, fy, fz, f_norm, tx, ty, tz, f_index_N, f_thumb_N, hand_phase]
        )

        # -------- rate control --------
        elapsed = time.time() - t0
        remaining = sample_period_s - elapsed
        if remaining > 0:
            time.sleep(remaining)

    # cleanup when recording stops
    try:
        apply_angles(hand, DEFAULT_OPEN)
    except Exception:
        pass


try:
    if "robot" not in locals():
        robot = ur5.UR5_Interface()
        robot.start()
        robot.start_ft_sensor(ip_address=ROBOT_IP, poll_rate=100)
        print("UR5 Connected & F/T Sensor Started.")
    else:
        print("Using existing robot connection.")

    # connect hand once, pass into thread
    hand = RH56Hand(port=HAND_PORT, hand_id=HAND_ID)

    recording_active = True
    hand_phase = 0
    hand_done_event.clear()

    recorder_thread = threading.Thread(
        target=background_recorder_with_hand,
        args=(robot, hand),
        kwargs={"sample_period_s": 0.01},  # 100 Hz
        daemon=True,
    )
    recorder_thread.start()

    print("Executing movement sequence...")

    robot.moveL(pose_peg_prep, linSpeed=0.5, linAccel=0.75, asynch=False)
    robot.moveL(pose_peg_go, linSpeed=0.5, linAccel=0.75, asynch=False)
    robot.moveL(pose_peg_n, linSpeed=0.5, linAccel=0.75, asynch=False)

    time.sleep(12)

    robot.moveL(pose_peg_lift, linSpeed=0.5, linAccel=0.75, asynch=False)
    robot.moveL(pose_peg_put, linSpeed=0.5, linAccel=0.75, asynch=False)

    force = np.array([0, 1, -3.75])
    goal_delta = np.array([0, 0, 0.09])
    wrench_goal = np.hstack((force, np.zeros(3)))
    init_cmd = np.hstack((force / -300, np.zeros(3)))
    duration = 5
    max_force = 5.0
    P = 0.0005
    robot.force_position_control(
        wrench=wrench_goal,
        init_cmd=init_cmd,
        goal_delta=goal_delta,
        duration=duration,
        max_force=max_force,
        p=P,
        tolerance=0.015,
    )

    robot.stop()
    robot.moveL(pose_peg_lift, linSpeed=0.5, linAccel=0.75)

finally:
    recording_active = False
    if "recorder_thread" in locals():
        recorder_thread.join(timeout=2.0)

    if len(data_log) > 0:
        print(f"Saving {len(data_log)} samples to {OUTPUT_CSV}...")
        with open(OUTPUT_CSV, mode="w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "timestamp_epoch",
                    "wrist_fx_N",
                    "wrist_fy_N",
                    "wrist_fz_N",
                    "wrist_f_norm_N",
                    "wrist_tx",
                    "wrist_ty",
                    "wrist_tz",
                    "index_force_N",
                    "thumb_force_N",
                    "hand_phase",
                ]
            )
            w.writerows(data_log)
        print("File saved successfully.")
    else:
        print("Warning: No data recorded.")
