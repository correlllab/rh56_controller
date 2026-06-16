"""Hardware-free core helpers for the RH56 ROS driver.

This module deliberately avoids importing rclpy or custom ROS messages.  The
ROS node can use it for serial ownership, state conversion, and command routing,
while tests can exercise the same behavior with fake hands.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, MutableMapping, Protocol


RIGHT = "right"
LEFT = "left"
HAND_ORDER = (RIGHT, LEFT)
HAND_ID_TO_SIDE = {1: RIGHT, 2: LEFT}
SIDE_TO_HAND_ID = {RIGHT: 1, LEFT: 2}
JOINT_NAMES = ("pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation")
DOF_PER_HAND = 6
TOTAL_DOF = 12


class HandLike(Protocol):
    hand_id: int
    force_limits: list[int]

    def angle_read(self) -> list[int] | None: ...
    def force_act(self) -> list[int] | None: ...
    def current_read(self) -> list[int] | None: ...
    def temp_read(self) -> list[int] | None: ...
    def status_read(self) -> list[int] | None: ...
    def angle_set(self, angles: list[int]): ...
    def speed_set(self, speeds: list[int]): ...
    def force_set(self, thresholds: list[int]): ...
    def current_limit_set(self, limits: list[int]): ...
    def clear_errors(self): ...
    def save_parameters(self): ...
    def gesture_force_clb(self, direction: int): ...
    def adaptive_force_control_iter(self, **kwargs): ...


class HandReadError(RuntimeError):
    """Raised when a required hand state read fails."""


@dataclass(frozen=True)
class HandSnapshot:
    side: str
    hand_id: int
    angles_raw: list[int]
    forces_raw: list[int]
    force_limits_raw: list[int]
    currents_raw: list[int]
    temperatures: list[int]
    status_raw: list[int]


def parse_hand_ids(value) -> tuple[int, ...]:
    """Parse ROS-friendly hand id config into a validated tuple.

    Accepts values such as ``"1,2"``, ``"1"``, ``[1, 2]``, or ``1``.
    Only hand IDs 1 and 2 are supported by this driver.
    """
    if value is None:
        ids: Iterable[int | str] = (1, 2)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("hand_ids cannot be empty")
        ids = [part.strip() for part in stripped.replace(";", ",").split(",")]
    elif isinstance(value, int):
        ids = (value,)
    else:
        ids = value

    parsed: list[int] = []
    for item in ids:
        hand_id = int(item)
        if hand_id not in HAND_ID_TO_SIDE:
            raise ValueError(f"unsupported hand_id {hand_id}; expected 1 and/or 2")
        if hand_id in parsed:
            raise ValueError(f"duplicate hand_id {hand_id}")
        parsed.append(hand_id)
    if not parsed:
        raise ValueError("at least one hand_id is required")
    return tuple(parsed)


def side_for_hand_id(hand_id: int) -> str:
    try:
        return HAND_ID_TO_SIDE[int(hand_id)]
    except KeyError as exc:
        raise ValueError(f"unsupported hand_id {hand_id}") from exc


def clamp_raw_values(values: Iterable[float | int], *, minimum: int = 0, maximum: int = 1000) -> list[int]:
    clamped: list[int] = []
    for value in values:
        clamped.append(max(minimum, min(maximum, int(round(float(value))))))
    if len(clamped) != DOF_PER_HAND:
        raise ValueError(f"expected {DOF_PER_HAND} values, got {len(clamped)}")
    return clamped


def raw_to_rad(raw: float | int) -> float:
    return (float(raw) / 1000.0) * math.pi


def rad_to_raw(rad: float) -> int:
    return max(0, min(1000, int(round((float(rad) / math.pi) * 1000.0))))


def command_q_to_raw_by_side(q_values: Iterable[float], enabled_sides: Iterable[str]) -> dict[str, list[int]]:
    values = list(q_values)
    if len(values) != TOTAL_DOF:
        raise ValueError(f"expected {TOTAL_DOF} command values, got {len(values)}")

    enabled = set(enabled_sides)
    raw_by_side: dict[str, list[int]] = {}
    for index, side in enumerate(HAND_ORDER):
        if side not in enabled:
            continue
        start = index * DOF_PER_HAND
        raw_by_side[side] = [rad_to_raw(q) for q in values[start:start + DOF_PER_HAND]]
    return raw_by_side


def _disabled_record(side: str, joint: str) -> dict[str, float | int | str]:
    return {
        "side": side,
        "joint": joint,
        "mode": -1,
        "q": 0.0,
        "dq": 0.0,
        "ddq": 0.0,
        "tau": 0.0,
        "tau_lim": 0.0,
        "current": 0.0,
        "temperature": 0.0,
        "q_raw": -1.0,
        "dq_raw": 0.0,
        "tau_raw": 0.0,
        "tau_lim_raw": 0.0,
        "status_raw": -1.0,
    }


def motor_state_records(states_by_side: Mapping[str, HandSnapshot]) -> list[dict[str, float | int | str]]:
    """Return 12 message-shaped records in right[6] + left[6] order."""
    records: list[dict[str, float | int | str]] = []
    for side in HAND_ORDER:
        snap = states_by_side.get(side)
        if snap is None:
            records.extend(_disabled_record(side, joint) for joint in JOINT_NAMES)
            continue
        for i, joint in enumerate(JOINT_NAMES):
            angle = snap.angles_raw[i]
            force = snap.forces_raw[i]
            limit = snap.force_limits_raw[i]
            records.append({
                "side": side,
                "joint": joint,
                "mode": 0,
                "q": raw_to_rad(angle),
                "dq": 0.0,
                "ddq": 0.0,
                "tau": float(force),
                "tau_lim": float(limit),
                "current": float(snap.currents_raw[i]),
                "temperature": float(snap.temperatures[i]),
                "q_raw": float(angle),
                "dq_raw": 0.0,
                "tau_raw": float(force),
                "tau_lim_raw": float(limit),
                "status_raw": float(snap.status_raw[i]),
            })
    return records


class SerialHandManager:
    """Owns RH56 serial hand objects and serializes all hardware access."""

    def __init__(
        self,
        serial_port: str,
        hand_ids=(1, 2),
        hand_factory: Callable[[str, int], HandLike] | None = None,
    ) -> None:
        self.serial_port = serial_port
        self.hand_ids = parse_hand_ids(hand_ids)
        self._lock = threading.RLock()

        if hand_factory is None:
            from .rh56_hand import RH56Hand

            hand_factory = lambda port, hand_id: RH56Hand(port=port, hand_id=hand_id)

        self._hands: MutableMapping[str, HandLike] = {}
        for hand_id in self.hand_ids:
            self._hands[side_for_hand_id(hand_id)] = hand_factory(serial_port, hand_id)

    @property
    def enabled_sides(self) -> tuple[str, ...]:
        return tuple(side for side in HAND_ORDER if side in self._hands)

    def hand(self, side: str) -> HandLike:
        try:
            return self._hands[side]
        except KeyError as exc:
            raise ValueError(f"{side} hand is not enabled") from exc

    def read_states(self) -> tuple[dict[str, HandSnapshot], list[str]]:
        states: dict[str, HandSnapshot] = {}
        errors: list[str] = []
        with self._lock:
            for side, hand in self._hands.items():
                try:
                    states[side] = self._read_one(side, hand)
                except HandReadError as exc:
                    errors.append(str(exc))
        return states, errors

    def _read_one(self, side: str, hand: HandLike) -> HandSnapshot:
        angles = hand.angle_read()
        forces = hand.force_act()
        if angles is None:
            raise HandReadError(f"{side}: angle_read failed")
        if forces is None:
            raise HandReadError(f"{side}: force_act failed")

        currents = hand.current_read() if hasattr(hand, "current_read") else None
        temps = hand.temp_read() if hasattr(hand, "temp_read") else None
        status = hand.status_read() if hasattr(hand, "status_read") else None

        return HandSnapshot(
            side=side,
            hand_id=hand.hand_id,
            angles_raw=clamp_raw_values(angles, minimum=-1, maximum=1000),
            forces_raw=clamp_raw_values(forces, minimum=-32768, maximum=32767),
            force_limits_raw=clamp_raw_values(getattr(hand, "force_limits", [1000] * DOF_PER_HAND)),
            currents_raw=clamp_raw_values(currents or [0] * DOF_PER_HAND, minimum=-32768, maximum=32767),
            temperatures=clamp_raw_values(temps or [0] * DOF_PER_HAND, minimum=-32768, maximum=32767),
            status_raw=clamp_raw_values(status or [0] * DOF_PER_HAND, minimum=-32768, maximum=32767),
        )

    def set_angles_raw(self, side: str, values: Iterable[float | int]):
        angles = clamp_raw_values(values, minimum=-1, maximum=1000)
        with self._lock:
            return self.hand(side).angle_set(angles)

    def set_speeds_raw(self, side: str, values: Iterable[float | int]):
        speeds = clamp_raw_values(values)
        with self._lock:
            return self.hand(side).speed_set(speeds)

    def set_force_limits_raw(self, side: str, values: Iterable[float | int]):
        limits = clamp_raw_values(values)
        with self._lock:
            return self.hand(side).force_set(limits)

    def set_current_limits_raw(self, side: str, values: Iterable[float | int]):
        limits = clamp_raw_values(values, minimum=0, maximum=1500)
        with self._lock:
            return self.hand(side).current_limit_set(limits)

    def apply_angle_commands_rad(self, q_values: Iterable[float]) -> None:
        raw_by_side = command_q_to_raw_by_side(q_values, self.enabled_sides)
        with self._lock:
            for side in self.enabled_sides:
                angles = raw_by_side.get(side)
                if angles is not None:
                    self.hand(side).angle_set(angles)

    def apply_to_sides(self, hand_selector: str, operation: Callable[[str], None]) -> tuple[bool, str]:
        selector = hand_selector.lower().strip()
        if selector == "both":
            sides = self.enabled_sides
        elif selector in HAND_ORDER:
            sides = (selector,)
        else:
            return False, f"invalid hand selector: {hand_selector}"

        missing = [side for side in sides if side not in self._hands]
        if missing:
            return False, f"requested hand(s) not enabled: {', '.join(missing)}"

        with self._lock:
            for side in sides:
                operation(side)
        return True, ", ".join(sides)

    def clear_errors(self, side: str):
        with self._lock:
            return self.hand(side).clear_errors()

    def save_parameters(self, side: str):
        with self._lock:
            return self.hand(side).save_parameters()

    def calibrate_force_sensors(self, side: str, direction: int = 1):
        with self._lock:
            return self.hand(side).gesture_force_clb(direction)

    def adaptive_force_control_iter(self, side: str, **kwargs):
        with self._lock:
            yield from self.hand(side).adaptive_force_control_iter(**kwargs)
