import math

import pytest

from rh56_controller.hand_ros_core import (
    LEFT,
    RIGHT,
    SerialHandManager,
    command_q_to_raw_by_side,
    motor_state_records,
    parse_hand_ids,
)


class FakeHand:
    def __init__(self, port: str, hand_id: int):
        self.port = port
        self.hand_id = hand_id
        self.force_limits = [1000] * 6
        self.angle_commands = []
        self.speed_commands = []
        self.force_commands = []
        self.current_limit_commands = []
        self.clear_count = 0
        self.save_count = 0
        self.calibrate_count = 0

    def angle_read(self):
        return [1000] * 6 if self.hand_id == 1 else [500] * 6

    def force_act(self):
        return [10 * self.hand_id + i for i in range(6)]

    def current_read(self):
        return [100 + i for i in range(6)]

    def temp_read(self):
        return [30 + i for i in range(6)]

    def status_read(self):
        return [i for i in range(6)]

    def angle_set(self, angles):
        self.angle_commands.append(list(angles))
        return [1]

    def speed_set(self, speeds):
        self.speed_commands.append(list(speeds))
        return [1]

    def force_set(self, thresholds):
        self.force_commands.append(list(thresholds))
        self.force_limits = list(thresholds)
        return [1]

    def current_limit_set(self, limits):
        self.current_limit_commands.append(list(limits))
        return [1]

    def clear_errors(self):
        self.clear_count += 1
        return [1]

    def save_parameters(self):
        self.save_count += 1
        return [1]

    def gesture_force_clb(self, _direction):
        self.calibrate_count += 1
        return [1]

    def adaptive_force_control_iter(self, **_kwargs):
        yield {"forces": [1] * 6, "angles": [2] * 6}
        yield {"done": True, "final_forces": [3] * 6, "final_angles": [4] * 6}


def fake_factory(port, hand_id):
    return FakeHand(port, hand_id)


def test_parse_hand_ids_accepts_ros_friendly_forms():
    assert parse_hand_ids("1,2") == (1, 2)
    assert parse_hand_ids("2") == (2,)
    assert parse_hand_ids([1]) == (1,)
    assert parse_hand_ids(1) == (1,)


@pytest.mark.parametrize("bad_value", ["", "3", "1,1"])
def test_parse_hand_ids_rejects_invalid_values(bad_value):
    with pytest.raises(ValueError):
        parse_hand_ids(bad_value)


def test_one_enabled_hand_publishes_disabled_records_for_other_side():
    manager = SerialHandManager("/dev/fake", hand_ids="1", hand_factory=fake_factory)

    states, errors = manager.read_states()
    records = motor_state_records(states)

    assert errors == []
    assert len(records) == 12
    assert records[0]["side"] == RIGHT
    assert records[0]["mode"] == 0
    assert records[0]["q_raw"] == 1000.0
    assert records[0]["q"] == pytest.approx(math.pi)
    assert records[6]["side"] == LEFT
    assert records[6]["mode"] == -1
    assert records[6]["q_raw"] == -1.0


def test_motor_command_conversion_ignores_disabled_hand():
    q_values = [math.pi] * 6 + [0.0] * 6

    raw_by_side = command_q_to_raw_by_side(q_values, enabled_sides=(RIGHT,))

    assert raw_by_side == {RIGHT: [1000] * 6}


def test_manager_applies_commands_only_to_enabled_sides():
    manager = SerialHandManager("/dev/fake", hand_ids="1", hand_factory=fake_factory)

    manager.apply_angle_commands_rad([math.pi] * 6 + [0.0] * 6)

    assert manager.hand(RIGHT).angle_commands == [[1000] * 6]
    with pytest.raises(ValueError):
        manager.hand(LEFT)


def test_both_selector_means_all_enabled_hands():
    manager = SerialHandManager("/dev/fake", hand_ids="1", hand_factory=fake_factory)

    ok, message = manager.apply_to_sides(
        "both",
        lambda side: manager.set_force_limits_raw(side, [900] * 6),
    )

    assert ok is True
    assert message == RIGHT
    assert manager.hand(RIGHT).force_commands == [[900] * 6]


def test_disabled_specific_hand_request_fails_cleanly():
    manager = SerialHandManager("/dev/fake", hand_ids="1", hand_factory=fake_factory)

    ok, message = manager.apply_to_sides("left", lambda side: manager.clear_errors(side))

    assert ok is False
    assert "not enabled" in message


def test_current_limits_allow_firmware_range_to_1500():
    manager = SerialHandManager("/dev/fake", hand_ids="1", hand_factory=fake_factory)

    manager.set_current_limits_raw(RIGHT, [0, 250, 500, 1000, 1250, 1500])

    assert manager.hand(RIGHT).current_limit_commands == [[0, 250, 500, 1000, 1250, 1500]]
