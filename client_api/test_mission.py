import pytest


mission_module = pytest.importorskip("mission")
MissionManager = mission_module.MissionManager


def test_start_transitions_to_running():
    mission = MissionManager({"enabled": True})

    mission.start()

    snapshot = mission.snapshot()
    assert snapshot["enabled"] is True
    assert snapshot["state"] == "RUNNING"
    assert snapshot["stop_reason"] is None


def test_checkpoint_tags_are_de_duplicated():
    mission = MissionManager({"enabled": True})

    first = mission.consume_tags([20], now_ts=100.0)
    second = mission.consume_tags([20], now_ts=100.4)
    third = mission.consume_tags([20], now_ts=101.6)

    assert first == [20]
    assert second == []
    assert third == [20]


def test_goal_before_checkpoint_is_ignored():
    mission = MissionManager({"enabled": True})
    mission.start()

    consumed = mission.consume_tags([30], now_ts=100.0)

    snapshot = mission.snapshot()
    assert consumed == []
    assert snapshot["checkpoint_seen"] is False
    assert snapshot["goal_seen"] is False
    assert snapshot["state"] == "RUNNING"
    assert snapshot["expected_next_tag"] == 20


def test_checkpoint_then_goal_advances_progress():
    mission = MissionManager({"enabled": True})
    mission.start()

    first = mission.consume_tags([20], now_ts=100.0)
    after_checkpoint = mission.snapshot()
    second = mission.consume_tags([30], now_ts=102.0)
    after_goal = mission.snapshot()

    assert first == [20]
    assert after_checkpoint["checkpoint_seen"] is True
    assert after_checkpoint["state"] == "CHECKPOINT_SEEN"
    assert after_checkpoint["expected_next_tag"] == 30
    assert second == [30]
    assert after_goal["goal_seen"] is True
    assert after_goal["state"] in ("APPROACH_GOAL", "COMPLETE")
    assert after_goal["expected_next_tag"] in (None, 30)


def test_obstacle_threshold_forces_fault_stop():
    mission = MissionManager({"enabled": True})
    mission.start()
    mission.update_obstacle(0.59)

    snapshot = mission.snapshot()
    assert snapshot["state"] == "FAULT_STOP"
    assert snapshot["stop_reason"] == "obstacle"


def test_disabled_mission_is_neutral_and_json_safe():
    mission = MissionManager({"enabled": False})

    mission.start()
    mission.consume_tags([20, 30], now_ts=100.0)
    mission.update_obstacle(0.4)
    snapshot = mission.snapshot()

    assert snapshot["enabled"] is False
    assert snapshot["state"] == "IDLE"
    assert snapshot["stop_reason"] is None
    assert snapshot["expected_next_tag"] is None
    assert snapshot["checkpoint_seen"] is False
    assert snapshot["goal_seen"] is False
