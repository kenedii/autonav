import torch

from car import CarClient
from models import (
    default_invert_steering,
    default_throttle_output_scale,
    infer_control_architecture,
    infer_control_model_layout,
    infer_control_output_dim,
)


def _sequential_resnet34_state(output_dim=2):
    state = {}
    for layer_idx, blocks in ((4, 3), (5, 4), (6, 6), (7, 3)):
        for block_idx in range(blocks):
            state[f"0.{layer_idx}.{block_idx}.conv1.weight"] = torch.zeros(1)
    state["1.8.weight"] = torch.zeros(output_dim, 128)
    state["1.8.bias"] = torch.zeros(output_dim)
    return state


def _named_resnet18_state(output_dim=1):
    state = {}
    for layer_idx, blocks in ((4, 2), (5, 2), (6, 2), (7, 2)):
        for block_idx in range(blocks):
            state[f"features.{layer_idx}.{block_idx}.conv1.weight"] = torch.zeros(1)
    state["head.6.weight"] = torch.zeros(output_dim, 128)
    state["head.6.bias"] = torch.zeros(output_dim)
    return state


def test_infer_autonav_v2_checkpoint_shape():
    state = _sequential_resnet34_state(output_dim=2)

    assert infer_control_model_layout(state) == "sequential"
    assert infer_control_output_dim(state) == 2
    assert infer_control_architecture(state) == "resnet34"


def test_infer_legacy_single_output_checkpoint_shape():
    state = _named_resnet18_state(output_dim=1)

    assert infer_control_model_layout(state) == "named"
    assert infer_control_output_dim(state) == 1
    assert infer_control_architecture(state) == "resnet18"


def test_autonav_defaults_match_pretrained_checkpoint_path():
    model_path = "checkpoints/AutoNav-v2/AutoNav-v2-34/AutoNav-v2-34.pth"

    assert default_invert_steering(model_path) is True
    assert default_throttle_output_scale(model_path, None) == 3.33


def test_throttle_to_pulse_width_supports_reverse_and_forward():
    client = CarClient.__new__(CarClient)
    client.THROTTLE_CENTER = 1500
    client.THROTTLE_MAX = 1900
    client.THROTTLE_MIN = 1200

    assert client._throttle_to_pulse_width(1.0) == 1900
    assert client._throttle_to_pulse_width(-1.0) == 1200


def test_extract_control_prediction_accepts_structured_output():
    steer, throttle = CarClient._extract_control_prediction(
        {"steering": -0.25, "throttle": 0.4}
    )

    assert steer == -0.25
    assert throttle == 0.4
