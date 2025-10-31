from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gait_controller import DiagonalGaitController, GaitParameters


def test_gait_controller_cycles_diagonal_pairs():
    params = GaitParameters(cycle_time=0.4, step_length=0.04, step_height=0.01, body_height=0.06)
    controller = DiagonalGaitController(params)
    controller.reset()

    dt = params.cycle_time / 10.0
    targets = controller.update(dt)

    assert set(targets.keys()) == {"FL", "FR", "RL", "RR"}
    for foot in targets.values():
        assert foot.shape == (3,)

    first_state = controller.state
    for _ in range(5):
        controller.update(dt)
    second_state = controller.state
    for _ in range(5):
        controller.update(dt)
    third_state = controller.state

    assert second_state != first_state
    assert third_state == first_state
