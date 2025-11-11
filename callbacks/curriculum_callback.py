"""Optional curriculum callback to gradually increase difficulty.

This is a lightweight stub that calls `set_terrain_scale(scale)` on the
environment if available. It does not modify the MuJoCo model; the env method
is a placeholder to be expanded later.
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class TerrainCurriculumCallback(BaseCallback):
    """Linearly ramps a terrain scale parameter during training."""

    def __init__(
        self,
        start_scale: float = 0.1,
        end_scale: float = 1.0,
        total_steps: int = 2_000_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.start_scale = float(start_scale)
        self.end_scale = float(end_scale)
        self.total_steps = int(total_steps)

    def _on_step(self) -> bool:  # type: ignore[override]
        progress = min(1.0, self.num_timesteps / max(1, self.total_steps))
        current_scale = self.start_scale + (self.end_scale - self.start_scale) * progress

        # Update all environments in a VecEnv if method exists
        try:
            self.training_env.env_method("set_terrain_scale", current_scale)  # type: ignore[attr-defined]
        except Exception:
            pass

        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            print(f"[Curriculum] terrain_scale={current_scale:.2f}")
        return True

