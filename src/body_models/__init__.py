from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig

if TYPE_CHECKING:
    from src.body_models.base import BodyModel


def get_body_model(cfg: DictConfig) -> BodyModel:
    """Factory for the `BodyModel` backends, mirroring the get_rigid_deform/
    get_texture/get_pose_correction dispatch pattern in src/models/. Imports
    are lazy so this module stays importable without pulling in `smplx` or
    the MHR rig unless the corresponding backend is actually requested."""
    name = cfg.name
    if name == "smplx":
        from src.body_models.smplx_body_model import SMPLXBodyModel

        return SMPLXBodyModel(cfg)
    if name == "mhr":
        from src.body_models.mhr_body_model import MHRBodyModel

        return MHRBodyModel(cfg)
    raise ValueError(f"Unknown body_model.name: {name!r}")
