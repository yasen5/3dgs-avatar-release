from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig

if TYPE_CHECKING:
    from src.dataset.mhr_native import MHRNativeDataset


def load_dataset(cfg: DictConfig, split: str = 'train') -> MHRNativeDataset:
    # Import lazily: mhr_native imports scene.cameras, and importing a
    # submodule also initializes scene/__init__.py.  Keeping this import out
    # of module scope avoids a dataset -> scene -> dataset cycle.
    from .mhr_native import MHRNativeDataset

    dataset_dict = {
        'mhr_native': MHRNativeDataset,
    }
    return dataset_dict[cfg.name](cfg, split)
