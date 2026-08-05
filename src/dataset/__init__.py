from __future__ import annotations

from typing import TYPE_CHECKING, Union

from omegaconf import DictConfig

if TYPE_CHECKING:
    from src.dataset.mhr_native import MHRNativeDataset
    from src.dataset.neuman_smplx import NeumanSMPLXDataset


def load_dataset(cfg: DictConfig, split: str = 'train') -> Union['MHRNativeDataset', 'NeumanSMPLXDataset']:
    # Imports lazy: mhr_native imports scene.cameras, and importing a
    # submodule also initializes scene/__init__.py.  Keeping this import out
    # of module scope avoids a dataset -> scene -> dataset cycle.
    from .mhr_native import MHRNativeDataset
    from .neuman_smplx import NeumanSMPLXDataset

    dataset_dict = {
        'mhr_native': MHRNativeDataset,
        'neuman_smplx': NeumanSMPLXDataset,
    }
    return dataset_dict[cfg.name](cfg, split)
