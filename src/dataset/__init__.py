def load_dataset(cfg, split='train'):
    # Import lazily: mhr_native imports scene.cameras, and importing a
    # submodule also initializes scene/__init__.py.  Keeping this import out
    # of module scope avoids a dataset -> scene -> dataset cycle.
    from .mhr_native import MHRNativeDataset

    dataset_dict = {
        'mhr_native': MHRNativeDataset,
    }
    return dataset_dict[cfg.name](cfg, split)
