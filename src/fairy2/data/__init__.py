"""Data loading for Fairy2 training.

Uses WrinkleFree-CheaperTraining's data infrastructure for:
- Multi-source dataset mixing (DCLM, FineWeb-Edu, GitHub Code, etc.)
- Sequence packing for efficient training
- Influence-based data selection (MobileLLM-R1 style)

Install:
    pip install -e ../WrinkleFree-CheaperTraining

Usage (RECOMMENDED - no config needed):
    from fairy2.data import create_pretraining_dataloader

    train_dl, mixed_dataset, probes = create_pretraining_dataloader(
        tokenizer,
        batch_size=32,
        max_length=2048,
    )

Usage (legacy - with config):
    from fairy2.data import create_dataloader, create_probe_dataloaders

    dataloader, mixed_dataset = create_dataloader(
        config=cfg.data,
        tokenizer=tokenizer,
        batch_size=32,
        max_length=2048,
    )
"""

# Try to import from cheapertraining (required)
try:
    from cheapertraining.data import (
        # High-level API (RECOMMENDED)
        create_pretraining_dataloader,
        load_data_config,
        list_available_configs,
        # Lower-level API
        create_dataloader,
        create_probe_dataloaders,
        get_dataset_weights_from_config,
        MixedDataset,
        PackedDataset,
        DatasetMixture,
        TokenPacker,
        batch_tokenize,
        compute_position_ids,
    )
    from cheapertraining.influence import (
        InfluenceTracker,
        create_influence_tracker,
        MixtureWeightCalculator,
        create_mixture_calculator,
        InfluenceConfig,
    )
    CHEAPERTRAINING_AVAILABLE = True
except ImportError:
    CHEAPERTRAINING_AVAILABLE = False

    # Fail loudly - no silent fallbacks
    def _fail_missing_cheapertraining(*args, **kwargs):
        raise ImportError(
            "cheapertraining is required for Fairy2 training. "
            "Install with: pip install -e ../WrinkleFree-CheaperTraining"
        )

    create_pretraining_dataloader = _fail_missing_cheapertraining
    load_data_config = _fail_missing_cheapertraining
    list_available_configs = _fail_missing_cheapertraining
    create_dataloader = _fail_missing_cheapertraining
    create_probe_dataloaders = _fail_missing_cheapertraining
    get_dataset_weights_from_config = _fail_missing_cheapertraining
    MixedDataset = None
    PackedDataset = None
    DatasetMixture = None
    TokenPacker = None
    batch_tokenize = None
    compute_position_ids = None
    InfluenceTracker = None
    create_influence_tracker = _fail_missing_cheapertraining
    MixtureWeightCalculator = None
    create_mixture_calculator = _fail_missing_cheapertraining
    InfluenceConfig = None


def is_cheapertraining_available() -> bool:
    """Check if cheapertraining is installed."""
    return CHEAPERTRAINING_AVAILABLE


__all__ = [
    # Status check
    "is_cheapertraining_available",
    "CHEAPERTRAINING_AVAILABLE",
    # High-level API (RECOMMENDED)
    "create_pretraining_dataloader",
    "load_data_config",
    "list_available_configs",
    # Lower-level data API
    "create_dataloader",
    "create_probe_dataloaders",
    "get_dataset_weights_from_config",
    # Dataset classes
    "MixedDataset",
    "PackedDataset",
    "DatasetMixture",
    # Packing utilities
    "TokenPacker",
    "batch_tokenize",
    "compute_position_ids",
    # Influence API
    "InfluenceTracker",
    "create_influence_tracker",
    "MixtureWeightCalculator",
    "create_mixture_calculator",
    "InfluenceConfig",
]
