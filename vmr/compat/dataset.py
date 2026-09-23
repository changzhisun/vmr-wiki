"""Legacy config-to-dataset adapter."""

from vmr.datasets.manifest import load_dataset, select_split


def dataset_context(cfg: dict, split: str | None = None, *, evaluation: bool = False):
    from vmr.compat.config import dataset_path

    directory = dataset_path(cfg, "datasets")
    metadata = load_dataset(directory, cfg["dataset"]["name"])
    selected = select_split(
        metadata,
        split if split is not None else cfg["dataset"].get("split"),
        evaluation=evaluation,
    )
    return directory, metadata, selected
