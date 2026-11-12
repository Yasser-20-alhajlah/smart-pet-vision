"""Record future training source identity without modifying historical run records."""
from pathlib import Path
from .common import save_json
from .provenance import record, file_sha256


def attach_training_provenance(model, cfg, data_yaml, name, task):
    evidence = record(extra={"task": task, "run_name": name,
                             "data_yaml_sha256": file_sha256(Path(data_yaml)),
                             "seed": cfg["seed"]})
    directory = Path(cfg["paths"]["runs"]) / task / name
    path = directory / "source_provenance.json"
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Preserve existing run provenance; choose a new run name: {path}")
    save_json(evidence, path)

    def log_source(trainer):
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            wandb.run.config.update({"source_provenance": evidence,
                                     "source_commit": evidence.get("commit")}, allow_val_change=False)
    model.add_callback("on_train_start", log_source)
    return evidence
