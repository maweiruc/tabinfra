"""Reserved native interface. Fill real imports and mappings when source arrives."""

IMPLEMENTED = False
TASKS = set()
FINETUNE_TASKS = set()


def build(task, model_kwargs):
    raise NotImplementedError("SOMTab is reserved: source is not integrated in this version")


def load_artifact(manifest, artifact_dir, device=None):
    raise NotImplementedError("SOMTab artifact loading is reserved")


def finetune(*, task, X, y, validation_data, model_kwargs, train_kwargs, output_dir):
    raise NotImplementedError("SOMTab finetuning is reserved")
