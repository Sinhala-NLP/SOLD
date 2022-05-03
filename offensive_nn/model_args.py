import json
import os
from dataclasses import asdict, field, dataclass


@dataclass
class ModelArgs:

    best_model_dir: str = "outputs/best_model"
    cache_dir: str = "cache_dir/"
    embed_size: int = 300

    early_stopping: bool = True
    early_stopping_min_delta: float = 0.0001
    early_stopping_patience: int = 10

    learning_rate: float = 1e-3

    manual_seed: int = None

    max_features: int = None
    max_len: int = 256

    not_saved_args: list = field(default_factory=list)
    num_classes: int = 2
    num_train_epochs: int = 50

    reduce_lr_on_plateau: bool = True
    reduce_lr_on_plateau_factor: float = 0.6
    reduce_lr_on_plateau_patience: int = 2
    reduce_lr_on_plateau_min_lr: float = 0.0001



    save_best_model: bool = True

    test_batch_size: int = 128
    train_batch_size: int = 128

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    def get_args_for_saving(self):
        args_for_saving = {key: value for key, value in asdict(self).items() if key not in self.not_saved_args}
        return args_for_saving

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            json.dump(self.get_args_for_saving(), f)

    def load(self, input_dir):
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)