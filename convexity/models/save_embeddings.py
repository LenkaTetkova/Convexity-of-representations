import logging
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from typing import List

from convexity.utils import path_edit
from convexity.models.load_model import load_model
from convexity.data_images import load_data_images


log = logging.getLogger(__name__)
print = log.info


def save_hidden_states(features: torch.Tensor,
                       files: List[str],
                       path: str) -> None:
    """
    Arguments:
        features: tensor of shape [batch_size, num_layers, num_tokens, num_features]
        files: list of ids (names) of files to save
        path: path to the folder where to save the hidden states
    """
    for j in range(len(files)):
        file_name = files[j].split("/")
        file_name = file_name[-1].split(".")[0]
        tensor_to_save = features[j].numpy()
        np.save(f"{path}{file_name}.npy", tensor_to_save)


@hydra.main(config_path="../config", config_name="images.yaml")
def save_embeddings(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_data = path_edit(cfg.path_data, orig_cwd)
    path_features = path_edit(cfg.path_features, orig_cwd)
    if not os.path.isdir(path_features):
        os.makedirs(path_features)
    if not os.path.isdir(path_features+cfg.model_name):
        os.makedirs(path_features+cfg.model_name)
    model = load_model(cfg.modality, cfg.model_name, cfg.n_classes, device, full=False)
    for param in model.base_model.parameters():
        param.requires_grad = False
    model.to(device)
    print(f"Model is on: {next(model.parameters()).device}")

    datasets = load_data_images(path_data=path_data,
                                path_features=path_features,
                                test_data=cfg.test_data,
                                embeddings=False,
                                model_name=cfg.model_name,
                                device=device,
                                classes=cfg.classes,
                                )
    if cfg.test_data:
        files = datasets["test"].files
        new_path = f"{path_features}{cfg.model_name}/test/"
        dataset_type = "test"
    else:
        files = datasets["train"].files
        new_path = f"{path_features}{cfg.model_name}/train/"
        dataset_type = "train"
    if not os.path.isdir(new_path):
        os.makedirs(new_path)
    for i in range(int(len(files) // cfg.batch_size) + 1):
        batch_files = [files[j] for j in range(i * cfg.batch_size,
                                               min((i + 1) * cfg.batch_size, len(files)))]
        X = [
            datasets[dataset_type][j]["pixel_values"]
            for j in range(i * cfg.batch_size, min((i + 1) * cfg.batch_size, len(files)))
        ]
        if len(X) > 0:
            X = torch.stack(X).to(device)
            out = model.base_model(X, output_hidden_states=True, return_dict=True)
            if -1 in cfg.layers:
                features = out["last_hidden_state"].to("cpu")
                path_last = f"{new_path}last/"
                if not os.path.isdir(path_last):
                    os.makedirs(path_last)
                save_hidden_states(features, batch_files, path_last)
            features = out["hidden_states"]
            features = torch.stack(features).to("cpu")
            features = torch.swapaxes(features, 0, 1)
            save_hidden_states(features, batch_files, new_path)
            print(f"Completed {(i+1)*cfg.batch_size}.")


if __name__ == "__main__":
    save_embeddings()