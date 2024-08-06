import itertools
import logging
import os
import random

import hydra
import numpy as np
import pandas
import torch
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Union

from convexity.models.load_model import load_model, Linear
from convexity.utils import (path_edit, sem_adjusted,
                             load_available_features, load_dict)


log = logging.getLogger(__name__)
print = log.info


def sample_on_segment(feat1: torch.Tensor,
                      feat2: torch.Tensor,
                      n_sampled: int) -> List[torch.Tensor]:
    new_points = []
    lambda_ = 1 / (n_sampled + 1)
    for i in range(n_sampled):
        new_lambda = (i + 1) * lambda_
        new_points.append(new_lambda * feat1 + (1 - new_lambda) * feat2)
    return new_points


def forward_from_middle(
        hidden_states: torch.Tensor,
        model: torch.nn.Module,
        layer: int,
    ) -> torch.Tensor:
    for i, layer_module in enumerate(model.encoder.layer):
        if i < layer:
            continue
        layer_outputs = layer_module(hidden_states)

        hidden_states = layer_outputs[0]
    hidden_states = model.layernorm(hidden_states)
    if model.pooler is not None:
        hidden_states = model.pooler(hidden_states)
    return hidden_states


def predict_segment(points: torch.Tensor,
                    model: torch.nn.Module,
                    device: torch.device,
                    classification_layer: torch.nn.Module,
                    layer: int,
                    pooler_output: bool=False) -> np.ndarray:
    if not pooler_output:
        embeddings = forward_from_middle(points, model, layer)
    else:
        embeddings = points
    classification_layer = classification_layer.to(device)
    logits = classification_layer(embeddings)
    classification_layer = classification_layer.to(device)
    predictions = torch.argmax(logits.logits, dim=-1).to("cpu").numpy()
    return predictions


def euclidean_one_concept(
        features: torch.Tensor,
        indices: List[int],
        label_id: int,
        model: torch.nn.Module,
        device: torch.device,
        classification_layer: torch.nn.Module,
        layer: int,
        n_paths:int =5000,
        n_sampled: int=10,
        pooler_output: bool=False,
    ) -> List:
    features = torch.from_numpy(features).to(device)
    scores = []
    all_paths = list(itertools.combinations(indices, r=2))
    n_paths_max = min(len(all_paths), n_paths)
    sampled_indices = np.random.choice(list(range(len(all_paths))), n_paths_max, replace=False)
    sampled_paths = [all_paths[index] for index in sampled_indices]
    for id1, id2 in sampled_paths:
        new_points = sample_on_segment(features[id1], features[id2], n_sampled)
        predictions = predict_segment(
            torch.stack(new_points), model, device, classification_layer, layer, pooler_output
        )
        is_correct = [pred == label_id for pred in predictions]
        n_correct = sum(is_correct) / len(is_correct)
        scores.append(n_correct)
    features = features.to("cpu")
    return scores


def run_analysis(
        features_dict: Dict[str, np.ndarray],
        cfg: Dict[str, Union[str,int,List[str],List[int],bool,float]],
        path_outputs: str,
        model: torch.nn.Module,
        device: torch.device,
        labels_concepts: Dict[str, int],
        classification_layer: torch.nn.Module,
    ) -> None:
    file_name_mean = f"{path_outputs}_euclidean.csv"
    file_name_sem = f"{path_outputs}_euclidean_sem.csv"
    if os.path.exists(file_name_mean):
        df_all_layers = pandas.read_csv(
            file_name_mean,
            index_col=0,
        )
    else:
        df_all_layers = pandas.DataFrame()
    if os.path.exists(file_name_sem):
        df_sem_layers = pandas.read_csv(
            file_name_sem,
            index_col=0,
        )
    else:
        df_sem_layers = pandas.DataFrame()

    concepts_filtered = {}
    for concept_name, features in features_dict.items():
        if concept_name not in df_all_layers.columns:
            concepts_filtered[concept_name] = features
    print(f"Evaluating {len(concepts_filtered)} out of {len(features_dict)} concepts.")
    results_euclidean = {}
    results_euclidean_sem = {}
    for layer in cfg.layers:
        results_euclidean[layer] = []
        results_euclidean_sem[layer] = []
    for concept_name, features_concept in concepts_filtered.items():
        label_id = labels_concepts[concept_name]
        indices = list(range(len(features_concept)))
        for layer in cfg.layers:
            results = euclidean_one_concept(features_concept[:, layer, :, :],
                                            indices,
                                            label_id,
                                            model,
                                            device,
                                            classification_layer,
                                            layer=layer,
                                            n_paths=cfg.n_paths,
                                            n_sampled=cfg.n_sampled,
                                            pooler_output=False,
                                            )
            mean = np.mean(results) * 100
            sem = sem_adjusted(results, len(indices)) * 100
            df_all_layers.loc[layer, concept_name] = mean
            df_sem_layers.loc[layer, concept_name] = sem
            results_euclidean[layer].append(mean)
            results_euclidean_sem[layer].append(sem)
            print(f"Layer {layer} -- Mean: {mean}, sem_adjusted: {sem} "
                  f"({len(results)} paths).")
        pandas.DataFrame.to_csv(
            df_sem_layers,
            file_name_sem,
        )
        pandas.DataFrame.to_csv(
            df_all_layers,
            file_name_mean,
        )
    print("In total:")
    for layer in cfg.layers:
        print(f"Layer {layer} -- Mean: {np.mean(results_euclidean[layer])}, "
              f"SEM adjusted: {np.mean(results_euclidean_sem[layer])}.")
    return


@hydra.main(config_path="./config", config_name="images.yaml")
def euclidean_convexity(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    path_data = path_edit(cfg.path_data, orig_cwd)
    path_features = f"{path_edit(cfg.path_features, orig_cwd)}{cfg.model_name}/test"
    path_outputs = path_edit(cfg.path_outputs, orig_cwd)
    if not os.path.isdir(path_outputs):
        os.makedirs(path_outputs)

    model_dir = path_edit(cfg.path_models, orig_cwd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    model = load_model(cfg.modality, cfg.model_name, cfg.n_classes, device, full=True)
    for param in model.base_model.parameters():
        param.requires_grad = False
    model.to(device)
    print(f"Model is on: {next(model.parameters()).device}")
    classification_layer = Linear(cfg.in_channels, 1000)
    if cfg.from_checkpoint:
        checkpoint = torch.load(
            f"{model_dir}{cfg.model_name}/{cfg.checkpoint_name}/pytorch_model.bin", map_location=device
        )
        classification_layer.load_state_dict(checkpoint)
    else:
        try:
            state_dict = classification_layer.state_dict()
            state_dict_full = model.state_dict()
            state_dict["layer.weight"] = state_dict_full["classifier.weight"].to(device)
            state_dict["layer.bias"] = state_dict_full["classifier.bias"].to(device)
            classification_layer.load_state_dict(state_dict)
        except KeyError:
            print("No weights found for classifier, random initialization!")
    if "finetuned" in cfg.model_name:
        model = model.base_model

    features, classes_ids = load_available_features(path_features, cfg.classes, cfg.n_per_class)
    dictionary = load_dict(f"{path_data}synset_words.txt")

    print("Euclidean convexity")

    run_analysis(
        features,
        cfg,
        path_outputs+cfg.model_name,
        model,
        device,
        dictionary,
        classification_layer,
    )


if __name__ == "__main__":
    euclidean_convexity()