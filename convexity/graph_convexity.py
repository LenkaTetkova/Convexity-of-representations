import itertools
import os
import random
from typing import Dict, List, Tuple
from scipy.spatial.distance import cdist

import hydra
import numpy as np
import pandas
import pandas as pd
import torch
from igraph import Graph
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf

from convexity.utils import sem_adjusted, path_edit, load_available_features


def nearest_neighbors_one(id: int,
                          n_neigh: int,
                          df: pandas.DataFrame,
                          df_dist: pandas.DataFrame,
                          df_inc: pandas.DataFrame):
    """
    Finds nearest neighbors for given id.
    Arguments:
        id: index
        n_neigh: number of nearest neighbors
        df: distance matrix
        df_dist: distance matrix of only n_neigh nearest neighbors saved, other distances are 0
        df_inc: incidence matrix of df_dist
    """
    distances = df.loc[id]
    shortest = distances.nsmallest(n=n_neigh + 1)
    for key, val in shortest.items():
        if val > 0:
            df_inc.loc[key, id] = 1
            df_dist.loc[key, id] = val


def get_nearest_neighbors(
    df: pandas.DataFrame,
    n: int,
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Find n nearest neighbors. For each node, its nearest neigbors are in rows of
    df (distances) and df_inc (incidence).
    Arguments:
        df:     Distances between each pair of nodes
        n:      Number of nearest neighbors
    Returns:
        df_dist:    Only distances within n nearest neighbors are kept, the rest
                    is changed to 0
        df_inc:     Incidence matrix
    """
    df_inc = pd.DataFrame(0, index=df.index, columns=df.columns)
    df_dist = pd.DataFrame(0, index=df.index, columns=df.columns)
    Parallel(n_jobs=1, verbose=10, require="sharedmem")(
        delayed(nearest_neighbors_one)(id, n, df, df_dist, df_inc) for id in df.index
    )
    return df_dist, df_inc


def get_concept_idx(
    columns_names: List[str],
    concept_ids: Dict[str, List[int]],
) -> Dict[str, List[int]]:
    """
    Arguments:
        columns_names   List of names (id) of the data
        concept_ids     Dictionary of ids of all the datapoints belonging
                        to each concept
    Returns:
        concepts        Dictionary of ids from columns_names belonging to each concept
    """
    concepts = {}
    for idx, name in enumerate(columns_names):
        for key, value in concept_ids.items():
            if name in value:
                concept = key
                if concept in concepts.keys():
                    concepts[concept].append(idx)
                else:
                    concepts[concept] = [idx]
                break
    return concepts


def is_path_in_concept(
    shortest_path: List[int],
    indices: List[int],
) -> Tuple[bool, float]:
    """
    Compute the proportion of the path that is within the concept.
    Arguments:
        shortest_path:  list of all vertices on the path
        indices:        list of all vertices belonging to the concept
    Returns:
        is_whole:       bool indicating whether the path is fully inside the concept
        prop:           the proportion of the path that is inside the concept
    """
    if len(shortest_path) <= 2:
        prop = 1
    else:
        length = 0
        outside = 0
        for idx in shortest_path[1:-1]:
            length += 1
            if idx not in indices:
                outside += 1
        prop = (length - outside) / length
    return prop


def compute_paths(graph: Graph,
                  concept: str,
                  indices: List[int],
                  n_paths: int) -> List[float]:
    """
    Arguments:
        graph: graph with datapoints in vertices and weighted edges are Euclidean distances to
               the n closest nearest neighbors
        concept: name of the conceptto test
        indices: indices of vertices belonging to the concept
        n_paths: maximum number of paths
    Returns:
        proportion: list of proportion of each path inside the concept
    """
    all_paths = list(itertools.permutations(indices, r=2))
    n_paths_max = min(len(all_paths), n_paths)
    sampled_indices = np.random.choice(list(range(len(all_paths))), n_paths_max, replace=False)
    sampled_paths = [all_paths[index] for index in sampled_indices]
    proportion = []
    path_exists = []
    for id1, id2 in sampled_paths:
        shortest_path = graph.get_shortest_paths(id1, to=id2, weights=graph.es["weight"], output="vpath")
        if len(shortest_path[0]) == 0:
            exists = False
            proportion.append(0)
        else:
            prop = is_path_in_concept(shortest_path[0], indices)
            proportion.append(prop)
            exists = True
        path_exists.append(exists)
        path_exists.append(exists)
    print(
        f"Class {concept}: "
        f"{'{:.2f}'.format(np.sum(path_exists) / len(path_exists) * 100)}% "
        f"of paths exist. "
        f"Graph convexity is {'{:.2f}'.format(np.mean(proportion) * 100)}%."
    )
    return proportion


@hydra.main(config_path="./config", config_name="images.yaml")
def graph_convexity(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    path_features = f"{path_edit(cfg.path_features, orig_cwd)}{cfg.model_name}/test"
    path_outputs = path_edit(cfg.path_outputs, orig_cwd)
    path_csv = f"{path_outputs}{cfg.model_name}_graph_convexity"

    if not os.path.isdir(path_outputs):
        os.makedirs(path_outputs)

    features, class_ids = load_available_features(path_features, cfg.classes, cfg.n_per_class)
    features_pooled = {}
    for key, value in features.items():
        if "vit" in cfg.model_name:     # Use the classifier token only
            features_pooled[key] = value[:, :, 0, :]
        else:          # Average the rest of the tokens
            features_pooled[key] = np.mean(value[:, :, 1:, :], 2)
    features = features_pooled
    all_ids = []
    for value in class_ids.values():
        all_ids.extend(value)
    df_proportion = pandas.DataFrame(columns=list(class_ids.keys()))
    df_sem = pandas.DataFrame(columns=list(class_ids.keys()))
    results = {}
    results_sem = {}
    for layer in cfg.layers:
        print(f"Layer {layer}")
        results[layer] = []
        results_sem[layer] = []
        # Get features for the layer
        features_layer = []
        for value in features.values():
            features_layer.append(value[:, layer])
        features_layer = np.concatenate(features_layer, axis=0)

        # Create graph
        dist_matrix = cdist(features_layer, features_layer, metric="euclidean")
        distances = pd.DataFrame(dist_matrix, columns=all_ids, index=all_ids)
        distances_nearest, _ = get_nearest_neighbors(distances, cfg.n_neighbors)
        graph_matrix = distances_nearest.to_numpy().astype(float)
        symmetric = np.maximum(graph_matrix, graph_matrix.T)
        graph = Graph.Weighted_Adjacency(symmetric)

        # Get indices of the vertices in the graph for the class ids
        class_indices_graph = get_concept_idx(distances.columns, class_ids)
        for concept, ids in class_indices_graph.items():
            proportion = compute_paths(graph, concept, ids, cfg.n_paths)
            mean = round(np.mean(proportion) * 100, 2)
            sem = sem_adjusted(proportion, len(ids)) * 100
            df_proportion.loc[layer, concept] = mean
            df_sem.loc[layer, concept] = sem
            results[layer].append(mean)
            results_sem[layer].append(sem)
    pandas.DataFrame.to_csv(
        df_sem,
        f"{path_csv}_sem.csv",
    )
    pandas.DataFrame.to_csv(
        df_proportion,
        f"{path_csv}.csv",
    )
    print("In total:")
    for layer in cfg.layers:
        print(f"Layer {layer} -- Mean: {np.mean(results[layer])}, "
              f"SEM adjusted: {np.mean(results_sem[layer])}.")


if __name__ == "__main__":
    graph_convexity()
