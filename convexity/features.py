import numpy as np
from typing import List, Dict, Tuple
from convexity.utils import get_list_of_files
import pickle
import os


def load_available_features(
    concepts: List[str],
    n_per_concept: int,
    modality: str,
    data_path: str,
    concept_type=None,  # 'detailed_y_0', 'detailed_y_1', 'detailed_y_2',
    # 'y_labels' or 'person_id' ;
    # only applies for human-activity
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Expects the data to be saved in {orig_cwd}/data/{name of the dataset}/concept/image1.png
    Expects the features to be saved in {data_path}/image1.npy
    Returns:
        concept_ids     Dictionary that for each concept contains IDs of the files belonging
                        to this concept
        concept_names   Dictionary that for each concept contains a "nice" name of the concept
                        (can be the same as the key)
    """
    concept_ids = {}
    concept_names = {}
    if modality == "images":
        class_names_imagenet = load_class_names(data_path)
    elif modality == "human-activity":
        concept_path = f"{data_path}/{concept_type}.npy"
        concept_list = np.load(concept_path)
        concepts = remove_sleep_concept(np.unique(concept_list), concept_type)

    for concept in concepts:
        if modality == "images":
            concept_names[concept] = class_names_imagenet[concept]
            path_folder = data_path + concept
            data_files = get_list_of_files(path_folder)
            data_ids = [file.split("/")[-1].split(".")[0] for file in data_files]
            feature_files = get_list_of_files(f"{data_path}/")
            feature_ids = [file.split("/")[-1].split(".")[0] for file in feature_files]

            ids = [id for id in data_ids if id in feature_ids]
        elif modality == "human-activity":
            concept_names[concept] = load_human_activity_names(concept_type, concept)
            ids = np.where(concept_list == concept)[0]
        else:
            raise NotImplementedError(f"Modality {modality} not implemented.")
        if n_per_concept == 0 or len(ids) < n_per_concept:
            concept_ids[concept] = ids
        else:
            concept_ids[concept] = list(np.random.choice(ids, min(n_per_concept, len(ids)), replace=False))
    return concept_ids, concept_names


def remove_sleep_concept(concepts, concept_type):
    new_concepts = []
    for concept in concepts:
        concept_name = load_human_activity_names(concept_type, concept)
        if concept_name != "sleep":
            new_concepts.append(concept)
    return new_concepts


def load_human_activity_names(concept_type, concept):
    """
    Function to load the names of the human activity concepts
    ----------
    Parameters
    ----------
    concept_type: str
        Type of concept to load. Can be 'detailed_y_0', 'detailed_y_1', 'detailed_y_2',
        y_labels' or 'person_id'
    concept: int
        Index of the concept to load
    ----------
    Returns
    ----------
    concept: str
        Name of the concept
    """
    concept_dict = {
        "detailed_y_0": [
            "bicycling",
            "household-chores",
            "manual-work",
            "mixed-activity",
            "sitting",
            "sleep",
            "sports",
            "standing",
            "vehicle",
            "walking",
        ],
        "detailed_y_1": [
            "bicycling",
            "gym",
            "sitstand+activity",
            "sitstand+lowactivity",
            "sitting",
            "sleep",
            "sports",
            "standing",
            "vehicle",
            "walking",
            "walking+activity",
        ],
        "detailed_y_2": [
            "bicycling",
            "sedentary-non-screen",
            "sedentary-screen",
            "sleep",
            "sport-interrupted",
            "sports-continuous",
            "tasks-light",
            "tasks-moderate",
            "vehicle",
            "walking",
        ],
        "y_labels": ["light", "moderate-vigorous", "sedentary", "sleep"],
    }
    if concept_type in concept_dict.keys():
        return concept_dict[concept_type][concept]
    else:
        return concept


def load_class_names(path_data, path_dict="synset_words.txt"):
    dict_names = {}
    with open(path_data + path_dict, "r") as f:
        content = f.readlines()
    for line in content:
        split = line.split(" ")
        name = " ".join(split[1:])
        if name[-1] == "\n":
            name = name[:-1]
        dict_names[split[0]] = name
    return dict_names


def update_results(
    key: str,
    data_to_update: List[float],
    model_name: str,
    path_outputs: str,
    layers: List[str],
    is_range: bool = False,
    n: int = 10,
) -> None:
    if n == 10:
        neigh = ""
    else:
        neigh = f"_{n}"
    if is_range:
        file_path = f"{path_outputs}results_{model_name}_range{neigh}.pkl"
    else:
        file_path = f"{path_outputs}results_{model_name}{neigh}.pkl"
    if os.path.isfile(file_path):
        with open(file_path, "rb") as fp:
            results = pickle.load(fp)
    else:
        results = {
            "layers": layers,
        }
    results[key] = data_to_update
    with open(file_path, "wb") as fp:
        pickle.dump(results, fp)
    return

