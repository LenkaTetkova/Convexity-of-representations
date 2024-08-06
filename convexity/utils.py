from typing import Dict, List, Tuple
import os
import numpy as np
import json


def path_edit(path: str, orig_cwd: str) -> str:
    if path[0] == ".":
        return orig_cwd + path[1:]
    else:
        return path


def load_dict(mypath: str) -> Dict[str, int]:
    dictionary = {}
    with open(mypath, "r") as f:
        content = f.readlines()
    for i, line in enumerate(content):
        id = line.split(" ")[0]
        dictionary[id] = i
    return dictionary


def get_list_of_files(dir):
    files_list = []
    for path, subdirs, files in os.walk(dir):
        for file in files:
            files_list.append(path + "/" + file)
    return files_list


def sem_adjusted(data, n_samples):
    result = np.std(data, ddof=1) / np.sqrt(n_samples)
    return result


def load_available_features(path_features: str,
                            classes: List[str],
                            n_per_class: int) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    Loads at most n_per_class features for each class.
    Arguments:
        path_features: path to where the features are stored
        classes: list of the names of classes
        n_per_class: maximum number of data per class
    Returns:
        all_features: dictionary of features for each class
        classes_existing: dictionary of ids loaded for each class
    """
    with open(f"{path_features}/labels.json", 'r') as f:
        labels = json.load(f)
    all_features = {}
    classes_existing = {}
    for class_name in classes:
        if class_name not in labels:
            print(f"{class_name} is not in the labels")
        else:
            possible_files = labels[class_name]
            files_existing = []
            for file in possible_files:
                complete_path = os.path.join(path_features, file+".npy")
                if os.path.exists(complete_path):
                    files_existing.append((file, complete_path))
            if len(files_existing) > n_per_class:
                indices = np.random.choice(list(range(len(files_existing))), n_per_class, replace=False)
                files = [files_existing[ind] for ind in indices]
                files_existing = files
            loaded_features = []
            loaded_ids = []
            for file, complete_path in files_existing:
                features = np.load(complete_path)
                loaded_features.append(features)
                loaded_ids.append(file)
            if len(loaded_ids) > 0:
                classes_existing[class_name] = loaded_ids
                all_features[class_name] = np.stack(loaded_features)
    return all_features, classes_existing
