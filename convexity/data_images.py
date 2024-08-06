from convexity.utils import load_dict
from convexity.models.load_model import get_transform
import torch
from PIL import Image
import os
import numpy as np
from typing import List


def load_data_images(
        path_data: str,
        path_features: str,
        test_data: bool,
        embeddings: bool,
        model_name: str,
        device,
        classes = None,
    ):
    dictionary = load_dict(f"{path_data}synset_words.txt")
    if classes is not None:
        new_dictionary = {}
        for class_name in classes:
            new_dictionary[class_name] = dictionary[class_name]
        dictionary = new_dictionary
    if test_data:
        folder = "val"
    else:
        folder = "train"
    files_list = []
    ids = []
    labels = []
    if embeddings:
        source_folder = f"{path_features}{model_name}/last/"
    else:
        source_folder = f"{path_data}{folder}/"
    for path, subdirs, files in os.walk(source_folder):
        for file in files:
            file_path = path + "/" + file
            if embeddings and file[-3:] != "npy":
                continue
            if not embeddings and file[-3:] == "npy":
                continue
            for id, index in dictionary.items():
                if id in file_path:
                    labels.append(index)
                    files_list.append(file_path)
                    ids.append(file)
                    break
    assert len(labels) == len(files_list)
    datasets = {}
    if test_data:
        if embeddings:
            NotImplementedError(f"Invalid combination of parameters: no embeddings for test data.")
        images_dataset = ImagesDataset(files_list, device, model_name, test_data=True, embeddings=False, model_name=model_name)
        datasets["test"] = images_dataset
    else:
        indices = list(range(len(files_list)))
        ind_val = np.random.choice(indices, size=int(0.01 * len(files_list)), replace=False)
        files_train = [files_list[i] for i in range(len(files_list)) if i not in ind_val]
        labels_train = [labels[i] for i in range(len(files_list)) if i not in ind_val]
        files_val = [files_list[i] for i in ind_val]
        labels_val = [labels[i] for i in ind_val]

        if embeddings:
            datasets["train"] = ImagesDataset(files_train, labels_train, device, test_data=False, embeddings=True, model_name=model_name)
            datasets["val"] = ImagesDataset(files_val, labels_val, device, test_data=False, embeddings=True, model_name=model_name)
        else:
            datasets["train"] = ImagesDataset(files_train, labels_train, device, test_data=False, embeddings=False, model_name=model_name)
            datasets["val"] = ImagesDataset(files_val, labels_val, device, test_data=False, embeddings=False, model_name=model_name)
    return datasets


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 files: List[str],
                 labels: List[int],
                 device,
                 test_data: bool=True,
                 embeddings: bool=False,
                 model_name: str="data2vec",
                 ):
        self.files = files
        self.device = device
        self.labels = labels
        self.test_data = test_data
        self.embeddings = embeddings
        if not embeddings:
            self.transform = get_transform("images", model_name)

    def __getitem__(self, idx):
        item = {}
        if self.embeddings:
            item["input_embeddings"] = np.load(self.files[idx], allow_pickle=True)
        else:
            image = Image.open(self.files[idx]).convert("RGB")
            image = self.transform(image, return_tensors="pt")["pixel_values"][0]
            item["pixel_values"] = image
        if not self.test_data:
            item["label"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.files)
