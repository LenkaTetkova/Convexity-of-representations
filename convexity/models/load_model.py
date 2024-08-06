import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import ImageClassifierOutput
from transformers import (
    BeitForImageClassification,
    BeitModel,
    Data2VecVisionForImageClassification,
    Data2VecVisionModel,
    ViTForImageClassification,
    ViTModel,
    AutoImageProcessor,
    BeitFeatureExtractor,
    ViTImageProcessor
)


class Linear(torch.nn.Module):
    def __init__(self, inp_channels, classes) -> None:
        super(Linear, self).__init__()
        self.inp_channels = inp_channels
        self.classes = classes
        self.layer = torch.nn.Linear(
            in_features=inp_channels,
            out_features=classes,
        )
        self.num_labels = classes

    def forward(self, input_embeddings, labels=None):
        logits = self.layer(input_embeddings)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )


def load_model(modality, model_name, n_classes, device, full=False):
    if modality != "images":
        raise NotImplementedError(f"Model loading for modality {modality} not implemented.")
    if model_name == "data2vec":
        model = Data2VecVisionModel.from_pretrained("facebook/data2vec-vision-base",
                                                    add_pooling_layer=True,
                                                    num_labels=n_classes,
                                                    ignore_mismatched_sizes=True,
                                                    )
    elif model_name == "data2vec_finetuned":
        model = Data2VecVisionForImageClassification.from_pretrained("facebook/data2vec-vision-base-ft1k",
                                                                     num_labels=n_classes,
                                                                     ignore_mismatched_sizes=True,
                                                                     )
        if not full:
            model = model.data2vec_vision
        else:
            model.config.return_dict = False
    elif model_name == "data2vec_large":
        model = Data2VecVisionModel.from_pretrained("facebook/data2vec-vision-large",
                                                    add_pooling_layer=False,
                                                    num_labels=n_classes,
                                                    ignore_mismatched_sizes=True,
                                                    )
    elif model_name == "data2vec_large_finetuned":
        model = Data2VecVisionForImageClassification.from_pretrained("facebook/data2vec-vision-large-ft1k",
                                                                     num_labels=n_classes,
                                                                     ignore_mismatched_sizes=True,
                                                                     )
        if not full:
            model = model.data2vec_vision
        else:
            model.config.return_dict = False
    elif model_name == "vit":
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k",
                                         add_pooling_layer=False,
                                         num_labels=n_classes,
                                         ignore_mismatched_sizes=True,
                                         )
    elif model_name == "vit_finetuned":
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224",
                                                          num_labels=n_classes,
                                                          ignore_mismatched_sizes=True,
                                                          )
        if not full:
            model = model.base_model
        else:
            model.config.return_dict = False
    elif model_name == "vit_large":
        model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k",
                                         add_pooling_layer=False,
                                         num_labels=n_classes,
                                         ignore_mismatched_sizes=True,
                                         )
    elif model_name == "vit_large_finetuned":
        model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224",
                                                          num_labels=n_classes,
                                                          ignore_mismatched_sizes=True,
                                                          )
        if not full:
            model = model.base_model
        else:
            model.config.return_dict = False
    elif model_name == "beit":
        model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k",
                                          num_labels=n_classes,
                                          ignore_mismatched_sizes=True,
                                          )
    elif model_name == "beit_finetuned":
        model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224",
                                                           num_labels=n_classes,
                                                           ignore_mismatched_sizes=True,
                                                           )
        if not full:
            model = model.base_model
        else:
            model.config.return_dict = False
    elif model_name == "beit_large":
        model = BeitModel.from_pretrained("microsoft/beit-large-patch16-224-pt22k",
                                          num_labels=n_classes,
                                          ignore_mismatched_sizes=True,
                                          )
    elif model_name == "beit_large_finetuned":
        model = BeitForImageClassification.from_pretrained("microsoft/beit-large-patch16-224",
                                                           num_labels=n_classes,
                                                           ignore_mismatched_sizes=True,
                                                           )
        if not full:
            model = model.base_model
        else:
            model.config.return_dict = False
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
    model.eval()
    model.to(device)
    return model


def get_transform(modality, model_name):
    if modality != "images":
        raise NotImplementedError(f"Model loading for modality {modality} not implemented.")
    if model_name == "data2vec":
        transform = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
    elif model_name in ["data2vec_finetuned", "data2vec_binary", "data2vec_4", "data2vec_5", "data2vec_6"]:
        transform = BeitFeatureExtractor.from_pretrained("facebook/data2vec-vision-base-ft1k")
    elif model_name == "data2vec_large":
        transform = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-large")
    elif model_name == "data2vec_large_finetuned":
        transform = BeitFeatureExtractor.from_pretrained("facebook/data2vec-vision-large-ft1k")
    elif model_name == "vit":
        transform = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    elif model_name == "vit_finetuned":
        transform = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    elif model_name == "vit_large":
        transform = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
    elif model_name == "vit_large_finetuned":
        transform = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")
    elif model_name == "beit":
        transform = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
    elif model_name == "beit_finetuned":
        transform = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224")
    elif model_name == "beit_large":
        transform = BeitFeatureExtractor.from_pretrained("microsoft/beit-large-patch16-224-pt22k")
    elif model_name == "beit_large_finetuned":
        transform = BeitFeatureExtractor.from_pretrained("microsoft/beit-large-patch16-224")
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
    return transform
