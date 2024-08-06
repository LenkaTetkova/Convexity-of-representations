import logging
import random

import evaluate
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import (
    Trainer,
    TrainingArguments,
)
from convexity.data_images import load_data_images
from convexity.utils import path_edit
from convexity.models.load_model import Linear, load_model

log = logging.getLogger(__name__)
print = log.info


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    usual = metric.compute(predictions=predictions, references=labels)
    return {"accuracy": usual}


@hydra.main(config_path="../config", config_name="images.yaml")
def train(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_data = path_edit(cfg.path_data, orig_cwd)
    path_features = path_edit(cfg.path_features, orig_cwd)
    path_model = path_edit(cfg.path_models, orig_cwd)

    model = Linear(cfg.in_channels, 1000)
    if cfg.from_checkpoint:
        model = torch.load(f"{path_model}{cfg.model_name}/{cfg.checkpoint_name}")
    model.to(device)
    print(f"Model is on: {next(model.parameters()).device}")

    if cfg.modality == "images":
        datasets = load_data_images(path_data=path_data,
                                    path_features=path_features,
                                    test_data=False,
                                    embeddings=True,
                                    model_name=cfg.model_name,
                                    device=device,
                                    )
    else:
        raise NotImplementedError(f"Modality {cfg.modality} not implemented.")

    fsdp = ""
    if torch.cuda.device_count() > 1:
        fp16 = True
    else:
        fp16 = False

    training_args = TrainingArguments(
        output_dir=f"{path_model}{cfg.model_name}",
        evaluation_strategy="epoch",
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        log_level="info",
        logging_steps=20,
        save_strategy="epoch",
        seed=cfg.seed,
        dataloader_num_workers=10,
        load_best_model_at_end=True,
        report_to="wandb",
        fsdp=fsdp,
        fp16=fp16,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        compute_metrics=compute_metrics,
    )
    print(f"Model is on: {next(model.parameters()).device}")
    trainer.train()
    trainer.evaluate(datasets["val"])
    model.to("cpu")


    # ######### Evaluate full model ############
    if cfg.modality == "images":
        datasets = load_data_images(path_data=path_data,
                                    path_features=path_features,
                                    test_data=True,
                                    embeddings=False,
                                    model_name=cfg.model_name,
                                    device=device,
                                    )
    model_full = load_model(cfg.modality, cfg.model_name, cfg.n_classes, device, full=True)
    model_full.to(device)
    state_dict = model_full.state_dict()
    state_dict["classifier.weight"] = torch.tensor(model.layer.weight).to(device)
    state_dict["classifier.bias"] = torch.tensor(model.layer.bias).to(device)
    model_full.load_state_dict(state_dict)

    trainer_full = Trainer(
        model=model_full,
        args=training_args,
        eval_dataset=datasets["test"],
        compute_metrics=compute_metrics,
    )
    print(f"Model is on: {next(model_full.parameters()).device}")
    score = trainer_full.evaluate(datasets["test"])
    print(f"Accuracy on validation set ({len(datasets['test'])} images) is {score}.")


if __name__ == "__main__":
    train()
