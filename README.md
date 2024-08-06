# Convexity-of-representations
This repository contains the implementation of convexity scores, a framework to measure convexity of latent representations of neural networks. For more details, please read our paper "On convex decision regions in deep network representations".

## Installation
From repository:
1. Clone the repository
2. Create a new virtual environment with Python 3.9
3. Run the following command from the repository folder:

```shell
pip install -r requirements.txt
```

When the packages are installed, you are ready to perform convexity evaluation.

## Structure of the code
### Arguments
The project is using [Hydra](https://hydra.cc/docs/intro/) configuration. The default arguments are stored in convexity/config/images.yaml. They can be overridden from the command line (e.g., `python3 convexity/models/save_embeddings.py model_name=data2vec n_per_concept=100`). You can also create a new config file in the same folder and override the default one from the command line
`python3 convexity/models/save_embeddings.py --config-name new_config.yaml`

Some of the arguments are described below:
Argument | Description
--- | ---
modality | images (currently supported: images)
model_name | Name of the model (currently supported: data2vec, data2vec_finetuned, data2vec_large, data2vec_large_finetuned, vit, vit_finetuned, vit_large, vit_large_finetuned, beit, beit_finetuned, beit_large, beit_large_finetuned)
n_per_class | Number of data points per concept (0 means all available)
classes | Names or IDs of classes
layers | List of layer names (or IDs) to test
n_neighbors | Number of neighbors
n_paths | Number of paths to sample within each concept/class in the graph convexity evaluation
n_sampled | Number of points to sample on each segment in the Euclidean convexity evaluation
path_features | Path where the features are saved. If it begins with '.', it is interpreted as relative path starting from the woring directory. Otherwise, it is interpreted as absolute path.
path_data | Path where the data are saved. If it begins with '.', it is interpreted as relative path starting from the woring directory. Otherwise, it is interpreted as absolute path.
path_models | Path where the models are saved. If it begins with '.', it is interpreted as relative path starting from the woring directory. Otherwise, it is interpreted as absolute path.
path_outputs | Path where the outputs will be saved. If it begins with '.', it is interpreted as relative path starting from the woring directory. Otherwise, it is interpreted as absolute path.
from_checkpoint | Boolean: whether to load the linear classifier from a saved checkpoint
checkpoint_name | Name of the checkpoint saved in path_models (currently: checkpoint-data2vec, checkpoint-data2vec_large, checkpoint-vit, checkpoint-vit_large) 




## Models used for the experiments in the paper:
### Images:

|Model	|	Hugging Face name |	Accuracy in % |
|-------|-------------------------|--------------|
| data2vec | facebook/data2vec-vision-base |	46.76 |
| data2vec fine-tuned | facebook/data2vec-vision-base-ft1k |	83.59 |
| data2vec large | facebook/data2vec-vision-large | 76.88 |
| data2vec large fine-tuned | facebook/data2vec-vision-large-ft1k | 86.42 |
| ViT	| google/vit-base-patch16-224-in21k | 79.37 |
| ViT fine-tuned | google/vit-base-patch16-224 | 80.33 |
| ViT large | google/vit-large-patch16-224-in21k | 78.63 |
| ViT large fine-tuned | google/vit-large-patch16-224 | 81.99 |


### Text:

|Model	|	Hugging Face name |	Accuracy in % |
|-------|-------------------------|--------------|
| RoBERTa | rasgaard/roberta-newsgroups-probe	|	60	|
| RoBERTa fine-tuned |	rasgaard/roberta-newsgroups-finetuned	|	68.9	|
| BERT | bert-newsgroups-probe	|		|
| BERT fine-tuned | bert-newsgroups-finetuned	|	67.71	|
| distilBERT |	distilbert-newsgroups-probe	|		|
| distilBERT fine-tuned |	distilbert-newsgroups-finetuned	|	67.73	|
| squeeze-BERT |	squeezebert-newsgroups-probe 	|		|
| squeeze-BERT fine-tuned |	squeezebert-newsgroups-finetuned 	|	65.96	|
| LUKE |	luke-base-newsgroups-probe	|	|
| LUKE fine-tuned | luke-base-newsgroups-finetuned	|	66.01	|


### Audio:

|Model	|	Hugging Face name |	Accuracy in % |
|-------|-------------------------|--------------|
| wavLM base |	microsoft/wavlm-base |	94.3 |
| wavLM base fine-tuned	|	|	97.7 |	
| wavLM large |	microsoft/wavlm-large	|	93.0	|
| wavLM large fine-tuned |	|	98.4	|
| wav2vec2 base | facebook/wav2vec2-base	|	 	|
| wav2vec2 base fine-tuned |	|	97.8	|
| wav2vec2 large | facebook/wav2vec2-large	|	 	|
| wav2vec2 large fine-tuned |	|	97.7	|
| HuBERT base | facebook/hubert-base-ls960	|	 92.3	|
| HuBERT base fine-tuned | 	|	97.5 	|
| HuBERT large | facebook/hubert-large-ll60k	|	55.2 	|
| HuBERT large fine-tuned | 	|	98.0 	|


