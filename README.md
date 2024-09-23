# Exploring the Robustness of Multilingual LLMs to Real-World Noisy Data

## Overview

This project focuses on fine-tuning and evaluating various language models on different datasets in multiple languages.

## Directory Structure

- `Datasets/`: Contains all the datasets used for training and evaluation.
- `Evaluations/`: Contains scripts and notebooks for evaluating the models.
- `Fine_tuning/`: Contains scripts for fine-tuning the models.
- `WikiTypo/`: Screpts for creating the WikiTypo noise collection and the collection of typos for six languages.
- `results/`: Results of the clean and noisy test sets for the three XNLI, WikiANN, and Snips datasets.

## WikiTypo
The **WikiTypo** dataset is a collection of noisy data for six languages (en, de, es, fr, hi, tr). The resulting noise collection could be found in the `WikiTypo/` directory. 
To add other languages use the notebook `WikiTypo/WikiTypo class.ipynb` and follow the instructions in the notebook.

## Fine-tuning
- For fine-tuning the models we used DeepSpeed and Huggingface Transformers. To change the DeepSpeed configuration, use the `ds_[dataset_name].json` file in the `Fine_tuning/` directory.
- Change the parameters in the args dictionary in the fine-tuning scripts to fine-tune the models on different datasets and languages.
- The list of the models could be found in the `utils.py` file.

- For multi_GPU training use : `deepspeed --num_gpus=4 /fine_tune_[Task_name].py` .
- Otherwise, remove the deepspeed path and then use : `python fine_tune_[Task_name].py` .

## Evaluations
For evaluation use the `evaluation_[Task_name].py` scripts. make sure to change the path to the model and the dataset in the script!

## Results

## Contributing

## License