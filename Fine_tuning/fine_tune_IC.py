import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback ,AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset, DatasetDict, load_from_disk, Dataset
import pandas as pd
import evaluate
import numpy as np
from utils import data_loader,get_model_name
torch.cuda.empty_cache()
import json
import argparse
import os

label2id = {
    "AddToPlaylist": 0,
    'BookRestaurant': 1,
    'GetWeather': 2,
    'PlayMusic': 3,
    'RateBook': 4,
    'SearchCreativeWork': 5,
    'SearchScreeningEvent': 6,
}

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



class CustomCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 1000 == 0:
                print(f"Step {state.global_step}: memory1: {torch.cuda.memory_reserved(0)/(1<<30)} /{torch.cuda.get_device_properties(0).total_memory/(1<<30)}")
                print(f"Step {state.global_step}: memory2: {torch.cuda.memory_reserved(1) / (1 << 30)} /{torch.cuda.get_device_properties(1).total_memory / (1 << 30)}")

def add_label_column(example):
    example['label'] = label2id[example['category']]
    return example

parser = argparse.ArgumentParser(description='Process some integers.')

# Add an argument
parser.add_argument('--run', type=int, help='Description of the argument')
parser.add_argument('--model', type=str, help='Description of the argument')
parser.add_argument('--local_rank', type=int)

# Parse the arguments
args_f = parser.parse_args()
args_dict = dict(
    model_name=args_f.model,
    dataset_name='snips',
    # languages_list=['en', 'de', 'es', 'fr',],
    languages_list=['en', 'de', 'es', 'fr','hi','tr'],
    use_deepspeed=True,
    num_labels=7,
    deepspeed_path='ds_snips.json',
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    learning_rate=1e-4,
    weight_decay=0.1, 
    gradient_checkpointing=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=1,
    logging_steps=30,
    report_to='tensorboard',
    save_strategy="no",
    bf16=True,
    run= args_f.run,
    # 'output_dir': f"./models/{dataset_name}/{model_name}-{dataset_name}",
)
args = argparse.Namespace(**args_dict)
def fine_tune(args):
    torch.cuda.empty_cache()
    load_model_name = get_model_name(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(load_model_name)
    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True)

    def compute_metrics(eval_pred):
        metric = evaluate.load("accuracy")
        predictions, labels = eval_pred
        if 'mt5' in args.model_name:
            predictions = np.argmax(predictions[0], axis=-1)
        else:
            predictions = np.argmax(predictions, axis=-1)
        # 
        return metric.compute(predictions=predictions, references=labels)
    model = AutoModelForSequenceClassification.from_pretrained(load_model_name, num_labels=7)
    # Tokenize and preprocess the data
    if load_model_name =="tiiuae/falcon-7b":
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    dataset_snips=data_loader(args.dataset_name,args.languages_list ,'train')
    ds_train_valid = dataset_snips.train_test_split(test_size=0.05)
    ds_valid = ds_train_valid['test']
    dataset_snips = ds_train_valid['train']

    # Apply the function using dataset.map for each split
    dataset_snips = dataset_snips.map(add_label_column)
    ds_valid = ds_valid.map(add_label_column)

    tokenized_ds = dataset_snips.map(tokenize_batch, batched=True)
    tokenized_valid = ds_valid.map(tokenize_batch, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Add the callback to the Trainer
    output_dir = f"./multi_run2/{args.dataset_name}/{args.model_name}-{args.dataset_name}"
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        deepspeed=args.deepspeed_path if args.use_deepspeed else None,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        evaluation_strategy=args.evaluation_strategy,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        # use_reentrant=True,
        # eval_accumulation_steps=1,
        logging_dir=f"{output_dir}",
        report_to='tensorboard',
        save_strategy=args.save_strategy,
        bf16 = args.bf16,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        eval_dataset=tokenized_valid,
        compute_metrics=compute_metrics,
        data_collator= data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    all_res = eval_test(args, model, tokenizer, trainer)
    results = {}
    # print(all_res)
    for lang in all_res:
        results[lang] = {
            'clean_accuracy': all_res[lang]['clean'].metrics['test_accuracy'] * 100,
            'noisy_accuracy': all_res[lang]['noisy'].metrics['test_accuracy'] * 100
        }
    

    file_name = f"./multi_run2/{args.dataset_name}/{args.model_name}-{args.dataset_name}/results{args.run}.json"
    # i = 1

    # # Check if the file exists
    # while os.path.exists(file_name):
    #     # If the file exists, increment the file name
    #     base, ext = os.path.splitext(file_name)
    #     file_name = f"{base}_{i}{ext}"
    #     i += 1

    # Save the results to the file
    with open(file_name, "w") as f:
        json.dump(results, f, indent=4)
    return results
    

def eval_test(args, model, tokenizer, trainer):
    all_res = {}
        
    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True)

    for lang in args.languages_list:
        print(lang)
        temp_res = {}
        if lang == 'en':
            ds_orig = load_dataset('benayas/snips')
        else:
            ds_orig = load_from_disk("./Datasets/snips_dataset/snips_" + lang + ".hf")
        
        # path = '/Snips_noisy_dataset/'+lang+'_typos'
        # ds_typo = dataset_noisy(path,ds_orig)
        ds_typo = load_from_disk(f"./Datasets/noisy_test_sets/snips_wikitypo_test_sets/{lang}_noisy.hf")
        df_test = pd.DataFrame(ds_orig['test'])
        # ## for the papers noisy test
        # np.random.seed(10)
        # df_test = df_test[700:]
        # remove_n = 700-len(ds_typo)
        # drop_indices = np.random.choice(df_test.index, remove_n, replace=False)
        # df_test = df_test.drop(drop_indices)

        ds_test = Dataset.from_pandas(df_test, features=ds_orig['train'].features, preserve_index=False)

        ds_test = ds_test.map(add_label_column)
        ds_typo = ds_typo.map(add_label_column)

        tokenized_ds_orig = ds_test.map(tokenize_batch, batched=True)
        tokenized_ds_typo = ds_typo.map(tokenize_batch, batched=True)

        # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # trainer = Trainer(model=model, data_collator=data_collator, tokenizer=tokenizer,
        #                   compute_metrics=compute_metrics, )

        with torch.no_grad():
            res_orig = trainer.predict(tokenized_ds_orig)
            res_typo = trainer.predict(tokenized_ds_typo)
        temp_res['clean'] = res_orig
        temp_res['noisy'] = res_typo
        all_res[lang] = temp_res

    return all_res


results = fine_tune(args)
  






