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


# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



class CustomCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 1000 == 0:
                print(f"Step {state.global_step}: memory1: {torch.cuda.memory_reserved(0)/(1<<30)} /{torch.cuda.get_device_properties(0).total_memory/(1<<30)}")
                print(f"Step {state.global_step}: memory2: {torch.cuda.memory_reserved(1) / (1 << 30)} /{torch.cuda.get_device_properties(1).total_memory / (1 << 30)}")



args_dict = dict(
    model_name='mt5-300M',
    
    dataset_name='xnli',
    # languages_list=['fr'],
    languages_list=['en', 'de', 'es', 'fr','hi','tr'],
    use_deepspeed=True,
    num_labels=3,
    deepspeed_path='/ds_xnli.json',
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=6,
    learning_rate=3e-4,
    weight_decay=0.1,
    gradient_checkpointing=True,
    evaluation_strategy="no", 
    gradient_accumulation_steps=4,
    logging_steps=30,
    report_to='tensorboard',
    save_strategy="no",
    bf16=True,
    # 'output_dir': f"./models/{dataset_name}/{model_name}-{dataset_name}",
)
args = argparse.Namespace(**args_dict)
def fine_tune(args):
    torch.cuda.empty_cache()
    load_model_name = get_model_name(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(load_model_name)
    def tokenize_batch(batch):
        return tokenizer(batch["premise"], batch["hypothesis"],  truncation=True, max_length=128)

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # predictions = np.argmax(predictions, axis=1)
        if 'mt5' in args.model_name:
            predictions = np.argmax(predictions[0], axis=-1)
        else: predictions = np.argmax(predictions, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    model = AutoModelForSequenceClassification.from_pretrained(load_model_name, num_labels=3)
    # Tokenize and preprocess the data
    if load_model_name =="tiiuae/falcon-7b":
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    ds_train=data_loader(args.dataset_name,args.languages_list ,'train')
    ds_valid = data_loader(args.dataset_name,args.languages_list ,'validation')
    

    # Apply the function using dataset.map for each split
    # dataset_snips = dataset_snips.map(add_label_column)
    # ds_valid = ds_valid.map(add_label_column)

    tokenized_ds = ds_train.map(tokenize_batch, batched=True)
    tokenized_valid = ds_valid.map(tokenize_batch, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Add the callback to the Trainer
    output_dir = f"./multi_run/{args.dataset_name}/{args.model_name}-{args.dataset_name}"
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
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
    with open(f"./multi_run/{args.dataset_name}/{args.model_name}-{args.dataset_name}/{args.languages_list[0]}.json", "w") as f:
        json.dump(results, f, indent=4)
    return results
    

def eval_test(args, model, tokenizer, trainer):
    all_res = {}
        
    def tokenize_batch(batch):
        return tokenizer(batch["premise"], batch["hypothesis"],  truncation=True, max_length=128)


    for lang in args.languages_list:
        print(lang)
        temp_res = {}

        ds_orig = load_dataset('xnli', lang)
        ds_typo = load_from_disk('./Datasets/noisy_test_sets/xnli_wikitypo_test_sets/' + lang + '_noisy.hf')
        # path = '/Snips_noisy_dataset/'+lang+'_typos'
        # ds_typo = dataset_noisy(path,ds_orig)
    
        df_test = pd.DataFrame(ds_orig['test'])
        # ## for the papers noisy test
        # np.random.seed(10)
        # df_test = df_test[700:]
        # remove_n = 700-len(ds_typo)
        # drop_indices = np.random.choice(df_test.index, remove_n, replace=False)
        # df_test = df_test.drop(drop_indices)

        ds_test = Dataset.from_pandas(df_test, features=ds_orig['train'].features, preserve_index=False)

    
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








