import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from transformers import DataCollatorForTokenClassification, MT5ForConditionalGeneration, AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, TrainerCallback
from datasets import load_dataset, DatasetDict, load_from_disk, Dataset
import pandas as pd
import evaluate
import numpy as np
from utils import data_loader,get_model_name
torch.cuda.empty_cache()
import json
import argparse

def tokenize_and_align_labels(examples,tokenizer):
    # tokenizer = AutoTokenizer.from_pretrained(load_model_name)
    

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


id2label = {
0: "O",
1: 'B-PER',
2: 'I-PER',
3: 'B-ORG',
4: 'I-ORG',
5: 'B-LOC',
6: 'I-LOC',
}
label2id = {
"O": 0,
'B-PER': 1,
'I-PER': 2,
'B-ORG': 3,
'I-ORG': 4,
'B-LOC': 5,
'I-LOC': 6,
}

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



class CustomCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 1000 == 0:
                print(f"Step {state.global_step}: memory1: {torch.cuda.memory_reserved(0)/(1<<30)} /{torch.cuda.get_device_properties(0).total_memory/(1<<30)}")
                print(f"Step {state.global_step}: memory2: {torch.cuda.memory_reserved(1) / (1 << 30)} /{torch.cuda.get_device_properties(1).total_memory / (1 << 30)}")

ds_name = 'train_clean_clean'
label_list = []
args_dict = dict(
    model_name='bloom7b',
    dataset_name='wikiann',
    languages_list=['en' , 'es', 'fr'],
    use_deepspeed=True,
    num_labels=7,
    deepspeed_path='/ds_wikiann.json',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    learning_rate=3e-4,
    weight_decay=0.1,
    gradient_checkpointing=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=4,

    logging_steps=30,
    report_to='tensorboard',
    save_strategy="no",
    bf16=True,
    # 'output_dir': f"./models/{dataset_name}/{model_name}-{dataset_name}",
)
args = argparse.Namespace(**args_dict)
def fine_tune(args):
    
    load_model_name = get_model_name(args.model_name)
    if "bloom7b" in args.model_name:

        tokenizer = AutoTokenizer.from_pretrained(load_model_name,add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(load_model_name)
    if 'mt5' in load_model_name:
        model = MT5ForConditionalGeneration.from_pretrained(load_model_name, num_labels=7, id2label=id2label, label2id=label2id)
    else: model = AutoModelForTokenClassification.from_pretrained(load_model_name, num_labels=7, id2label=id2label, label2id=label2id)
    # Tokenize and preprocess the data
    if load_model_name =="tiiuae/falcon-7b":
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    ds = load_dataset('wikiann','en', split='train')
    # dataset_train = data_loader(args.dataset_name, args.languages_list,'train')
    # label_list=ds.features[f"ner_tags"].feature.names
    dataset_train = pd.read_pickle(f'./Dataset/train_datasets/wikiann_{ds_name}.dataset')
    dataset_train=dataset_train.drop(columns=['language'])
    dataset_train = Dataset.from_pandas(dataset_train, features=ds.features, preserve_index=False)
    dataset_val = data_loader(args.dataset_name, args.languages_list,'validation')
    label_list = dataset_train.features[f"ner_tags"].feature.names

    # Apply the function using dataset.map for each split
    tokenized_dataset = dataset_train.map(lambda example: tokenize_and_align_labels(example,tokenizer), batched=True)
    tokenized_dataset_val = dataset_val.map(lambda example: tokenize_and_align_labels(example,tokenizer), batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    def compute_metrics(p):
        seqeval = evaluate.load("seqeval")
        predictions, labels = p
        predictions=predictions[0]

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    def preprocess_logits_for_metrics(logits, labels):
        if 'mt5' in load_model_name:
            pred_ids = torch.argmax(logits[0], axis=2)
        else: pred_ids = torch.argmax(logits, axis=2)
        # print(pred_ids[0])
        return pred_ids, labels    
    
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
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset_val,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        data_collator= data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    all_res = eval_test(args, model, tokenizer, trainer)
    results = {}
    # print(all_res)
    for lang in all_res:
        results[lang] = {
            'clean_f1': all_res[lang]['clean'].metrics['test_f1'] * 100,
            'noisy_f1': all_res[lang]['noisy'].metrics['test_f1'] * 100,
        }
    with open(f"./multi_run/{args.dataset_name}/{args.model_name}-{args.dataset_name}/results_{ds_name}.json", "w") as f:
        
        json.dump(results, f, indent=4)
    # del model
    # del tokenizer
    
    # del trainer
    # del tokenized_dataset_val
    # del tokenized_dataset

    return results
    
    

def eval_test(args, model, tokenizer, trainer):
    all_res = {}
    load_model_name = get_model_name(args.model_name)
   
    for lang in args.languages_list:
        print(lang)
        temp_res = {}
        # model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased")
        ds_orig = load_dataset('wikiann', lang, split='test')
        # ds_orig = ds_orig.train_test_split(test_size=0.1)['test']
        ds_typo = load_from_disk(f'./Datasets/noisy_test_sets/wikiann_nlpaug_test_sets/{lang}_noisy.hf' )
        label_list = ds_orig.features[f"ner_tags"].feature.names
        tokenized_ds_orig = ds_orig.map(lambda example: tokenize_and_align_labels(example,tokenizer), batched=True)
        tokenized_ds_typo = ds_typo.map(lambda example: tokenize_and_align_labels(example,tokenizer), batched=True)
        # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        with torch.no_grad():
            # print(tokenized_ds_orig)
            res_orig = trainer.predict(tokenized_ds_orig)
            res_typo = trainer.predict(tokenized_ds_typo)
        temp_res['clean'] = res_orig
        temp_res['noisy'] = res_typo
        all_res[lang] = temp_res
    wrong_predictions = {}

    for lang in all_res:
        predictions = all_res[lang]['clean'].predictions[0]
        # print(predictions)
        labels = all_res[lang]['clean'].label_ids
        # print(labels)
        # Get the actual and predicted labels
        true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
        true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]


    return all_res


results = fine_tune(args)







