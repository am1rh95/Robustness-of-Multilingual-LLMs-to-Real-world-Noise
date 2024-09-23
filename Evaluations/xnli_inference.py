import numpy as np
import evaluate
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, TrainingArguments , Trainer
import torch

languages = ['en','de','es','fr','hi', 'tr']
model_name = 'mbert'
dataset_name = 'xnli'
dir = f'/home/s6amalia/models/{dataset_name}'
tokenizer = AutoTokenizer.from_pretrained(f"{dir}/{model_name}-{dataset_name}/{model_name}-{dataset_name}.tk")
model = AutoModelForSequenceClassification.from_pretrained(f"{dir}/{model_name}-{dataset_name}/{model_name}-{dataset_name}.pt").to("cuda")


def tokenize_batch(batch):
    return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=128)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_preds):
    #     metric1 = evaluate.load("f1")
    metric2 = evaluate.load("accuracy")
    logits, labels = eval_preds
    #     print(logits[1].shape)
    if 'mt5' in model_name:
        predictions = np.argmax(logits[0], axis=-1)
    else:
        predictions = np.argmax(logits, axis=-1)
    # predictions = np.argmax(logits[0], axis=-1)   ####### for T5 model
    return {'m2': metric2.compute(predictions=predictions, references=labels)}


all_res = {}




def eval_test():
    mismatched_rows = {}
    for lang in languages:
        print(lang)
        temp_res = {}
        ds_orig = load_dataset('xnli', lang, split='test')
        ds_typo = load_from_disk('/xnli_WikiTypos/' + lang + '_noisy.hf')
        tokenized_ds_orig = ds_orig.map(tokenize_batch, batched=True)
        tokenized_ds_typo = ds_typo.map(tokenize_batch, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        args = TrainingArguments(output_dir=f"{dir}/{model_name}-{dataset_name}/",
                                per_device_eval_batch_size = 32,
                                gradient_accumulation_steps=1,
                                bf16= True,
                                do_eval =True)
        trainer = Trainer(model=model, data_collator=data_collator, tokenizer=tokenizer,
                          compute_metrics=compute_metrics, args=args )
        with torch.no_grad():
            res_orig = trainer.predict(tokenized_ds_orig)
            res_typo = trainer.predict(tokenized_ds_typo)
        temp_res['clean'] = res_orig
        temp_res['noisy'] = res_typo
        all_res[lang] = temp_res

        # Find mismatched rows
        mismatched_rows[lang] = []
        for i in range(len(res_orig.predictions[0] if 'mt5' in model_name else res_orig.predictions )):
            orig_label = int(ds_orig[i]['label'])
            typo_label = int(ds_typo[i]['label'])
            if  np.argmax(res_typo.predictions[0][i] if 'mt5' in model_name else res_typo.predictions[i]) != orig_label:
                mismatched_rows[lang].append({
                    'index': i,
                    'correct_text': [ds_orig[i]['premise'],ds_orig[i]['hypothesis']],
                    'incorrect_text': [ds_typo[i]['premise'],ds_typo[i]['hypothesis']],
                    'correct_label': orig_label,
                    'clean_label': np.argmax(res_orig.predictions[0][i] if 'mt5' in model_name else res_orig.predictions[i]),
                    'incorrect_label': np.argmax(res_typo.predictions[0][i] if 'mt5' in model_name else res_typo.predictions[i])
                })

    return all_res, mismatched_rows

all_res, mismatched_rows = eval_test()
import json
with open(f"mismatched_rows/xnli/{model_name}.json", "w") as f:
    json.dump(mismatched_rows, f,default=str)

for lang in all_res:
    print(f"{lang} clean: {all_res[lang]['clean'].metrics['test_m2']['accuracy']*100}")
    print(f"{lang} noisy: {all_res[lang]['noisy'].metrics['test_m2']['accuracy']*100}")



results = {}

for lang in all_res:
    results[lang] = {
        'clean_accuracy': all_res[lang]['clean'].metrics['test_m2']['accuracy'] * 100,
        'noisy_accuracy': all_res[lang]['noisy'].metrics['test_m2']['accuracy'] * 100
    }
import json
# with open(f"{dir}/{model_name}-{dataset_name}/inference_results_{model_name}.json", "w") as f:
#     json.dump(results, f,default=str)