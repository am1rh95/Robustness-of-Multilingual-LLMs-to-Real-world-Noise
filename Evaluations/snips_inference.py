
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding,AutoTokenizer,AutoModelForSequenceClassification
from datasets import load_dataset,DatasetDict,load_from_disk,Dataset
import evaluate
import torch
import nltk
import random
import numpy as np
import pandas as pd
label2id = {
    "AddToPlaylist": 0,
    'BookRestaurant': 1,
    'GetWeather': 2,
    'PlayMusic': 3,
    'RateBook': 4,
    'SearchCreativeWork': 5,
    'SearchScreeningEvent': 6,
}

def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True)


def compute_metrics(eval_preds):
    metric2 = evaluate.load("accuracy")
    logits, labels = eval_preds
    #     print(logits[1].shape)
    if 'mt5' in model_name:
        predictions = np.argmax(logits[0], axis=-1)
    else:
        predictions = np.argmax(logits, axis=-1)  ####### for T5 model
    return {'m2': metric2.compute(predictions=predictions, references=labels)}


def add_label_column(example):
    example['label'] = label2id[example['category']]
    return example

# languages = ['en']
languages = ['en','de','es','fr','hi','tr']
model_name = 'mt5-3B'

dir = '/home/s6amalia/models/snips_same_config'
tokenizer = AutoTokenizer.from_pretrained(f"{dir}/{model_name}-snips/{model_name}-snips.tk")
model = AutoModelForSequenceClassification.from_pretrained(f"{dir}/{model_name}-snips/{model_name}-snips.pt")
all_res = {}

mismatched_rows={}
# languages = ['en']
def eval_test():
    for lang in languages:
        print(lang)
        temp_res = {}
        if lang == 'en':
            ds_orig = load_dataset('benayas/snips')
        else:
            ds_orig = load_from_disk("/snips_dataset/snips_" + lang + ".hf")

        # path = '/Snips_noisy_dataset/'+lang+'_typos'
        # ds_typo = dataset_noisy(path,ds_orig)
        ds_typo = load_from_disk(f"/snips_WikiTypos/{lang}_noisy.hf")
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

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        trainer = Trainer(model=model, data_collator=data_collator, tokenizer=tokenizer,
                          compute_metrics=compute_metrics, )

        with torch.no_grad():
            res_orig = trainer.predict(tokenized_ds_orig)
            res_typo = trainer.predict(tokenized_ds_typo)
        temp_res['clean'] = res_orig
        temp_res['noisy'] = res_typo
        all_res[lang] = temp_res

        mismatched_rows[lang] = []
        for i in range(len(res_orig.predictions[0] if 'mt5' in model_name else res_orig.predictions)):
            orig_label = int(ds_test[i]['label'])
            typo_label = int(ds_typo[i]['label'])
            # print(len(res_typo.predictions))
            if np.argmax(res_typo.predictions[0][i] if 'mt5' in model_name else res_typo.predictions[i]) != typo_label:
                mismatched_rows[lang].append({
                    'index': i,
                    'correct_text': ds_test[i]['text'],
                    'incorrect_text': ds_typo[i]['text'],
                    'correct_label': orig_label,
                    'incorrect_label': np.argmax(res_typo.predictions[0][i] if 'mt5' in model_name else res_typo.predictions[i])
                })

    return all_res, mismatched_rows

all_res, mismatched_rows = eval_test()
import json
with open(f"mismatched_rows/snips/{model_name}.json", "w") as f:
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
# import json
# with open(f"{dir}/{model_name}-snips/inference_results_{model_name}.json", "w") as f:
#     json.dump(results, f,default=str)