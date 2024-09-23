from evaluate import evaluator
from datasets import load_dataset
import json
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification,MT5ForConditionalGeneration

import numpy as np
import evaluate
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding, TrainingArguments , Trainer,DataCollatorForTokenClassification
import torch
torch.cuda.empty_cache()
# languages = ['en']
languages = ['en','es','fr']
# languages = ['en']
model_name = 'bloom-7b'
dataset_name = 'wikiann'
dir = f'/home/s6amalia/models/{dataset_name}'
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1",add_prefix_space=True)
if 'mt5' in model_name:
    model = MT5ForConditionalGeneration.from_pretrained(f"{dir}/{model_name}-{dataset_name}/{model_name}-{dataset_name}.pt",num_labels=7)
else:
    model = AutoModelForTokenClassification.from_pretrained(f"{dir}/{model_name}-{dataset_name}/{model_name}-{dataset_name}.pt",num_labels=7)



def tokenize_and_align_labels(examples):
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
seqeval = evaluate.load("seqeval")
dataset = load_dataset('wikiann', 'en')
label_list = dataset["train"].features[f"ner_tags"].feature.names
def preprocess_logits_for_metrics(logits, labels):
    if 'mt5' in model_name:
        pred_ids = np.argmax(logits[0].to('cpu'), axis=2)
    else: pred_ids = np.argmax(logits.to('cpu'), axis=2)
    # print(pred_ids[0])
    return pred_ids, labels
def compute_metrics(pred):
    predictions, labels = pred
    # if 'mt5' in model_name:
    #     predictions = np.argmax(predictions[0], axis=2)
    # else: predictions = np.argmax(predictions, axis=2)
    # print(predictions)
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
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

all_res = {}

def eval_test():
    for lang in languages:
        print(lang)
        temp_res = {}
        # model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased")
        ds_orig = load_dataset('wikiann', lang, split='test')
        # ds_orig = ds_orig.train_test_split(test_size=0.1)['test']
        ds_typo = load_from_disk(f'/wikiann_nlpaug/{lang}_noisy.hf' )

        tokenized_ds_orig = ds_orig.map(tokenize_and_align_labels, batched=True)
        tokenized_ds_typo = ds_typo.map(tokenize_and_align_labels, batched=True)
        # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        args = TrainingArguments(output_dir=f"{dir}/{model_name}-{dataset_name}/",
                                 per_device_eval_batch_size = 16,
                                 bf16 = True,
                                 # deepspeed = '/ds_wikiann.json',

                                 do_eval =True)
        trainer = Trainer(model=model, data_collator=data_collator, tokenizer=tokenizer,
                          args = args,
                          preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                          compute_metrics=compute_metrics, )
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

        # Find the wrong predictions
        wrong_predictions[lang] = {
        'inds': [],
        'labels': [],
        'predicted_labels': [],
        }
        for i, (actual, predicted) in enumerate(zip(true_labels, true_predictions)):
            if actual != predicted:
                wrong_predictions[lang]['inds'].append(i)
                wrong_predictions[lang]['labels'].append(actual)
                wrong_predictions[lang]['predicted_labels'].append(predicted)

    # Save the wrong predictions to a JSON file
    with open(f"mismatched_rows/wikiann/{model_name}.json", 'w') as f:
        json.dump(wrong_predictions, f)
    return all_res


all_res = eval_test()


for lang in all_res:
    print(f"{lang} clean: {all_res[lang]['clean'].metrics['test_f1'] * 100}")
    print(f"{lang} noisy: {all_res[lang]['noisy'].metrics['test_f1'] * 100}")

results = {}

for lang in all_res:
    results[lang] = {
        'clean_f1': all_res[lang]['clean'].metrics['test_f1'] * 100,
        'noisy_f1': all_res[lang]['noisy'].metrics['test_f1'] * 100,
        'clean_precision': all_res[lang]['clean'].metrics['test_precision'] * 100,
        'noisy_precision': all_res[lang]['noisy'].metrics['test_precision'] * 100,
        'clean_recall': all_res[lang]['clean'].metrics['test_recall'] * 100,
        'noisy_recall': all_res[lang]['noisy'].metrics['test_recall'] * 100,
        'clean_accuracy': all_res[lang]['clean'].metrics['test_accuracy'] * 100,
        'noisy_accuracy': all_res[lang]['noisy'].metrics['test_accuracy'] * 100
    }
import json
# with open(f"{dir}/{model_name}-{dataset_name}/inference_results_{model_name}.json", "w") as f:
#     json.dump(results, f,default=str)