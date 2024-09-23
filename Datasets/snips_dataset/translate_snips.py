from seamless_communication.inference import Translator
import torch
import pandas as pd
from datasets import Dataset,DatasetDict
# from seamless_communication.streaming.dataloaders.s2tt import SileroVADSilenceRemover

# Initialize a Translator object with a multitask model, vocoder on the GPU.

model_name = "seamlessM4T_v2_large"
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"

translator = Translator(
    model_name,
    vocoder_name,
    device=torch.device("cuda"),
    dtype=torch.float16,
)
from datasets import load_dataset
snips_ds = load_dataset("benayas/snips")
features_ds=snips_ds['train'].features
from tqdm import tqdm
translated_ds_train = []

for i in tqdm(range(len(snips_ds['train']))):
    text_output, speech_output = translator.predict(
                                 input=snips_ds['train']['text'][i],
                                 task_str="t2tt",
                                 tgt_lang="tur",
                                 src_lang="eng",
                                 )
    translated_ds_train.append(text_output[0])

translated_ds_test = []

for i in tqdm(range(len(snips_ds['test']))):
    text_output, speech_output = translator.predict(
                                 input=snips_ds['test']['text'][i],
                                 task_str="t2tt",
                                 tgt_lang="tur",
                                 src_lang="eng",
                                 )
    translated_ds_test.append(text_output[0])


def create_dataset(translated_test):
    translated_test = [str(text) for text in translated_test]
    if len(translated_test)<2000:
        df_test=pd.DataFrame({'text':translated_test, 'category':snips_ds['test']['category'] })
    else:
        df_test=pd.DataFrame({'text':translated_test, 'category':snips_ds['train']['category'] })    
    df_test['text'] = df_test['text'].str.replace('  ', '')
    df_test['text'] = df_test['text'].str.replace('⁇', '')
    df_test['text'] = df_test['text'].str.replace('¿', '')
    ds_test = Dataset.from_pandas(df_test , features = snips_ds['train'].features,preserve_index=False)
    return ds_test

de_test =create_dataset(translated_ds_test)
de_train =create_dataset(translated_ds_train)

import datasets
snips_de_dataset = datasets.DatasetDict({"train":de_train,"test":de_test})

snips_de_dataset.save_to_disk("snips_dataset/snips_tr.hf")