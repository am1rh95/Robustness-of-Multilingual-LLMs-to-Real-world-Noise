from datasets import load_dataset, DatasetDict, load_from_disk, Dataset
import pandas as pd

import spacy
from collections import Counter
import pandas as pd


all_df=[]
languages = ['en','es','de','fr']  # Add the 6 languages here


def get_pos_tags(row):
    # Concatenate 'premise' and 'hypothesis'
    sentence = df_test.loc[row.name, 'premise'] + ' ' + df_test.loc[row.name, 'hypothesis']
    # Tokenize the sentence into words
    # words = sentence.split()

    # Get POS tags for each word
    pos_tags = [token.pos_ for token in nlp(sentence)]

    return pos_tags

for lang in languages:
    print(lang)
    if lang=='en':
        nlp = spacy.load('en_core_web_sm')
    else: nlp = spacy.load(f'{lang}_core_news_sm')
    ds_orig = load_dataset('xnli', lang) 

    df_test = pd.DataFrame(ds_orig['test'])
    df_test['pos_tags'] = df_test.apply(get_pos_tags, axis=1)
    pos_tags_list = sum(df_test['pos_tags'].tolist(), [])
    
    pos_counts = Counter(pos_tags_list)
    df_pos_counts = pd.DataFrame({'POS': pos_counts.keys(), f'Count_{lang}': pos_counts.values()})
    all_df.append(df_pos_counts)

from functools import reduce

# Assuming dfs is your list of dataframes
merged_df = reduce(lambda left,right: pd.merge(left,right,on='POS', how='outer'), all_df)

# Fill in missing values with 0
merged_df = merged_df.fillna(0)

print(merged_df)