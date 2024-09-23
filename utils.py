
import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict, load_from_disk

def data_loader(name , languages, split):
    values_to_drop=list(set(['en', 'de', 'es', 'fr', 'hi', 'tr'])-set(languages))
    if split == 'train':
        test=pd.read_pickle(f'/train_datasets/{name}_train.dataset')
    elif split == 'validation':
        test = pd.read_pickle(f'/train_datasets/{name}_val.dataset')
    if name == 'snips':
        ds = load_dataset('benayas/snips')
    else:
        ds=load_dataset(name, 'en')
    # Drop rows where column 'B' equals the specified values
    filtered_df = test[~test['language'].isin(values_to_drop)]
    df_ready=filtered_df.drop(columns=['language'])
    ds_s = Dataset.from_pandas(df_ready, features=ds['train'].features, preserve_index=False)
    return ds_s


def get_model_name(model_name):
    models_list = {
    'xlmr' : 'xlm-roberta-base',
    'mbert' : 'google-bert/bert-base-multilingual-cased',
    'falcon7b' : 'tiiuae/falcon-7b',
    'mt5-300M': 'google/mt5-small',
    'mt5-580M': 'google/mt5-base',
    'mt5-1B' : 'google/mt5-large',
    'mt5-3B': 'google/mt5-xl',
    'mt5-13B': 'google/mt5-xxl',
    'bloom7b' : 'bigscience/bloom-7b1'
    }
    return models_list[model_name]