from categories import *
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_dataset(environment: str='local', file: str=None) -> pd.DataFrame:
    """
    Load a dataset
    """

    if environment == 'colab':
        from google.colab import drive
        drive.mount('/content/drive')
    
        file = f'/content/drive/My Drive/ECE 461P Project/{file}'
    elif environment == 'kaggle':
        import os
        for dirname, _, filenames in os.walk('/kaggle/input'):
            for filename in filenames:
                print(os.path.join(dirname, filename))
        # TODO: Extract and set BASE_PATH
    
    if file == 'train.tsv':
        df = pd.read_csv(file, sep='\t', index_col='train_id')
    elif file in ['test.tsv', 'test_stg2.tsv']:
        df = pd.read_csv(file, sep='\t', index_col='test_id')
    
    return df

def remove_pricezero(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where the price is listed as 0 USD
    """

    return train_df[train_df['price'] > 0]

def mark_brand_boolean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a new column that has 1 if the item has a listed brand, 0 otherwise
    """

    df = df.assign(has_brand=df['brand_name'].notnull().astype(int))
    df = df.drop('brand_name', axis=1)
    return df

def search_category_map(category_str: str, category_map: dict=CATEGORY_MAP) -> str:
    """
    Given a category string (three increasingly refined categories separated by slashes), return a general category
    """

    if not isinstance(category_str, str):
        return None

    cats = category_str.split('/', 2)

    cur_map = category_map
    for cat in cats:
        cur_map = cur_map.get(cat)
        if not isinstance(cur_map, dict):
            return cur_map

    raise ValueError()

def convert_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a DataFrame's category string into a one-hot encoding of general categories
    """

    df = df.assign(category=df['category_name'].apply(lambda cat_str: search_category_map(cat_str)))
    df = df.drop('category_name', axis=1)
    df = pd.get_dummies(df, prefix='cat', columns=['category'], dtype=int)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps
    """

    if 'price' in df.columns:
        df = remove_pricezero(df)
    df = mark_brand_boolean(df)
    df = convert_categories(df)

    return df

def extract_from_trainset(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the train set into X and y
    """

    X_train = train_df.drop('price', axis=1)
    y_train = train_df['price']
    return X_train, y_train
