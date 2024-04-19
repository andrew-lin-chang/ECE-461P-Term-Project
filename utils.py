from categories import *
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.corpus import CorpusReader, stopwords, product_reviews_2
from nltk.stem import WordNetLemmatizer
import pandas as pd
import string


# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('product_reviews_2')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Stopwords list
stop_words = set(stopwords.words('english'))

def gen_corpus_document(corpus: CorpusReader):
    for i, sentence in enumerate(corpus.sents()):
        tokens = gensim.utils.simple_preprocess(' '.join(sentence))
        yield TaggedDocument(tokens, [i])

TEXT_VECTOR_SIZE = 32
doc2vec = Doc2Vec(vector_size=TEXT_VECTOR_SIZE, epochs=10)
doc2vec.build_vocab(gen_corpus_document(product_reviews_2))


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

def fill_missing_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing item descriptions with empty string
    """

    df = df.assign(item_description=df['item_description'].fillna(''))
    return df

def combine_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine item name, brand, and item description into a single text column
    """

    df = df.assign(text=(df['name'] + ' ' + df['brand_name'].fillna('').astype(str) + ' ' + df['item_description']))
    return df

def clean_text(text: str) -> str:
    """
    Clean up text
    """

    text = text.replace('/', ' or ')
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords and lemmatize
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

def process_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a processed text column
    """

    df = fill_missing_descriptions(df)
    df = combine_text(df)
    df = df.assign(text=df['text'].apply(clean_text))
    return df

def vectorize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorize the text column
    """

    index_list = []
    vector_list = []
    for index, row in df.iterrows():
        index_list.append(index)
        vector = doc2vec.infer_vector(row['text'].split(' '))
        vector_list.append({
            f'vec{i}': vector[i] for i in range(TEXT_VECTOR_SIZE)
        })
    vectors_df = pd.DataFrame.from_records(vector_list, index=index_list)
    df = df.join(vectors_df)
    return df

def mark_brand_boolean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a new column that has 1 if the item has a listed brand, 0 otherwise
    """

    df = df.assign(has_brand=df['brand_name'].notnull().astype(int))
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
    df = pd.get_dummies(df, prefix='cat', columns=['category'], dtype=int)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps
    """

    if 'price' in df.columns:
        df = remove_pricezero(df)
    df = process_text(df)
    df = vectorize_text(df)
    df = mark_brand_boolean(df)
    df = convert_categories(df)
    df = df.drop(['name', 'category_name', 'brand_name', 'item_description', 'text'], axis=1)

    return df

def extract_from_trainset(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the train set into X and y
    """

    X_train = train_df.drop('price', axis=1)
    y_train = train_df['price']
    return X_train, y_train
