import re
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import ast
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z\s]', '', text)
    return text


def make_features_1(df, task):

    y = get_output(df, task)

    X = df["video_name"].apply(clean_text)


    return X, y

def make_features(df, task):
    X = None
    y = None
    if task == "is_comic_video":
        X, y = make_features_1(df,task)

    elif task == "is_name":
        X, y = make_feature_2(df,task)

    elif task == "find_comic_name":
        X, y = make_feature_3(df,task)

    return X, y
def string_to_list(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        return []
def make_feature_2(df, task):
    df['is_name'] = df['is_name'].apply(string_to_list)
    df['tokens'] = df['tokens'].apply(string_to_list)


    exploded_data = df.explode(['tokens', 'is_name']).reset_index(drop=True)
    exploded_data['is_final_word'] = False
    exploded_data['is_starting_word'] = False
    exploded_data['is_capitalized'] = False


    sentence_end_punctuations = {'.', '!', '?', '."', '!"', '?"'}
    for i in range(len(exploded_data)) :
        token = exploded_data.at[i, 'tokens']
        # Mark the first word of the dataset as the starting word of a sentence
        if i == 0 :
            exploded_data.at[i, 'is_starting_word'] = True

        else :

            if exploded_data.at[i - 1, 'tokens'] in sentence_end_punctuations :
                exploded_data.at[i, 'is_starting_word'] = True
                exploded_data.at[i - 1, 'is_final_word'] = True

            elif token in sentence_end_punctuations :
                exploded_data.at[i, 'is_final_word'] = True

        if token and token[0].isupper() :
            exploded_data.at[i, 'is_capitalized'] = True

        X = exploded_data[['is_final_word', 'is_starting_word', 'is_capitalized']]  # Features
        y = exploded_data['is_name'].astype(int)

        return X, y

def convert_to_list(value):

    if isinstance(value, list):
        return value

    elif isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)

    else:
        return [value]
def make_feature_3(df, task):
    df['comic_name'] = df['comic_name'].apply(convert_to_list)

    # One-hot encoding pour 'comic_name'
    mlb = MultiLabelBinarizer()
    comic_name_encoded = mlb.fit_transform(df['comic_name'])


    df['tokens'] = df['tokens'].apply(lambda x : ' '.join(ast.literal_eval(x)))

    tfidf = TfidfVectorizer(max_features=100)  # Nous limitons à 100 features pour simplifier le modèle
    tokens_tfidf = tfidf.fit_transform(df['tokens'])

    # Transformer 'is_name' en nombre de noms propres
    df['num_names'] = df['is_name'].apply(lambda x : sum(ast.literal_eval(x)))
    X = pd.concat([pd.DataFrame(tokens_tfidf.toarray()), df['num_names'], df['is_comic']], axis=1)
    y = comic_name_encoded
    X = X.astype(str)

    return X, y

def get_output(df, task):
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"]
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")

    return y
