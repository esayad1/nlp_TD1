import pandas as pd

def make_dataset(filename):
    return pd.read_csv(filename)


if __name__ == '__main__':
    df = make_dataset('C:/Users/saids_k/PycharmProjects/nlp_esgi/src/data/raw/train.csv')
    print(df.head())
