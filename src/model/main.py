from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier



def make_model():
    return Pipeline([
        ("count_vectorizer", CountVectorizer()),
        ("random_forest", RandomForestClassifier()),
    ])

stop_words_list = stopwords.words('english') + stopwords.words('french')

def make_model2(task):
    if task == "is_comic_video":
       return make_imb_pipeline(
           TfidfVectorizer(max_features=5000, stop_words=stop_words_list),
           SMOTE(random_state=42),
           LogisticRegression(max_iter=1000, random_state=42)
       )
    elif task == "is_name":
        return make_imb_pipeline(
            RandomForestClassifier(random_state=42)
        )
    elif task == "find_comic_name":
        return make_imb_pipeline(
            RandomForestClassifier(random_state=42)
        )


def make_model4():
    return make_imb_pipeline(
        RandomForestClassifier(random_state=42)
    )





