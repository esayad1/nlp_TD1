import re



def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z\s]', '', text)
    return text





def make_features(df, task):

    y = get_output(df, task)

    X = df["video_name"].apply(clean_text)


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
