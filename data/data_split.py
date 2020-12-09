import datetime
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import WordPunctTokenizer


def process_dataset(json_path, select_cols, test_num, csv_path):
    print('## Read the json file...')
    df = pd.read_json(json_path, lines=True)
    df = df[select_cols]
    df.columns = ['userID', 'itemID', 'review', 'rating', 'reviewTime']  # Rename columns
    # project user/item name to unique id
    df['userID'] = df.groupby(df['userID']).ngroup()
    df['itemID'] = df.groupby(df['itemID']).ngroup()

    with open('../embedding/stopwords.txt') as f:  # stop vocabularies
        stop_words = set(f.read().splitlines())
    with open('../embedding/punctuations.txt') as f:  # Useless punctuations
        punctuations = set(f.read().splitlines())

    def clean_review(review):
        review = review.lower()
        for p in punctuations:
            review = review.replace(p, ' ')  # clear punctuation
        review = WordPunctTokenizer().tokenize(review)  # split word
        review = [word for word in review if word not in stop_words]  # remove stop word
        # review = [nltk.WordNetLemmatizer().lemmatize(word) for word in review]  # extract word root
        return ' '.join(review)

    df = df.drop(df[[not isinstance(x, str) or len(x) == 0 for x in df['review']]].index)  # remove empty reviews!
    df['review'] = df['review'].apply(clean_review)

    def str2timestamp(date_time):  # Convert datetime to form of second
        dt = datetime.datetime.strptime(date_time, '%m %d, %Y')
        dt = dt.timestamp()
        return round(dt)

    df['reviewTime'] = df['reviewTime'].apply(str2timestamp)

    # For every user, the last review is used for test while penultimate is used for development
    train, valid, test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for s in df.groupby(by='userID'):
        samples = s[1].sort_values(by='reviewTime', ascending=True)
        samples.drop('reviewTime', axis=1, inplace=True)  # reviewTime is not saved.
        if samples.shape[0] > test_num * 2:
            train = train.append(samples[:-test_num * 2])
            valid = valid.append(samples[-test_num * 2:-test_num])
            test = test.append(samples[-test_num:])
    train.to_csv(os.path.join(csv_path, 'train.csv'), index=False, header=False)
    valid.to_csv(os.path.join(csv_path, 'valid.csv'), index=False, header=False)
    test.to_csv(os.path.join(csv_path, 'test.csv'), index=False, header=False)
    print(f'#### Split and saved dataset as csv: train {len(train)}, valid {len(valid)}, test {len(test)}')
    print(f'#### Total: {len(df)} reviews, {len(df.groupby("userID"))} users, {len(df.groupby("itemID"))} items.')
    return train, valid, test


if __name__ == '__main__':
    start_time = time.perf_counter()

    # You must set following args depend on your dataset!
    data_path = os.path.abspath('./music/Digital_Music.json.gz')
    select_Cols = ['reviewerID', 'asin', 'reviewText', 'overall', 'reviewTime']  # Selected exactly 5 columns
    save_path = os.path.dirname(data_path)

    process_dataset(data_path, select_Cols, 3, save_path)

    end_time = time.perf_counter()
    print(f'## preprocess.py: Data loading complete! Time used {end_time - start_time:.0f} seconds.')
