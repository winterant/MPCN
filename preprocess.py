import datetime
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import WordPunctTokenizer


def process_dataset(json_path, test_num=3):
    print('## Read the json file...')
    df = pd.read_json(json_path, lines=True)
    df = df[['reviewerID', 'asin', 'reviewText', 'overall', 'reviewTime']]
    df.columns = ['userID', 'itemID', 'review', 'rating', 'reviewTime']
    # project user/item name to unique id
    df['userID'] = df.groupby(df['userID']).ngroup()
    df['itemID'] = df.groupby(df['itemID']).ngroup()

    with open('data/embedding/stopwords.txt') as f:  # stop vocabularies
        stop_words = set(f.read().splitlines())
    with open('data/embedding/punctuations.txt') as f:  # Useless punctuations
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
    df['review'] = df['review'].apply(clean_review)  # 清洗文本

    def str2timestamp(date_time):  # Convert datetime to form of second
        dt = datetime.datetime.strptime(date_time, '%m %d, %Y')
        dt = dt.timestamp()
        return round(dt)

    df['reviewTime'] = df['reviewTime'].apply(str2timestamp)

    print(f'## Got {len(df)} reviews from json! Split them into train,validation and test!')
    # For every user, the last review is used for test while penultimate is used for development
    train, valid, test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for s in df.groupby(by='userID'):
        samples = s[1].sort_values(by='reviewTime', ascending=True)
        samples.drop('reviewTime', axis=1, inplace=True)  # reviewTime is not saved.
        if samples.shape[0] > test_num * 2:
            train = train.append(samples[:-test_num * 2])
            valid = valid.append(samples[-test_num * 2:-test_num])
            test = test.append(samples[-test_num:])
    print(f'## Saving the data. count: train {len(train)}, valid {len(valid)}, test {len(test)}')
    train.to_csv(os.path.dirname(json_path) + '/train.csv', index=False, header=False)
    valid.to_csv(os.path.dirname(json_path) + '/valid.csv', index=False, header=False)
    test.to_csv(os.path.dirname(json_path) + '/test.csv', index=False, header=False)

    user_count = len(df.groupby('userID'))
    item_count = len(df.groupby('itemID'))
    print(f'## Total user count:{user_count}; total item count:{item_count}')
    return train, valid, test


if __name__ == '__main__':
    print('## preprocess.py: Begin to load the data...')
    start_time = time.perf_counter()
    process_dataset('data/music/Digital_Music.json', test_num=3)
    end_time = time.perf_counter()
    print(f'## preprocess.py: Data loading complete! Time used {end_time - start_time:.0f} seconds.')
