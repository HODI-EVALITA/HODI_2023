import argparse
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

def compute_baseline_task_a(train_path, test_path, output_path):
    train = pd.read_csv(train_path, sep='\t', usecols=['id', 'text', 'homotransphobic'],
                        converters={'id': str, 'text': str, 'homotransphobic': int})
    test = pd.read_csv(test_path, sep='\t', usecols=['id', 'text'], converters={'id': str, 'text': str})

    vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                 min_df=0.001,
                                 max_df=0.7,
                                 analyzer='word',
                                 sublinear_tf=True,
                                 stop_words=stopwords.words('italian')
                                 )

    X_train = vectorizer.fit_transform(train['text'])
    X_test = vectorizer.transform(test['text'])
    y_train = train['homotransphobic']

    # Train logistic classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # saving the predictions
    test['homotransphobic'] = classifier.predict(X_test)
    test[['id', 'homotransphobic']].to_csv(output_path, sep='\t', index=False)

    print("Prediction Subtask A saved at: " + output_path)


def compute_baseline_task_b(data_path, output_path):
    data = pd.read_csv(data_path, sep='\t', usecols=['id', 'text'], converters={'id': str, 'text': str})

    # creating a random baseline
    random_baseline = lambda text: [i for i, char in enumerate(text) if random.random() > 0.5]

    data['rationales'] = data.text.apply(random_baseline)
    data[['id', 'rationales']].to_csv(output_path, sep='\t', index=False)

    print("Prediction Subtask B saved at: " + output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EVALITA HODI 2023 - Baseline script.')
    parser.add_argument('--train_path', type=str, required=True,
                        help='path of the train tsv file')
    parser.add_argument('--test_path', type=str, required=True,
                        help='path of the test tsv file')
    parser.add_argument('--task', type=str, required=True, choices=['a', 'b'],
                        help='task you want to evaluate ("a" or "b")')
    parser.add_argument('--output_path', type=str, required=False, default="result.tsv",
                        help='path of output prediction file')

    args = parser.parse_args()

    if args.task.lower() == "a":
        compute_baseline_task_a(args.train_path, args.test_path, args.output_path)
    elif args.task.lower() == "b":
        compute_baseline_task_b(args.test_path, args.output_path)
    else:
        raise Exception('Task should be either "a" or "b"')
