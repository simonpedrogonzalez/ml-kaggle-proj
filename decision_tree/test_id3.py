from id3 import ID3
import pandas as pd
from utils import avg_error

metrics = ['infogain', 'majerr', 'gini']

def e1():
    cols = ['x1', 'x2', 'x3', 'x4', 'class']
    data = [
        [0,0,1,0, 0],
        [0,1,0,0, 0],
        [0,0,1,1, 1],
        [1,0,0,1, 1],
        [0,1,1,0, 0],
        [1,1,0,0, 0],
        [0,1,0,1, 0],
    ]
    df = pd.DataFrame(data, columns=cols)
    train = df.drop('class', axis=1)
    train_labels = df['class']
    for metric in metrics:
        id3 = ID3(metric).fit(train, train_labels)
        prediction = id3.predict(train)
        assert avg_error(prediction, train_labels) == 0
        print(id3.tree)

def e2():
    cols = ['O', 'T', 'H', 'W', 'class']
    data = [
        ['S','H','H','W', '-'],
        ['S','H','H','S', '-'],
        ['O','H','H','W', '+'],
        ['R','M','H','W', '+'],
        ['R','C','N','W', '+'],
        ['R','C','N','S', '-'],
        ['O','C','N','S', '+'],
        ['S','M','H','W', '-'],
        ['S','C','N','W', '+'],
        ['R','M','N','W', '+'],
        ['S','M','N','S', '+'],
        ['O','M','H','S', '+'],
        ['O','H','N','W', '+'],
        ['R','M','H','S', '-']
    ]

    for metric in metrics:
        df = pd.DataFrame(data, columns=cols)
        train = df.drop('class', axis=1)
        train_labels = df['class']
        id3 = ID3(metric).fit(train, train_labels)
        prediction = id3.predict(train)
        assert avg_error(prediction, train_labels) == 0
        print(id3.tree)

e1()
e2()