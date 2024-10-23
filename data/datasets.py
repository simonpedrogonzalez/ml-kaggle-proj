import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import pandas as pd
from utils.stats import sample

class Dataset:
    def __init__(self, train, test, train_labels, test_labels):
        self.train = train
        self.test = test
        self.train_labels = train_labels
        self.test_labels = test_labels
    
    def to_numpy(self):
        return Dataset(
            self.train.values,
            self.test.values if self.test is not None else None,
            self.train_labels.values,
            self.test_labels.values if self.test_labels is not None else None
        )

def cars_dataset():
    cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    train = pd.read_csv('data/car/train.csv')
    test = pd.read_csv('data/car/test.csv')
    train.columns = cols
    test.columns = cols
    train_labels = train['class']
    train = train.drop('class', axis=1)
    test_labels = test['class']
    test = test.drop('class', axis=1)
    return Dataset(train, test, train_labels, test_labels)

def bank_dataset():
    cols = columns = [
        "age", 
        "job", 
        "marital", 
        "education", 
        "default", 
        "balance", 
        "housing", 
        "loan", 
        "contact", 
        "day", 
        "month", 
        "duration", 
        "campaign", 
        "pdays", 
        "previous", 
        "poutcome",
        "class"
    ]

    train = pd.read_csv('data/bank/train.csv')
    test = pd.read_csv('data/bank/test.csv')
    train.columns = cols
    test.columns = cols
    train_labels = train['class']
    train = train.drop('class', axis=1)
    test_labels = test['class']
    test = test.drop('class', axis=1)
    return Dataset(train, test, train_labels, test_labels)

def play_tennis_dataset():
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

    df = pd.DataFrame(data, columns=cols)
    train = df.drop('class', axis=1)
    train_labels = df['class']
    return Dataset(train, None, train_labels, None)

def credit_card_default_dataset():
    df = pd.read_excel('data/credit_card_default/credit_card_default.xls', header=1, skiprows=0)
    df = df.drop('ID', axis=1)
    
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]

    # PAY_... are -1 pay duly, 1 month delay, 2 month delay, ..., 9 month delay or more
    # They are set as category since they have too low cardinality and their distribution
    # is heavily skewed to the left
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    cat_cols += pay_cols

    for col in cat_cols:
        df[col] = df[col].astype('category')

    # BILL_AMT... and PAY_AMT... are numerical

    train_size = 24000
    train = df.sample(n=train_size, random_state=0)
    test = df.drop(train.index)
    
    train_labels = train['default payment next month']
    train = train.drop('default payment next month', axis=1)
    test_labels = test['default payment next month']
    test = test.drop('default payment next month', axis=1)

    return Dataset(train, test, train_labels, test_labels)

def concrete_slump_dataset_original():

    df = pd.read_csv('data/concrete_slump/slump_test.data')
    df = df.drop('No', axis=1)
    train_size = 53
    train = df.sample(n=train_size, random_state=0)
    test = df.drop(train.index)
    train_labels = train['Compressive Strength (28-day)(Mpa)']
    train = train.drop('Compressive Strength (28-day)(Mpa)', axis=1)
    test_labels = test['Compressive Strength (28-day)(Mpa)']
    test = test.drop('Compressive Strength (28-day)(Mpa)', axis=1)
    return Dataset(train, test, train_labels, test_labels)

def concrete_slump_dataset():
    train = pd.read_csv('data/concrete/train.csv')
    test = pd.read_csv('data/concrete/test.csv')
    train.columns = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'Compressive Strength']
    test.columns = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'Compressive Strength']
    train_labels = train['Compressive Strength']
    train = train.drop('Compressive Strength', axis=1)
    test_labels = test['Compressive Strength']
    test = test.drop('Compressive Strength', axis=1)
    return Dataset(train, test, train_labels, test_labels)



def regression_toy_dataset():
    data = [
        [1, -1, 2],
        [1, 1, 3],
        [-1, 1, 0],
        [1, 2, -4],
        [3, -1, -1]
    ]
    y = [1, 4, -1, -2, 0]

    df = pd.DataFrame(data, columns=['x1', 'x2', 'x3'])
    train = df
    train_labels = pd.Series(y)
    return Dataset(train, None, train_labels, None)