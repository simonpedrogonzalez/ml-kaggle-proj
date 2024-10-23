import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import pandas as pd
from utils.stats import sample

class Dataset:
    def __init__(self, train, test, train_labels, test_labels, to_predict=None):
        self.train = train
        self.test = test
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.to_predict = to_predict
        try:
            self.all_data, self.indices = self.get_all_data()
            self.all_labels = self.get_all_labels()
        except:
            pass


    def get_all_labels(self):
        all_labels = self.train_labels
        if self.test_labels is not None:
            all_labels = pd.concat([all_labels, self.test_labels])
        all_labels = all_labels.reset_index(drop=True)
        return all_labels

    def get_all_data_pandas(self, train, test, to_predict):
        alldata = train
        if test is not None:
            alldata = pd.concat([alldata, test])
        if to_predict is not None:
            alldata = pd.concat([alldata, to_predict])
        alldata = alldata.reset_index(drop=True)
        indices = {
            'train': range(train.shape[0]),
            'test': range(train.shape[0], train.shape[0] + test.shape[0]) if test is not None else None,
            'to_predict': range(train.shape[0] + test.shape[0], alldata.shape[0]) if to_predict is not None else None
        }
        return alldata, indices

    def get_all_data(self):
        if isinstance(self.train, pd.DataFrame):
            return self.get_all_data_pandas(self.train, self.test, self.to_predict)
        else:
            raise ValueError('Invalid data type')
            # Assume CatEncodedDataset
            # train = self.train.to_pandas()
            # test = self.test.to_pandas() if self.test is not None else None
            # to_predict = self.to_predict.to_pandas() if self.to_predict is not None else None
            # alldata, indices = self.get_all_data_pandas(train, test, to_predict)
            # # Go back to CatEncodedDataset
            # return CatEncodedDataFrame().from_pandas(alldata), indices

    def to_numpy(self):
        return Dataset(
            self.train.values,
            self.test.values if self.test is not None else None,
            self.train_labels.values,
            self.test_labels.values if self.test_labels is not None else None
        )

    def apply_transform_on_all_data(self, transform):
        """Applied as a whole on the concatenated data"""
        data = self.all_data.copy()
        data = transform(data)
        if isinstance(data, pd.DataFrame):
            self.all_data = data
            self.train = data.loc[self.indices['train']]
            if self.test is not None:
                self.test = data.loc[self.indices['test']]
            if self.to_predict is not None:
                self.to_predict = data.loc[self.indices['to_predict']]
            return self
        else:
            self.all_data = data
            self.train = data[self.indices['train']]
            if self.test is not None:
                self.test = data[self.indices['test']]
            if self.to_predict is not None:
                self.to_predict = data[self.indices['to_predict']]
            return self
    
    def apply_transform_on_all_labels(self, transform):
        """Applied as a whole on the concatenated labels"""
        all_labels = transform(self.all_labels)
        if isinstance(all_labels, pd.Series):
            self.train_labels = self.all_labels.loc[self.indices['train']]
            if self.test_labels is not None:
                self.test_labels = self.all_labels.loc[self.indices['test']]
            return self
        else:
            self.train_labels = all_labels[self.indices['train']]
            if self.test_labels is not None:
                self.test_labels = all_labels[self.indices['test']]
            return self

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


def income_level_dataset():
    df = pd.read_csv('data/income_level/train_final.csv')
    # take 20% of the data for testing
    train_size = int(0.8 * df.shape[0])
    train = df.sample(n=train_size, random_state=0)
    target = "income>50K"
    test = df.drop(train.index)
    train_labels = train[target]
    train = train.drop(target, axis=1)
    test_labels = test[target]
    test = test.drop(target, axis=1)
    # to predict data
    df2 = pd.read_csv('data/income_level/test_final.csv')
    df2.drop('ID', axis=1, inplace=True)
    return Dataset(train, test, train_labels, test_labels, df2)


def income_level_dataset_plain():
    df = pd.read_csv('data/income_level/train_final.csv')
    df2 = pd.read_csv('data/income_level/test_final.csv')
    return Dataset(df, None, None, None, df2)