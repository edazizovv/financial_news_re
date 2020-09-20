
import pandas
from codes.util import Conductor

def load(verbose=True, test_partition=0.2, validation_partition=0.2):
    # Watch the Data

    categorical_columns = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    numerical_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

    outputs = ['Exited']

    d = './mpydge/datasets/data/Churn_Modelling.csv'
    data_set = pandas.read_csv(d)

    for category in categorical_columns:
        data_set[category] = data_set[category].astype('category')
    for numeric in numerical_columns:
        data_set[numeric] = data_set[numeric].astype('float64')

    data_set[outputs[0]] = data_set[outputs[0]].astype('category')

    data = Conductor(data_frame=data_set, target=outputs)

    return data