import pandas as pd
import sklearn
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

import joblib

model_filename = './result/model.pkl'
imputer_filename = './result/imputer.pkl'
scaler_filename = './result/scaler.pkl'
data_file       = './data/wow.csv'       # 理论上应该是 './data/train_data.csv'


def preprocess_data(data, imputer=None, scaler=None):

    column_name = [
        'Life expectancy ', 'infant deaths', 'Alcohol',
        'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ',
        'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
        ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources',
        'Schooling',
    ]

    data = data.drop(["Country", "Status", "Year"], axis=1)

    if imputer==None:
        imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        imputer = imputer.fit(data[column_name])
    data[column_name] = imputer.transform(data[column_name])

    if scaler==None:
        scaler = StandardScaler()
        scaler = scaler.fit(data)
    data_norm = pd.DataFrame(scaler.transform(data), columns=data.columns)

    return data_norm, imputer, scaler


def predict(test_data, filename):
    loaded_model = joblib.load(model_filename)
    imputer = joblib.load(imputer_filename)
    scaler = joblib.load(scaler_filename)

    test_data_norm, _, _ = preprocess_data(test_data, imputer, scaler)
    test_x = test_data_norm.values
    
    predictions = loaded_model.predict(test_x)

    return predictions


def model_fit(train_data):

    train_y = train_data.iloc[:,-1].values

    train_data = train_data.drop(["Adult Mortality"], axis=1)
    train_data_norm, imputer, scaler = preprocess_data(train_data)

    train_x = train_data_norm.values

    regressor = MLPRegressor(
        hidden_layer_sizes=(32, ), 
        activation='relu',
        solver='adam',
        learning_rate_init=0.006,
        random_state=1, 
        max_iter=2000, 
        shuffle=True,
        warm_start=True
    )

    regressor.fit(train_x, train_y)

    joblib.dump(regressor, model_filename)
    joblib.dump(imputer, imputer_filename)
    joblib.dump(scaler, scaler_filename)

    return regressor

def evaluate(train_data):
    label = train_data.loc[:,'Adult Mortality']
    data = train_data.iloc[:,:-1]
    y_pred = predict(data, './model.pkl')
    r2 = r2_score(label, y_pred)
    mse = mean_squared_error(label, y_pred)
    print("MSE is {}".format(mse))
    print("R2 score is {}".format(r2))

train_data = pd.read_csv(data_file)
model = model_fit(train_data)
evaluate(train_data)