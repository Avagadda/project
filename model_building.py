!pip install catboost
pip install scikit-learn lightgbm

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error
from sklearn import metrics
import pickle
df = df.drop(['name'], axis=1)
# label encode the categorical values
le_manufacturer = LabelEncoder()
le_engine = LabelEncoder()
le_transmission = LabelEncoder()

df['manufacturer'] = le_manufacturer.fit_transform(df['manufacturer'])
df['engine'] = le_engine.fit_transform(df['engine'])
df['transmission'] = le_transmission.fit_transform(df['transmission'])
# creating X and y variables
X = df.drop('price', axis=1)

# log transform the price column
y = np.log(df['price'])
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                    test_size=0.3, random_state=42)
from sklearn.preprocessing import MinMaxScaler

# Assuming X_train and X_test are your feature matrices
# Assuming you have a DataFrame, use select_dtypes to exclude non-numeric columns

numeric_columns = X_train.select_dtypes(include=['number']).columns

# Instantiate the Min-Max Scaler and fit on the training data
scaler = MinMaxScaler().fit(X_train[numeric_columns])

# Transform training data
X_train_scaled = scaler.transform(X_train[numeric_columns])

# Transform testing data using the same scaler
X_test_scaled = scaler.transform(X_test[numeric_columns])

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Assuming X_train, X_test, y_train, and y_test are defined

# Define a preprocessor for numeric and categorical columns
numeric_features = X_train.select_dtypes(include=['number']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

def train_model(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    my_dict = {
        'Name': [],
        'Train_Score': [],
        'R_squared': [],
        'Mean_absolute_error': [],
        'Root_mean_sqd_error': []
    }

    for name, estimator in models.items():
        # Construct a pipeline with the preprocessor and the estimator
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', estimator)
        ])

        # fit
        model.fit(X_train, y_train)

        # make predictions
        y_pred = model.predict(X_test)

        # metrics
        train_score = model.score(X_train, y_train)
        r_sqd = metrics.r2_score(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # add the metrics to the dictionary
        my_dict['Name'].append(name)
        my_dict['Train_Score'].append(train_score)
        my_dict['R_squared'].append(r_sqd)
        my_dict['Mean_absolute_error'].append(mae)
        my_dict['Root_mean_sqd_error'].append(rmse)

    my_dataframe = pd.DataFrame(my_dict)
    my_dataframe = my_dataframe.sort_values('Root_mean_sqd_error')
    return my_dataframe

result_df = train_model(models, X_train, X_test, y_train, y_test)
print(result_df)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Assuming 'colour' is a categorical column

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),  # Numeric features unchanged
        ('cat', OneHotEncoder(), ['colour'])  # One-hot encode 'colour'
    ])

# Use the preprocessor in your pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(random_state=123))
])

# Now you can fit the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

grid_model = pd.DataFrame({
    'model': ['LGBM'],
    'r_squared': [metrics.r2_score(y_test, y_pred)],
    'mae': [mean_absolute_error(y_test, y_pred)],
    'rmse': [np.sqrt(metrics.mean_squared_error(y_test, y_pred))]
    })
grid_model

data = {"model": model, "normalisation": numeric_columns}
with open('../models/regressor.pkl', 'wb') as file:
    pickle.dump(data, file)

import os
import pickle
from lightgbm import LGBMRegressor

# Assuming 'model' is your LGBMRegressor instance
model = LGBMRegressor(random_state=123)  # Replace this with your actual model

# Assuming 'numeric_columns' is a list of numeric column names used in normalization
numeric_columns = ['mileage','transmission']  # Replace with your actual numeric columns

# Specify the directory where you want to save the file
directory = '../models/'

# Create the directory if it does not exist
os.makedirs(directory, exist_ok=True)

# Create a dictionary with model and normalization information
data = {"model": model, "normalisation": numeric_columns}

# Save the data to a file using pickle
with open(os.path.join(directory, 'regressor.pkl'), 'wb') as file:
    pickle.dump(data, file)
