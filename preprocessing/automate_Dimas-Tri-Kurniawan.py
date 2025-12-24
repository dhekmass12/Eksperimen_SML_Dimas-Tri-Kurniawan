import pandas as pd
import matplotlib.pyplot as plt
import math
from category_encoders import OrdinalEncoder
from imblearn.over_sampling import SMOTE

df = pd.read_csv('../dataset_raw.csv')
df = df.dropna()
df = df.rename(columns={'Air temperature [K]': 'Air temperature',
                                'Process temperature [K]': 'Process temperature',
                                'Rotational speed [rpm]': 'Rotational speed',
                                'Torque [Nm]': 'Torque',
                                'Tool wear [min]': 'Tool wear',
                                })

# Drop unnecessary columns
df = df.drop(['UDI'], axis=1)
df = df.drop(['Product ID'], axis=1)
df = df.drop(['Type'], axis=1)
df = df.drop(['Target'], axis=1)
df = df.reset_index(drop=True)

# Drop duplicated rows
df = df.drop_duplicates()
df = df.reset_index(drop=True)

# Encode ordinal data
categorical_cols = list(set(df.columns.to_list()) - set(df._get_numeric_data().columns.to_list()))
enc_maps = {
    "Failure Type" : [{'col': "Failure Type", 'mapping': {
        "No Failure": 0,
        "Heat Dissipation Failure": 1,
        "Power Failure": 2,
        "Overstrain Failure": 3,
        "Tool Wear Failure": 4,
        "Random Failures": 5,
        }}],
}
for col in categorical_cols:
    enc_map = enc_maps[col]
    enc = OrdinalEncoder(mapping=enc_map)
    df = enc.fit_transform(df)

# Oversampling
X = df.drop(['Failure Type'], axis=1)
y = df['Failure Type']
sampling_strategy = {
    0: 9652,
    1: 5000,
    2: 5000,
    3: 5000,
    4: 5000,
    5: 5000,
    }
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X, y = smote.fit_resample(X, y)
X = pd.DataFrame(X, columns=X.columns)
y = pd.Series(y)
df = pd.concat([X, y], axis=1)

# Save preprocessed dataset
df.to_csv("preprocessing/dataset_preprocessing.csv",index=False)