import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data collection and processing
sonar_data = pd.read_csv(r"C:\Users\thenn\OneDrive\Desktop\sonar.csv", header=None)

print(sonar_data.head())

print(sonar_data.describe())
print(sonar_data[60].value_counts())  # Column 60 is the label

# Splitting data into features and label
x = sonar_data.drop(columns=60)  # Drop column 60 (label)
y = sonar_data[60]               # Label column

print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,stratify=y,random_state=1)

print(x.shape,x_train.shape,x_test.shape)

model=LogisticRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_train)
loss=accuracy_score(prediction,y_train)
print('accuracy on training data:',loss)

x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('accuracy on test data:',test_data_accuracy)


# Extracting features from the first sample (excluding the label)
sample_data = sonar_data.iloc[100, :-1]  # Get only columns 0 to 59 (features)

# Reshaping the data to fit model input shape
sample_data_reshaped = sample_data.values.reshape(1, -1)

print(sample_data_reshaped)

pred=model.predict(sample_data_reshaped)
print(pred)