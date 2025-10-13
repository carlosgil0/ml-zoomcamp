import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder

np.random.seed(42)

df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv')
print(df.dtypes)
print(df.isna().sum())

# Fill missing values for all columns of a specific dtype at once
for col_type, fill_value in [('object', 'NA'), (['float', 'int'], 0.0)]:
    cols = df.select_dtypes(include=col_type).columns
    df[cols] = df[cols].fillna(fill_value)


print(f'The industry with the highest number of leads is: {df.industry.value_counts().idxmax()}')

# Correlation matrix for numerical features
df.select_dtypes(include=['float', 'int']).corr().round(4)

# Separating my target variable 'converted'
X = df.drop('converted', axis=1)
y = df['converted']

# Split data into 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#Calculate the mutual information score between converted and other categorical variables in the dataset. Use the training set only.
for c in X_train.select_dtypes(include=['object']).columns:
    mi = metrics.mutual_info_score(X_train[c], y_train)
    print(f'Mutual Information between converted and {c} is {round(mi, 2)}')

# One-hot encoding for categorical features
categorical = list(X_train.select_dtypes(include=['object']).columns)
numerical = list(X_train.select_dtypes(include=['float', 'int']).columns)

ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(X_train[categorical])

# Transform categorical features for both train and validation sets
X_train_cat = ohe.transform(X_train[categorical])
X_val_cat = ohe.transform(X_val[categorical])

# Combine with numerical features
X_train_processed = hstack([X_train[numerical].values, X_train_cat])
X_val_processed = hstack([X_val[numerical].values, X_val_cat])

# Train the logistic regression model
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_processed, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val_processed)

# Calculate and print the accuracy
accuracy = metrics.accuracy_score(y_val, y_pred)
print(f'\nValidation accuracy: {round(accuracy, 2)}')


original_accuracy = accuracy
all_features = numerical + categorical

feature_performance = {}

for feature_to_drop in all_features:
    # Create new feature lists excluding the current feature
    temp_numerical = [f for f in numerical if f != feature_to_drop]
    temp_categorical = [f for f in categorical if f != feature_to_drop]

    # Prepare data without the dropped feature
    ohe_temp = OneHotEncoder(handle_unknown='ignore')
    ohe_temp.fit(X_train[temp_categorical])

    X_train_cat_temp = ohe_temp.transform(X_train[temp_categorical])
    X_val_cat_temp = ohe_temp.transform(X_val[temp_categorical])

    X_train_processed_temp = hstack([X_train[temp_numerical].values, X_train_cat_temp])
    X_val_processed_temp = hstack([X_val[temp_numerical].values, X_val_cat_temp])

    # Train and evaluate the model
    model_temp = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model_temp.fit(X_train_processed_temp, y_train)
    accuracy_without_feature = metrics.accuracy_score(y_val, model_temp.predict(X_val_processed_temp))
    accuracy_diff = original_accuracy - accuracy_without_feature
    print(f"Feature dropped: {feature_to_drop:<25} | New Accuracy: {accuracy_without_feature:.4f} | Accuracy Diff: {accuracy_diff:.4f}")


c_values = [0.01, 0.1, 1, 10, 100]

for c in c_values:
    # Train the logistic regression model with regularization
    print(f'Training with C={c}')
    model_reg = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=42)
    model_reg.fit(X_train_processed, y_train)
    y_pred_reg = model_reg.predict(X_val_processed)

    # Calculate and print the accuracy
    accuracy_reg = metrics.accuracy_score(y_val, y_pred_reg)
    print(f"C={c} | Validation accuracy: {round(accuracy_reg, 3)}")
