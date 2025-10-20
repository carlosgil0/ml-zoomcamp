import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

np.random.seed(1)
df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv')

print(df.dtypes)
print(df.isna().sum())

# Fill missing values for all columns of a specific dtype at once
for col_type, fill_value in [('object', 'NA'), (['float', 'int'], 0.0)]:
    cols = df.select_dtypes(include=col_type).columns
    df[cols] = df[cols].fillna(fill_value)

# Separating my target variable 'converted'
X = df.drop('converted', axis=1)
y = df['converted']

# Split data into 60% train, 20% validation, 20% test
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

# Separate target variable from features
y_train = df_train.converted
y_val = df_val.converted
y_full_train = df_full_train.converted
y_test = df_test.converted

X_train = df_train.drop('converted', axis=1)
X_val = df_val.drop('converted', axis=1)
X_full_train = df_full_train.drop('converted', axis=1)
X_test = df_test.drop('converted', axis=1)


# --- Numerical Feature Importance (ROC AUC) ---
numerical = list(X_train.select_dtypes(include=['float', 'int']).columns)

for col in X_train.select_dtypes(include=['float', 'int']).columns:
    auc = roc_auc_score(y_train, X_train[col])
    if auc < 0.5:
        # Invert the feature if AUC < 0.5
        auc = roc_auc_score(y_train, -X_train[col])
    print(f"Feature: {col:<30} | AUC: {auc:.4f}")

# one hot encoding for categorical features
categorical = list(X_train.select_dtypes(include=['object']).columns)
dv = DictVectorizer(sparse=False)

# Prepare training data
train_dict = X_train[categorical + numerical].to_dict(orient='records')
X_train_dv = dv.fit_transform(train_dict)

# Train the model
log_model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=1)
log_model.fit(X_train_dv, y_train)

# Prepare validation data and make predictions
val_dict = X_val[categorical + numerical].to_dict(orient='records')
X_val_dv = dv.transform(val_dict)
y_pred = log_model.predict_proba(X_val_dv)[:, 1]

# Calculate and print AUC
auc_val = roc_auc_score(y_val, y_pred)
print(f"Validation AUC: {round(auc_val, 3)}")

thresholds = np.linspace(0, 1, 101)
precision_scores = []
recall_scores = []
f1_scores = []

for t in thresholds:
    # Convert probabilities to binary predictions based on threshold
    y_binary_pred = (y_pred >= t).astype(int)
    
    # Calculate precision and recall
    p = precision_score(y_val, y_binary_pred, zero_division=0)
    r = recall_score(y_val, y_binary_pred, zero_division=0)
    f1 = f1_score(y_val, y_binary_pred, zero_division=0)

    precision_scores.append(p)
    recall_scores.append(r)
    f1_scores.append(f1)

# Plotting Precision and Recall
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision_scores, label='Precision')
plt.plot(thresholds, recall_scores, label='Recall')
plt.plot(thresholds, f1_scores, label='F1 Score')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1 Score vs. Decision Threshold')
plt.legend()
plt.grid(True)
plt.show()

metrics_df = pd.DataFrame({
    'Threshold': thresholds,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1 Score': f1_scores
})

metrics_df.columns = ['Threshold', 'Recall', 'Precision', 'F1 Score']


kfold = KFold(n_splits=5, shuffle=True, random_state=1)

scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_full_train)):
    X_train_fold = X_full_train.iloc[train_idx]
    y_train_fold = y_full_train.iloc[train_idx]
    X_val_fold = X_full_train.iloc[val_idx]
    y_val_fold = y_full_train.iloc[val_idx]

    dv_fold = DictVectorizer(sparse=False)
    train_dict_fold = X_train_fold[categorical + numerical].to_dict(orient='records')
    X_train_dv_fold = dv_fold.fit_transform(train_dict_fold)

    val_dict_fold = X_val_fold[categorical + numerical].to_dict(orient='records')
    X_val_dv_fold = dv_fold.transform(val_dict_fold)

    model_fold = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model_fold.fit(X_train_dv_fold, y_train_fold)

    y_pred_fold = model_fold.predict_proba(X_val_dv_fold)[:, 1]
    auc = roc_auc_score(y_val_fold, y_pred_fold)
    scores.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.4f}")

print(f"\nMean AUC: {np.mean(scores):.4f}")
print(f"Std Dev AUC: {np.std(scores):.4f}")

# --- K-Fold Cross-Validation for C Parameter Tuning ---
print("\n--- K-Fold Cross-Validation for C Parameter Tuning ---")
c_values = [0.000001, 0.001, 1]
for C in c_values:
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_full_train)):
        X_train_fold = X_full_train.iloc[train_idx]
        y_train_fold = y_full_train.iloc[train_idx]
        X_val_fold = X_full_train.iloc[val_idx]
        y_val_fold = y_full_train.iloc[val_idx]

        dv_fold = DictVectorizer(sparse=False)
        train_dict_fold = X_train_fold[categorical + numerical].to_dict(orient='records')
        X_train_dv_fold = dv_fold.fit_transform(train_dict_fold)

        val_dict_fold = X_val_fold[categorical + numerical].to_dict(orient='records')
        X_val_dv_fold = dv_fold.transform(val_dict_fold)

        model_fold = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model_fold.fit(X_train_dv_fold, y_train_fold)

        y_pred_fold = model_fold.predict_proba(X_val_dv_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, y_pred_fold)
        scores.append(auc)

    print(f"C={C}: Mean AUC = {np.mean(scores):.3f}, Std Dev = {np.std(scores):.3f}")

print("\nBased on the results, C=.001 leads to the best mean score.")
