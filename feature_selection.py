import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


# 1. LOAD DATA

train = pd.read_csv("assets/Train_data.csv")
test = pd.read_csv("assets/Test_data.csv")

target = "class"


# 2. SPLIT FEATURES & LABEL

X_train = train.drop(columns=[target])
y_train = train[target]

if target in test.columns:
    X_test = test.drop(columns=[target])
else:
    X_test = test.copy()

# 3. ENCODE CATEGORICAL DATA

combined = pd.concat([X_train, X_test], axis=0)

categorical_cols = combined.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# Split back
X_train = combined.iloc[:len(X_train), :].copy()
X_test = combined.iloc[len(X_train):, :].copy()


# 4. REMOVE LOW VARIANCE FEATURES

var_selector = VarianceThreshold(threshold=0.01)
X_train = var_selector.fit_transform(X_train)

selected_columns = train.drop(columns=[target]).columns[var_selector.get_support()]
X_train = pd.DataFrame(X_train, columns=selected_columns)
X_test = X_test[selected_columns]


# 5. MUTUAL INFORMATION

k = 15
mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_train_mi = mi_selector.fit_transform(X_train, y_train)

mi_features = X_train.columns[mi_selector.get_support()]


# 6. RANDOM FOREST IMPORTANCE

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_importance = pd.Series(rf.feature_importances_, index=X_train.columns)
rf_features = rf_importance.sort_values(ascending=False).head(k).index

# 7. FINAL FEATURE SET

final_features = list(set(mi_features).intersection(set(rf_features)))

if len(final_features) < 8:
    final_features = mi_features.tolist()

print("Final Selected Features:")
print(final_features)


# 8. SAVE FINAL DATASETS

final_train = X_train[final_features].copy()
final_train[target] = y_train

final_test = X_test[final_features].copy()

final_train.to_csv("Final_Selected_Train.csv", index=False)
final_test.to_csv("Final_Selected_Test.csv", index=False)

print("Feature selection completed successfully.")
