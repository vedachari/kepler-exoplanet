import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

df = pd.read_csv('cumulative.csv', delimiter=',') #used kaggle dataset for easier use

print(df.info())
print(len(df.columns))
print(df.shape)
print(df.isnull().sum())

# drop Equilibrium temp 1 and 2 (koi_teq_err1 and koi_teq_err2) since most are missing
# drop TCE deliver name (koi_tce_delivname ) since most are the same 
# drop kepler name (kepler_name)

df.drop(columns = ['koi_teq_err1', 'koi_teq_err2', 'koi_tce_delivname', 'kepler_name'], inplace=True)

# Separate out categorical features
# ['kepoi_name' KOINAME, 'koi_disposition' Disposition, 'koi_pdisposition' Disposition using Kepler Data]

#fill in null for numerical features with mean
df= df.fillna(df.median(numeric_only=True))
print("fixed null: ", df.isna().sum())

# Explor Disposition and Disposition using Kepler Data
print('Disposition Counts')
print(df['koi_disposition'].value_counts())
print('Disposition Counts Using Kepler Data')
print(df['koi_pdisposition'].value_counts())

# koi_pdisposition is more balanced (consistent with my approach with other csv)


# Map labels for ML:
# CONFIRMED = 1, FALSE POSITIVE = 0
label_map = {"FALSE POSITIVE": 0, "CANDIDATE": 1}
df["label"] = df["koi_pdisposition"].map(label_map)
print(df["label"].value_counts(dropna=False))


# Features
y = df["label"].values
X = df.drop(columns=["koi_pdisposition", "koi_disposition", "label", "kepid","kepoi_name"], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1, shuffle = True)

#Scale X
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
X_val = pd.DataFrame(scaler.transform(X_val), index = X_val.index, columns = X_val.columns)

#build neural netowrk
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose = 1)

# Evaluate the model
y_val_pred = (model.predict(X_val) > 0.5).astype("int32")

print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred))
print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# # apply to candidate set
# candidate_probs = model.predict(X_candidates)

# df_candidates = df[mask_test].copy()
# df_candidates['predicted_probability'] = candidate_probs
# df_candidates.to_csv('candidates_with_predictions.csv', index=False)
# print("Predictions for candidates saved to 'candidates_with_predictions.csv'")