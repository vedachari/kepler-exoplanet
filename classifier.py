import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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


# Features: 70% train, 15% val, 5% test
y = df["label"].values
X = df.drop(columns=["koi_pdisposition", "koi_disposition", "label", "kepid","kepoi_name"], axis=1)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1, shuffle = True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1, shuffle = True)

#Scale X
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
X_val = pd.DataFrame(scaler.transform(X_val), index = X_val.index, columns = X_val.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

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

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), verbose = 1)
# hyperparam tuning: increased batch size from 32 to 64 to remove gradient noise

# Evaluate the model
y_test_pred = (model.predict(X_test) > 0.5).astype("int32")

print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))
print("ValidTestation Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

#ROC_AUC and PR_AUC
probs = model.predict(X_test).ravel()
print("ROC-AUC:", roc_auc_score(y_test, probs))
print("PR-AUC:", average_precision_score(y_test, probs))

#visualize training metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(50)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epoch Trial')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch Trial')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.savefig('metrics_type.png')
plt.show()