import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

df = pd.read_csv('q1_q17_dr24_koi_2025.09.09_17.48.32.csv', delimiter=',')

print(len(df.columns))

# Classification:
# Exoplanet Archive Disposition (koi_disposition): confirmed, false positive, candidate (used as labels)
# disposition using kepler data (koi_pdisposition): CANDIDATE, FALSE POSITIVE, CONFIRMED
# columns of interest:
# koi_fpflag_nt: 1 if KOI light curve not consistent with transiting planet
# koi_fpflag_ss: 1 if KOI observed to be caused by eclipsing binary in Kepler data
# koi_fpflag_co: 1 if KOI observed to be caused by nearby star
# koi_fpflag_ec: 1 if KOI observed to be caused flux contamination of another object
# koi_period: orbital period in days
# koi_time0bk: transit epoch in BJD (Barycentric Julian Date)
# koi_eccen: orbital eccentricity
# koi_impact: transit impact parameter
# koi_duration: transit duration in hours
# koi_depth: transit depth in ppm
# koi_ror: ratio of planet radius to stellar radius
# koi_srho: stellar density in g/cm^3
# koi_prad: planet radius in Earth radii


columns_of_interest = [
    "koi_disposition", # label
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_period",
    "koi_time0bk",
    "koi_eccen",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_ror",
    "koi_srho",
    "koi_prad",
    "koi_teq",  # equilibrium temperature
    "koi_insol", # insolation flux
    "koi_model_snr", # signal to noise
    "koi_num_transits", # number of observed transits
    "koi_steff", # stellar effective temperature
    "koi_slogg", # stellar surface gravity
    "koi_srad", # stellar radius
    "koi_smass", # stellar mass
    "koi_kepmag", # Kepler magnitude
    "koi_depth_err1", 
    "koi_depth_err2",
    "koi_ror_err1", 
    "koi_ror_err2",
    "koi_srho_err1", 
    "koi_srho_err2",
    "koi_prad_err1", 
    "koi_prad_err2",
    "koi_gmag",
    "koi_rmag", 
    "koi_imag", 
    "koi_zmag", 
    "koi_jmag", 
    "koi_hmag", 
    "koi_kmag"
]

df = df[columns_of_interest]
print(df.columns)
print(df[columns_of_interest].isna().sum())


# Fill remaining NaNs with median
df = df.fillna(df.median(numeric_only=True))
print(df.isna().sum())

# Map labels for ML:
# CONFIRMED = 1, FALSE POSITIVE = 0, CANDIDATE = -1 (holdout test set)
label_map = {"CONFIRMED": 1, "FALSE POSITIVE": 0, "CANDIDATE": -1}
df["label"] = df["koi_disposition"].map(label_map)

# Features
X = df.drop(columns=["koi_disposition", "label"]).values
y = df["label"].values

# split into training and validation
# use candidate as test set
mask_trainval = y != -1
X_trainval = X[mask_trainval]
y_trainval = y[mask_trainval]

# holdout test set: disposition = CANDIDATE
mask_test = y == -1
X_candidates = X[mask_test]

X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval)

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

# apply to candidate set
candidate_probs = model.predict(X_candidates)

df_candidates = df[mask_test].copy()
df_candidates['predicted_probability'] = candidate_probs
df_candidates.to_csv('candidates_with_predictions.csv', index=False)
print("Predictions for candidates saved to 'candidates_with_predictions.csv'")