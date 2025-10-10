# Exoplanet Classifier

## Overview
This project applies a Neural Network to detect whether an object is an **exoplanet** using NASA's **Kepler** mission data.  
The model classifies Kepler Objects of Interest (KOIs) into:
- **False Positives**: not an exoplanet
- **Candidates**: potentially an exoplanet

The neural network learns from various astrophysical features (orbital period, stellar radius, temperature, etc.) to distinguish between true exoplanet signals and false detections.

---

## Model Architecture
The classifier is implemented using **TensorFlow/Keras** and trained on labeled Kepler data.  

**Architecture:**
- Input Layer → 128 neurons (ReLU)  
- Dropout (0.4)  
- Hidden Layer → 64 neurons (ReLU)  
- Dropout (0.3)  
- Hidden Layer → 32 neurons (ReLU)  
- Dropout (0.2)  
- Output Layer → 1 neuron (Sigmoid activation)

**Loss:** Binary Cross-Entropy  
**Optimizer:** Adam  
**Metric:** Accuracy  

---

## Data Description
The dataset used is **Kepler Q1–Q17 DR24** (NASA Exoplanet Archive).  
Each observation includes astrophysical and photometric parameters related to potential exoplanet detections.

### Key Columns Used
- **koi_fpflag_nt** – Light curve not consistent with transiting planet  
- **koi_fpflag_ss** – Eclipsing binary signature  
- **koi_period** – Orbital period (days)  
- **koi_depth** – Transit depth (ppm)  
- **koi_prad** – Planet radius (Earth radii)  
- **koi_steff** – Stellar effective temperature (K)  
- **koi_smass** – Stellar mass (Solar masses)  
- **koi_srad** – Stellar radius (Solar radii)  
- **koi_teq** – Equilibrium temperature (K)  
- and 20+ other features describing the system’s physical characteristics  

---

## Data Preprocessing
1. **Feature selection** – Extracts relevant astrophysical parameters from the Kepler dataset  
2. **Missing values** – Replaced using **median imputation**
3. **Choosing label** - Disposition counts using Kepler Data (`koi_pdisposition`) is more balanced. I used that as the labels instead of Disposition count (`koi_disposition`) .
4. **Label mapping:**
   - `FALSE POSITIVE` → `0`
   - `CANDIDATE` → `1` (excluded from training; used for prediction)
5. **Train/Validation Split:** 70/15/15 stratified split  
6. **Feature scaling:** Standardized using `StandardScaler` from scikit-learn  

---

## Model Training
The model is trained using TensorFlow’s Sequential API for **50 epochs** with a batch size of **64**, using early validation on the held-out set.

```python
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=1
)
```

## Results
My neural network was trained to classify exoplanet candidates as either **CANDIDATE** or **FALSE POSITIVE** using the Kepler dataset. The model achieved **98%** accuracy. 

### Classification Report
| Metric | Class 0 (False Positive) | Class 1 (Candidate) | Overall  |
|--------|-------------------------|-------------------|---------|
| Precision | 0.99 | 0.98 | - |
| Recall    | 0.98 | 0.98 | - |
| F1-Score  | 0.98 | 0.98 | 0.98 |
| Support   | 763  | 672  | 1435 |

**Validation Confusion Matrix:**
$ \begin{pmatrix}
751 & 12 \\
11 & 661
\end{pmatrix} $

### ROC and PR Metrics
- **ROC-AUC:** 0.9987  
- **PR-AUC:** 0.9984  

These results indicate that the model performs extremely well on unseen test data, with high precision and recall for both classes, and very few misclassifications.  

