# 🪐 Exoplanet Classifier

## 🌌 Overview
This project applies a Neural Network to detect whether an object is an **exoplanet** using NASA's **Kepler** mission data.  
The model classifies Kepler Objects of Interest (KOIs) into:
- **Confirmed Exoplanets**
- **False Positives**
- **Candidates** (potential exoplanets held out for testing)

The neural network learns from various astrophysical features (orbital period, stellar radius, temperature, etc.) to distinguish between true exoplanet signals and false detections.

---

## 🧠 Model Architecture
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

## 📊 Data Description
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

## ⚙️ Data Preprocessing
1. **Feature selection** – Extracts relevant astrophysical parameters from the Kepler dataset  
2. **Missing values** – Replaced using **median imputation**  
3. **Label mapping:**
   - `CONFIRMED` → `1`
   - `FALSE POSITIVE` → `0`
   - `CANDIDATE` → `-1` (excluded from training; used for prediction)
4. **Train/Validation Split:** 80/20 stratified split  
5. **Feature scaling:** Standardized using `StandardScaler` from scikit-learn  

---

## 🧩 Model Training
The model is trained using TensorFlow’s Sequential API for **50 epochs** with a batch size of **32**, using early validation on the held-out set.

```python
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)
