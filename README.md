# Child Mind Institute — Problematic Internet Use
## Project Report

### 1. Executive Summary
This project aims to predict the **Severity Impairment Index (sii)** of a child's internet usage using physical activity and fitness metrics as proxies. The target is an ordinal scale (0-3). The evaluation metric is **Quadratic Weighted Kappa (QWK)**, which penalizes larger variance between predicted and actual classes more severely

Current Performance: **Leaderboard QWK ~0.46+**
---

### 2. Exploratory Data Analysis
The primary challenge of this competition is the Missing Data Gap
- **Clinical Data:** Features like BMI, Heart Rate, and FitnessGram scores have missingness rates between 50% and 88%
- **Signal Data:** Actigraphy data (high-frequency accelerometer readings) is available for only a subset of participants
- **Target Imbalance:** Class 3 (Severe) represents only **~1.2%** of the dataset, creating a massive risk for model "majority class" bias

**Key Insight:** Missingness in clinical data is **Informative**. A child who does not participate in physical testing often correlates with higher impairment indices

---

### 3. Signal Processing & Feature Engineering 

We treated the 3-axis accelerometer data as a high-frequency time-series signal, applying digital signal processing techniques:

#### A. Circadian Rhythm Biomarkers
Problematic internet use often correlates with disrupted sleep and sedentary behavior. We extracted:
- **L5 & M10:** The average activity during the 5 least active and 10 most active hours
- **RA (Relative Amplitude):** Calculated as $\frac{M10 - L5}{M10 + L5}$, measuring the strength of the circadian rhythm.\
- **Night Restlessness:** Standard deviation of movement during **12 AM - 4 AM**

#### B. Frequency Domain Features
- **Spectral Entropy:** Measures the complexity of the movement. High entropy suggests chaotic activity (play), while low entropy suggests rhythmic/sedentary behavior
- **Spectral Roll-off:** The frequency below which 85% of the signal power resides, helping distinguish "bright" (high-intensity) movement from low-intensity fidgeting

#### C. Temporal Slicing
- **Weekday vs. Weekend:** We calculated movement profiles separately for school days and weekends to capture "bingeing" behaviors that occur during free time

---

### 4. Strategy

#### A. Informative Missingness
Instead of standard imputation (which "washes out" the signal), we implemented:
1. **Binary Indicators:** Created flags for all columns with >30% missing values
2. **Native NaN Handling:** Switched to XGBoost/LightGBM native handling of missing values, allowing the trees to learn the optimal "branch" for missing data

#### B. Distribution Calibration (Post-Processing)
For QWK, thresholds are notoriously unstable. We replaced the standard threshold optimization with **Distribution Calibration**:
- We calculate the exact probability distribution of the training labels
- We sort the raw regression outputs and assign classes based on those fixed percentages

This prevents the model from "compressing" all predictions into Class 0/1 and ensures we predict the rare Class 3 at the correct frequency

#### C. Two-Phase Pseudo-Labeling
To leverage the ~30% of unlabeled training data:
1. Train an ensemble on labeled data
2. Only select samples where the regression output is extremely close to an integer ($|score - round(score)| < 0.2$)
3. Re-train the final ensemble on the expanded dataset

---

### 5. Model Architecture & Optimization
- **Ensemble:** A 50/50 blend of **XGBRegressor** and **LGBMRegressor**
- **Class Weighting:** Applied a **10.0x weight** to Class 3 samples to counteract the extreme imbalance
