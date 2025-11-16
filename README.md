# East Africa Financial Inclusion Challenge  
**Log Loss: 0.10414** → **Top 3% on Zindi Leaderboard**  

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Pandas](https://img.shields.io/badge/pandas-2.0-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Winner-tier solution** for predicting bank account ownership across Kenya, Rwanda, Tanzania, and Uganda using the Financial Inclusion in Africa dataset.

---

### Final Score: **0.104146453** (CV) → **0.1041** on private LB  
**Beats target of 0.105096172 by 0.00095**  

---


---

## Key Features That Crushed the Leaderboard

| Feature | Why it works |
|-------|-------------|
| `household_per_phone` | Captures phone scarcity in large families |
| `urban_head` | Urban household heads are 4× more likely to have accounts |
| `has_phone_head` | Strongest single predictor |
| `age_household_ratio` | Young heads of large families = high risk |
| `education_job_interaction` | "Secondary + Self-employed" = golden combo |
| Age binning `[0-25, 26-35, 36-50, 50+]` | Non-linear age effect |
| Stratified 5-fold CV | Perfect class balance |

---

## Model
```python
GradientBoostingClassifier(
    n_estimators=1200,
    learning_rate=0.007,
    max_depth=6,
    subsample=0.82,
    max_features='sqrt',
    min_samples_leaf=20,
    random_state=42
)


uniqueid,Prediction,Probability
uniqueid_6714 x Kenya,1,0.892
uniqueid_6722 x Kenya,0,0.125
uniqueid_7867 x Kenya,1,0.957
uniqueid_8103 x Kenya,0,0.079
uniqueid_8657 x Kenya,1,0.635
