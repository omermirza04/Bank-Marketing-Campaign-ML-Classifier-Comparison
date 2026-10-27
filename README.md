# Bank Marketing Campaign — ML Classifier Comparison

Comparing 5 machine learning classifiers to predict whether a client will subscribe to a term deposit, using the UCI Bank Marketing dataset.

## 📊 Dataset
- **Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Size:** 41,188 rows × 21 columns
- **Target:** `y` — subscribed to term deposit (yes/no)
- **Note:** Class imbalance (~11.7% positive)

## ⚙️ Pipeline
1. **EDA & cleaning** — null/duplicate checks, verified clean dataset
2. **Preprocessing** — LabelEncoder for categorical features, 80/20 train-test split, StandardScaler for distance-based models
3. **Model training** — KNN, Random Forest, SVM (linear kernel), Logistic Regression, Decision Tree

## 📈 Results (Test Accuracy)

| Model | Accuracy |
|---|---|
| 🥇 Random Forest | 91.50% |
| 🥈 Logistic Regression | 91.09% |
| 🥉 SVM | 90.84% |
| KNN | 89.97% |
| Decision Tree | 88.87% |

**Winner:** Random Forest (n_estimators=100) — ensemble averaging reduced variance vs. the single Decision Tree (-2.63%).

![Model Accuracy Comparison](Bank%20Data%20ML%205%20Models.png)

## 🛠️ Tech Stack
`Python` `pandas` `numpy` `scikit-learn` `matplotlib` `seaborn`

## ▶️ Run It
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python bank_marketing.py
```

## 🚀 Next Steps
- [ ] Stratified split + SMOTE to handle class imbalance
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Evaluate precision / recall / F1 / ROC-AUC, not just accuracy
- [ ] Feature importance analysis from Random Forest

---
*Built as a learning project comparing classic ML algorithms on tabular banking data.*
