"""
DAT3 - Diabetes Dataset (CDC)
Ethan Zhuo
N-Number: N15906048

Column info:
1)  Diabetes               - Diabetes diagnosis (1 = yes, 0 = no) -- OUTCOME
2)  HighBP                 - High blood pressure (1 = yes, 0 = no)
3)  HighChol               - High cholesterol (1 = yes, 0 = no)
4)  BMI                    - Body Mass Index
5)  Smoker                 - Smoked 100+ cigarettes in life (1 = yes, 0 = no)
6)  Stroke                 - Prior stroke (1 = yes, 0 = no)
7)  Myocardial             - Prior heart attack (1 = yes, 0 = no)
8)  PhysActivity           - Physically active (1 = yes, 0 = no)
9)  Fruit                  - Eats fruit daily (1 = yes, 0 = no)
10) Vegetables             - Eats vegetables daily (1 = yes, 0 = no)
11) HeavyDrinker           - Heavy drinker per CDC threshold (1 = yes, 0 = no)
12) HasHealthcare          - Has healthcare coverage (1 = yes, 0 = no)
13) NotAbleToAffordDoctor  - Could not afford doctor in past year (1 = yes, 0 = no)
14) GeneralHealth          - Self-rated general health (1-5)
15) MentalHealth           - Poor mental health days in last 30 (0-30)
16) PhysicalHealth         - Poor physical health days in last 30 (0-30)
17) HardToClimbStairs      - Difficulty climbing stairs (1 = yes, 0 = no)
18) BiologicalSex          - Biological sex (1 = male, 2 = female)
19) AgeBracket             - Age bracket (1=18-24 ... 13=80+)
20) EducationBracket       - Education level (1=kindergarten ... 6=college grad)
21) IncomeBracket          - Annual income bracket (1=below $10k ... 8=above $75k)
22) Zodiac                 - Zodiac sign (1=Aries ... 12=Pisces)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Seed and load data
RNG = 15906048
df = pd.read_csv('diabetes.csv')

X_cols = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'Myocardial',
          'PhysActivity', 'Fruit', 'Vegetables', 'HeavyDrinker',
          'HasHealthcare', 'NotAbleToAffordDoctor', 'GeneralHealth',
          'MentalHealth', 'PhysicalHealth', 'HardToClimbStairs',
          'BiologicalSex', 'AgeBracket', 'EducationBracket',
          'IncomeBracket', 'Zodiac']

X = df[X_cols].values
y = df['Diabetes'].values

# 70/30 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RNG, stratify=y)

print(f"Dataset: {df.shape[0]} rows | Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Diabetes prevalence: {y.mean()*100:.1f}%\n")


# Helper: find best predictor by AUC-drop (shuffle one column, measure AUC drop)
def find_best_predictor(model, X_tr, X_te, y_te, col_names, proba_func):
    base_auc = roc_auc_score(y_te, proba_func(X_te))
    drops = {}
    for i, col in enumerate(col_names):
        X_shuf = X_te.copy()
        np.random.seed(RNG)
        np.random.shuffle(X_shuf[:, i])
        drops[col] = base_auc - roc_auc_score(y_te, proba_func(X_shuf))
    return drops, max(drops, key=drops.get), base_auc


# Helper: plot ROC curve
def plot_roc(fpr, tpr, auc_val, title, filename, color='steelblue'):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=color, lw=2, label=f'AUC = {auc_val:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# Helper: plot feature importance bar chart
def plot_importance(drops, title, filename, color='steelblue', top_n=10):
    sorted_items = sorted(drops.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in sorted_items]
    vals  = [x[1] for x in sorted_items]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names[::-1], vals[::-1], color=color, edgecolor='black', height=0.6)
    ax.set_xlabel('AUC Drop When Shuffled', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# Q1 – Logistic Regression

print()
print()
print("Q1: Logistic Regression")

scaler_lr = StandardScaler()
X_train_lr = scaler_lr.fit_transform(X_train)
X_test_lr  = scaler_lr.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=RNG, class_weight='balanced')
lr.fit(X_train_lr, y_train)

y_prob_lr = lr.predict_proba(X_test_lr)[:, 1]
auc_lr    = roc_auc_score(y_test, y_prob_lr)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)

drops_lr, best_lr, _ = find_best_predictor(
    lr, X_train_lr, X_test_lr, y_test, X_cols,
    lambda x: lr.predict_proba(x)[:, 1])

print(f"AUC = {auc_lr:.3f}")
print(f"Best predictor: {best_lr}  (AUC drop = {drops_lr[best_lr]:.3f})")
print(classification_report(y_test, lr.predict(X_test_lr), digits=3))

plot_roc(fpr_lr, tpr_lr, auc_lr,
         'Q1 - Logistic Regression ROC Curve', 'q1_roc.png')
plot_importance(drops_lr,
                'Q1 - Logistic Regression: Predictor Importance (AUC Drop)',
                'q1_importance.png')
print("Figures saved: q1_roc.png, q1_importance.png\n")


# Q2 – SVM (LinearSVC, calibrated for probabilities)

print()
print()
print("Q2: SVM (Linear Kernel, calibrated)")

scaler_svm = StandardScaler()
X_train_svm = scaler_svm.fit_transform(X_train)
X_test_svm  = scaler_svm.transform(X_test)

svc_base = LinearSVC(max_iter=2000, random_state=RNG, class_weight='balanced', C=1.0)
svm = CalibratedClassifierCV(svc_base, cv=3)
svm.fit(X_train_svm, y_train)

y_prob_svm = svm.predict_proba(X_test_svm)[:, 1]
auc_svm    = roc_auc_score(y_test, y_prob_svm)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)

drops_svm, best_svm, _ = find_best_predictor(
    svm, X_train_svm, X_test_svm, y_test, X_cols,
    lambda x: svm.predict_proba(x)[:, 1])

print(f"AUC = {auc_svm:.3f}")
print(f"Best predictor: {best_svm}  (AUC drop = {drops_svm[best_svm]:.3f})")
print(classification_report(y_test, svm.predict(X_test_svm), digits=3))

plot_roc(fpr_svm, tpr_svm, auc_svm,
         'Q2 - SVM (Linear Kernel) ROC Curve', 'q2_roc.png', color='darkorange')
plot_importance(drops_svm,
                'Q2 - SVM: Predictor Importance (AUC Drop)',
                'q2_importance.png', color='darkorange')
print("Figures saved: q2_roc.png, q2_importance.png\n")


# Q3 – Single Decision Tree

print()
print()
print("Q3: Single Decision Tree")

dt = DecisionTreeClassifier(max_depth=6, random_state=RNG, class_weight='balanced')
dt.fit(X_train, y_train)

y_prob_dt = dt.predict_proba(X_test)[:, 1]
auc_dt    = roc_auc_score(y_test, y_prob_dt)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)

drops_dt, best_dt, _ = find_best_predictor(
    dt, X_train, X_test, y_test, X_cols,
    lambda x: dt.predict_proba(x)[:, 1])

print(f"AUC = {auc_dt:.3f}")
print(f"Best predictor: {best_dt}  (AUC drop = {drops_dt[best_dt]:.3f})")
print(classification_report(y_test, dt.predict(X_test), digits=3))

plot_roc(fpr_dt, tpr_dt, auc_dt,
         'Q3 - Decision Tree ROC Curve', 'q3_roc.png', color='seagreen')
plot_importance(drops_dt,
                'Q3 - Decision Tree: Predictor Importance (AUC Drop)',
                'q3_importance.png', color='seagreen')
print("Figures saved: q3_roc.png, q3_importance.png\n")



# Q4 – Random Forest

print()
print()
print("Q4: Random Forest")

rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                            random_state=RNG, class_weight='balanced',
                            n_jobs=-1)
rf.fit(X_train, y_train)

y_prob_rf = rf.predict_proba(X_test)[:, 1]
auc_rf    = roc_auc_score(y_test, y_prob_rf)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

drops_rf, best_rf, _ = find_best_predictor(
    rf, X_train, X_test, y_test, X_cols,
    lambda x: rf.predict_proba(x)[:, 1])

print(f"AUC = {auc_rf:.3f}")
print(f"Best predictor: {best_rf}  (AUC drop = {drops_rf[best_rf]:.3f})")
print(classification_report(y_test, rf.predict(X_test), digits=3))

plot_roc(fpr_rf, tpr_rf, auc_rf,
         'Q4 - Random Forest ROC Curve', 'q4_roc.png', color='mediumpurple')
plot_importance(drops_rf,
                'Q4 - Random Forest: Predictor Importance (AUC Drop)',
                'q4_importance.png', color='mediumpurple')
print("Figures saved: q4_roc.png, q4_importance.png\n")



# Q5 – AdaBoost

print()
print()
print("Q5: AdaBoost")

ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                         random_state=RNG)
ada.fit(X_train, y_train)

y_prob_ada = ada.predict_proba(X_test)[:, 1]
auc_ada    = roc_auc_score(y_test, y_prob_ada)
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_prob_ada)

drops_ada, best_ada, _ = find_best_predictor(
    ada, X_train, X_test, y_test, X_cols,
    lambda x: ada.predict_proba(x)[:, 1])

print(f"AUC = {auc_ada:.3f}")
print(f"Best predictor: {best_ada}  (AUC drop = {drops_ada[best_ada]:.3f})")
print(classification_report(y_test, ada.predict(X_test), digits=3))

plot_roc(fpr_ada, tpr_ada, auc_ada,
         'Q5 - AdaBoost ROC Curve', 'q5_roc.png', color='firebrick')
plot_importance(drops_ada,
                'Q5 - AdaBoost: Predictor Importance (AUC Drop)',
                'q5_importance.png', color='firebrick')
print("Figures saved: q5_roc.png, q5_importance.png\n")



# Extra Credit a – Which model is best?

print()
print()
print("Extra Credit a: Model Comparison")

models = {'Logistic Regression': auc_lr,
          'SVM (Linear)':        auc_svm,
          'Decision Tree':       auc_dt,
          'Random Forest':       auc_rf,
          'AdaBoost':            auc_ada}

for name, auc in sorted(models.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:<22}  AUC = {auc:.3f}")

best_model = max(models, key=models.get)
print(f"\nBest model: {best_model}")

# Combined ROC plot
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot(fpr_lr,  tpr_lr,  color='steelblue',    lw=2, label=f'Logistic Regression (AUC={auc_lr:.3f})')
ax.plot(fpr_svm, tpr_svm, color='darkorange',   lw=2, label=f'SVM Linear         (AUC={auc_svm:.3f})')
ax.plot(fpr_dt,  tpr_dt,  color='seagreen',     lw=2, label=f'Decision Tree      (AUC={auc_dt:.3f})')
ax.plot(fpr_rf,  tpr_rf,  color='mediumpurple', lw=2, label=f'Random Forest      (AUC={auc_rf:.3f})')
ax.plot(fpr_ada, tpr_ada, color='firebrick',    lw=2, label=f'AdaBoost           (AUC={auc_ada:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('Extra Credit a – All Models: ROC Comparison', fontsize=10, fontweight='bold')
ax.legend(fontsize=7.5, loc='lower right')
plt.tight_layout()
plt.savefig('eca_roc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved: eca_roc_comparison.png")




# Extra Credit b – Something interesting

print()
print()
print("Extra Credit b: Something Interesting")

# Zodiac vs diabetes rate -- does your star sign predict diabetes?
zodiac_names = {1:'Aries',2:'Taurus',3:'Gemini',4:'Cancer',5:'Leo',
                6:'Virgo',7:'Libra',8:'Scorpio',9:'Sagittarius',
                10:'Capricorn',11:'Aquarius',12:'Pisces'}

df['ZodiacName'] = df['Zodiac'].map(zodiac_names)
zodiac_rates = df.groupby('ZodiacName')['Diabetes'].mean().sort_values(ascending=False)
print("\nDiabetes rate by zodiac sign:")
print(zodiac_rates.round(4))

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(zodiac_rates.index, zodiac_rates.values * 100,
       color='steelblue', edgecolor='black')
ax.axhline(y=df['Diabetes'].mean()*100, color='red', linestyle='--',
           lw=1.5, label=f'Overall rate = {df["Diabetes"].mean()*100:.1f}%')
ax.set_ylabel('Diabetes Rate (%)', fontsize=9)
ax.set_title('Extra Credit b – Diabetes Rate by Zodiac Sign', fontsize=10, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('ecb_zodiac.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved: ecb_zodiac.png")

print("\nThe End.")
