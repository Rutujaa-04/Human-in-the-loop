"""
train_model.py — Retrain the resume screening ensemble model
            with the expanded 54-category → 16-group dataset.

Run:  python train_model.py
Output: model.pkl, vectorizer.pkl, char_vectorizer.pkl, label_encoder.pkl
"""

import pickle
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack

# ──────────────────────────────────────────────
# 1. GROUP MAP  (raw label → semantic group)
#    Maps all 54 known category variants to 16 groups
# ──────────────────────────────────────────────
GROUP_MAP = {
    # Tech
    'information technology'      : 'tech',
    'information-technology'      : 'tech',
    'engineering'                 : 'tech',
    'electrical engineering'      : 'tech',
    'mechanical engineer'         : 'tech',
    'java developer'              : 'tech',
    'python developer'            : 'tech',
    'react developer'             : 'tech',
    'dotnet developer'            : 'tech',
    'sap developer'               : 'tech',
    'data science'                : 'tech',
    'etl developer'               : 'tech',
    'sql developer'               : 'tech',
    'devops'                      : 'tech',
    'database'                    : 'tech',
    'testing'                     : 'tech',
    'network security engineer'   : 'tech',
    'blockchain'                  : 'tech',
    'web designing'               : 'tech',

    # Finance
    'finance'                     : 'finance',
    'accountant'                  : 'finance',
    'banking'                     : 'finance',

    # Management
    'management'                  : 'management',
    'consultant'                  : 'management',
    'operations manager'          : 'management',
    'business analyst'            : 'management',
    'pmo'                         : 'management',
    'business-development'        : 'management',
    'business development'        : 'management',

    # HR
    'human resources'             : 'hr',
    'hr'                          : 'hr',

    # Sales
    'sales'                       : 'sales',
    'public relations'            : 'sales',
    'public-relations'            : 'sales',

    # Legal
    'advocate'                    : 'legal',

    # Healthcare
    'healthcare'                  : 'healthcare',

    # Creative
    'arts'                        : 'creative',
    'digital media'               : 'creative',
    'digital-media'               : 'creative',
    'apparel'                     : 'creative',
    'designing'                   : 'creative',
    'designer'                    : 'creative',
    'architecture'                : 'creative',

    # Education
    'education'                   : 'education',
    'teacher'                     : 'education',

    # Hospitality
    'chef'                        : 'hospitality',
    'food and beverages'          : 'hospitality',

    # Aviation
    'aviation'                    : 'aviation',

    # Construction
    'construction'                : 'construction',
    'building and construction'   : 'construction',
    'civil engineer'              : 'construction',

    # Fitness
    'fitness'                     : 'fitness',
    'health and fitness'          : 'fitness',

    # Agriculture
    'agriculture'                 : 'agriculture',

    # Automobile
    'automobile'                  : 'automobile',

    # BPO
    'bpo'                         : 'bpo',
}

# ──────────────────────────────────────────────
# 2. TEXT CLEANING
# ──────────────────────────────────────────────
def clean_resume(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_stopwords(text):
    return ' '.join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

def advanced_clean(text):
    text = clean_resume(text)
    text = remove_stopwords(text)
    return ' '.join(w for w in text.split() if 3 <= len(w) <= 15)

# ──────────────────────────────────────────────
# 3. LOAD DATA
# ──────────────────────────────────────────────
print("Loading data...")
# Try multiple possible CSV names / paths
import os
for fname in ['Resume.csv', 'cleaned_resume_dataset.csv', 'cleaned_multiclass_resume.csv']:
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        print(f"  Loaded: {fname}  ({len(df)} rows)")
        break
else:
    raise FileNotFoundError("No resume CSV found in current directory.")

# Detect text column
text_col = next((c for c in df.columns if 'resume' in c.lower()), df.columns[0])
cat_col  = next((c for c in df.columns if 'categ' in c.lower()), df.columns[-1])
print(f"  Text column : '{text_col}'")
print(f"  Label column: '{cat_col}'")

df = df[[text_col, cat_col]].rename(columns={text_col: 'Resume_str', cat_col: 'Category'})
df = df.dropna()
df['Category'] = df['Category'].str.lower().str.strip()

print(f"\nRaw category counts:\n{df['Category'].value_counts().to_string()}")

# ──────────────────────────────────────────────
# 4. APPLY GROUP MAP
# ──────────────────────────────────────────────
df['Category'] = df['Category'].map(
    lambda c: GROUP_MAP.get(c, c)   # unmapped categories keep their name
)
df = df.dropna()

print(f"\nGrouped category counts:\n{df['Category'].value_counts().to_string()}")

# ──────────────────────────────────────────────
# 5. CLEAN TEXT
# ──────────────────────────────────────────────
print("\nCleaning text...")
df['cleaned'] = df['Resume_str'].apply(advanced_clean)

# ──────────────────────────────────────────────
# 6. ENCODE + SPLIT
# ──────────────────────────────────────────────
le = LabelEncoder()
df['label'] = le.fit_transform(df['Category'])
print(f"\nClasses ({len(le.classes_)}): {list(le.classes_)}")

X = df['cleaned']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ──────────────────────────────────────────────
# 7. VECTORIZE  (improved features)
# ──────────────────────────────────────────────
print("\nVectorizing...")
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
)
char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 6),
    max_features=8000,
    min_df=2
)

X_train_word = vectorizer.fit_transform(X_train)
X_test_word  = vectorizer.transform(X_test)
X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char  = char_vectorizer.transform(X_test)

X_train_vec = hstack([X_train_word, X_train_char])
X_test_vec  = hstack([X_test_word,  X_test_char])
print(f"Feature matrix: {X_train_vec.shape}")

# ──────────────────────────────────────────────
# 8. TRAIN ENSEMBLE
# ──────────────────────────────────────────────
svm_base = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000)
svm_cal  = CalibratedClassifierCV(svm_base, cv=3)

log_reg  = LogisticRegression(
    C=5.0, max_iter=1000, class_weight='balanced',
    solver='lbfgs'
)

rf = RandomForestClassifier(
    n_estimators=300, class_weight='balanced',
    min_samples_leaf=2, n_jobs=-1, random_state=42
)

model = VotingClassifier(
    estimators=[('svm', svm_cal), ('logreg', log_reg), ('rf', rf)],
    voting='soft',
    weights=[3, 2, 1]
)

print("\nTraining ensemble (SVM + LogReg + RandomForest)...")
print("This may take 3-5 minutes...")
model.fit(X_train_vec, y_train)
print("Training complete!")

# ──────────────────────────────────────────────
# 9. EVALUATE
# ──────────────────────────────────────────────
y_pred = model.predict(X_test_vec)
acc    = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14, 11))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, linewidths=0.3)
plt.title(f'Confusion Matrix — Ensemble ({acc*100:.1f}%)', fontsize=13, fontweight='bold')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig('confusion_matrix_new.png', dpi=150, bbox_inches='tight')
plt.show()
print("Confusion matrix saved → confusion_matrix_new.png")

# ──────────────────────────────────────────────
# 10. SAVE MODEL FILES
# ──────────────────────────────────────────────
print("\nSaving model files...")
with open('model.pkl',           'wb') as f: pickle.dump(model,          f)
with open('vectorizer.pkl',      'wb') as f: pickle.dump(vectorizer,     f)
with open('char_vectorizer.pkl', 'wb') as f: pickle.dump(char_vectorizer, f)
with open('label_encoder.pkl',   'wb') as f: pickle.dump(le,             f)
print("Saved: model.pkl  vectorizer.pkl  char_vectorizer.pkl  label_encoder.pkl")
print("\nDone! Restart Streamlit to use the new model.")
