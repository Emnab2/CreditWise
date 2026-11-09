"""
Deuxième série de projet.ipynb (cellules 44 à 58) : 20 dossiers mis de côté
(bd_model.sample(n=20, random_state=42)), puis 75 % / 25 % des 2 950 restants (random_state=42).

Préparation : celle des cellules 0 à 31 du notebook (surendettement = 1 si taux_endettement > 1).
Date de référence pour l'âge et last_mvt : 06/08/2026 09:13, date d'exécution d'origine de cette série
(heure de création de bd_validation.csv).

Lit seulement data.csv et bd_validation.csv ; écrit uniquement dans resultats_rapport/
(resultats_serie_b.json, matrice_confusion.png). Ne lit ni n'écrit aucun fichier .pkl.
Lancement :  python resultats_rapport/serie_b.py
"""
import json
import os

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')   # évite un avertissement de joblib sous Windows
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
DATE_REFERENCE = pd.Timestamp('2026-08-06 09:13')
SEUIL_SURENDETTEMENT = 1     # comme dans projet.ipynb (cellule 16)

# Ordre des variables attendu par l'application (app.FEATURE_ORDER)
FEATURE_ORDER = [
    'm-pay_sub_solde', 'SoldeCashBack', 'revenu', 'anciennete', 'credits', 'retard',
    'taille_entreprise', 'taux_endettement', 'age', 'last_mvt', 'surendettement',
    'm-pay_sub_city_Nabeul', 'm-pay_sub_city_Sousse', 'm-pay_sub_city_Tunis',
    'type_contrat_CDD', 'type_contrat_CDI', 'type_contrat_Freelance',
    'm-pay_sub_status_1_Désinscrit', 'm-pay_sub_status_1_Enregistré', 'm-pay_sub_status_1_Inscrit',
]


# ------------------------------------------------------------------
# Préparation (cellules 0 à 31)
# ------------------------------------------------------------------
bd = pd.read_csv(os.path.join(RACINE, 'data.csv'))
bd = bd.drop(columns=['age'])
bd['m-pay_sub_birth_date'] = pd.to_datetime(bd['m-pay_sub_birth_date'], format='%d/%m/%Y')
bd['age'] = (DATE_REFERENCE - bd['m-pay_sub_birth_date']).dt.days // 365
bd['m-pay_sub_status_1_last_modif_date'] = pd.to_datetime(bd['m-pay_sub_status_1_last_modif_date'])
bd['last_mvt'] = (DATE_REFERENCE - bd['m-pay_sub_status_1_last_modif_date']).dt.days
bd = bd.drop(columns='m-pay_sub_status_1_last_modif_date')
bd = bd.drop(columns='m-pay_sub_birth_date')
bd['surendettement'] = (bd['taux_endettement'] > SEUIL_SURENDETTEMENT).astype(int)
bd['taille_entreprise'] = bd['taille_entreprise'].map({'Petite': 0, 'Moyenne': 1, 'Grande': 2})
bd = pd.get_dummies(bd, columns=['m-pay_sub_city', 'type_contrat', 'm-pay_sub_status_1'])
colonnes_bool = bd.select_dtypes('bool').columns
bd[colonnes_bool] = bd[colonnes_bool].astype(int)
bd_model = bd.copy()

# ------------------------------------------------------------------
# Cellules 44 à 47 : mise de côté de 20 dossiers, division 75 / 25, normalisation
# ------------------------------------------------------------------
bd_validation = bd_model.sample(n=20, random_state=42)
bd_reste = bd_model.drop(bd_validation.index)

def separer(table):
    labels = np.array(table['decision'])
    features = table.drop(['decision', 'Unnamed: 0', 'm-pay_sub_msisdn', 'm-pay_sub_client_bankcode'], axis=1)
    return list(features.columns), np.array(features), labels

feature_list, features, labels = separer(bd_reste)
assert feature_list == FEATURE_ORDER, "L'ordre des variables ne correspond pas à celui de l'application"
_, val_features, val_labels = separer(bd_validation)

# Les 20 dossiers sont les mêmes que dans bd_validation.csv
ids_csv = sorted(pd.read_csv(os.path.join(RACINE, 'bd_validation.csv'))['Unnamed: 0'])
assert sorted(bd_validation['Unnamed: 0']) == ids_csv, "Les 20 dossiers diffèrent de bd_validation.csv"

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42
)
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)
val_features_scaled = scaler.transform(val_features)

print(f"Entraînement : {train_features.shape}  Test : {test_features.shape}  Mis de côté : {val_features.shape}")

# ------------------------------------------------------------------
# Cellules 48 à 58 : les 8 modèles, mêmes paramètres et mêmes données (brutes ou normalisées)
# (la cellule 53 refait une Random Forest sans random_state : doublon non retenu)
# ------------------------------------------------------------------
modeles = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42), False),
    ('SVM (noyau RBF)', SVC(kernel='rbf'), True),
    ('KNN (k = 5)', KNeighborsClassifier(n_neighbors=5), True),
    ('Arbre de décision', DecisionTreeClassifier(random_state=42), False),
    ('Naive Bayes (gaussien)', GaussianNB(), False),
    ('Perceptron', Perceptron(random_state=42), True),
    ('Régression logistique', LogisticRegression(), True),
    ('SGD Classifier', SGDClassifier(random_state=42), True),
]

def mesures(y, y_pred):
    p, r, f, s = metrics.precision_recall_fscore_support(y, y_pred, labels=[0, 1], zero_division=0)
    return {
        'accuracy': metrics.accuracy_score(y, y_pred),
        'precision_accord': p[1], 'rappel_accord': r[1], 'f1_accord': f[1],
        'precision_refus': p[0], 'rappel_refus': r[0], 'f1_refus': f[0],
        'precision_macro': p.mean(), 'rappel_macro': r.mean(), 'f1_macro': f.mean(),
        'effectifs': s.tolist(),
        'matrice': metrics.confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
    }

resultats, entraines = {}, {}
for nom, modele, normalise in modeles:
    Xtr, Xte, Xva = (train_features_scaled, test_features_scaled, val_features_scaled) if normalise \
        else (train_features, test_features, val_features)
    modele.fit(Xtr, train_labels)
    entraines[nom] = modele
    resultats[nom] = {
        'normalise': normalise,
        'test': mesures(test_labels, modele.predict(Xte)),
        'mis_de_cote': mesures(val_labels, modele.predict(Xva)),
    }
    t, v = resultats[nom]['test'], resultats[nom]['mis_de_cote']
    print(f"{nom:24s} test acc={t['accuracy']:.4f} P={t['precision_accord']:.4f} R={t['rappel_accord']:.4f} "
          f"F1={t['f1_accord']:.4f} {t['matrice']} | 20 dossiers : {v['accuracy']:.2f} "
          f"({round(v['accuracy'] * 20)}/20)")

infos = {
    'date_reference': str(DATE_REFERENCE), 'seuil_surendettement': SEUIL_SURENDETTEMENT,
    'surendettement_1': int(bd_model['surendettement'].sum()),
    'tailles': {'entrainement': int(len(train_labels)), 'test': int(len(test_labels)), 'mis_de_cote': int(len(val_labels))},
    'repartition': {
        'entrainement': np.bincount(train_labels).tolist(),
        'test': np.bincount(test_labels).tolist(),
        'mis_de_cote': np.bincount(val_labels, minlength=2).tolist(),
    },
    'ids_mis_de_cote': [int(i) for i in bd_validation['Unnamed: 0']],
    'resultats': resultats,
}
json.dump(infos, open(os.path.join(ICI, 'resultats_serie_b.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# Matrice de confusion du Naive Bayes (ensemble de test)
# ------------------------------------------------------------------
cm = np.array(resultats['Naive Bayes (gaussien)']['test']['matrice'])
acc_nb = resultats['Naive Bayes (gaussien)']['test']['accuracy']
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(figsize=(6.2, 5.4))
ax.imshow(cm, cmap='Blues')
etiquettes = ['Refus (0)', 'Accord (1)']
noms_cases = [['Vrais négatifs', 'Faux positifs'], ['Faux négatifs', 'Vrais positifs']]
for i in range(2):
    for j in range(2):
        couleur = 'white' if cm[i, j] > cm.max() / 2 else '#14213D'
        ax.text(j, i - 0.08, str(cm[i, j]), ha='center', va='center', fontsize=22, fontweight='bold', color=couleur)
        ax.text(j, i + 0.18, f"{noms_cases[i][j]}\n({cm[i, j] / cm.sum() * 100:.1f} %)", ha='center', va='center',
                fontsize=9.5, color=couleur)
ax.set_xticks([0, 1], etiquettes, fontsize=11)
ax.set_yticks([0, 1], etiquettes, fontsize=11)
ax.set_xlabel('Classe prédite', fontsize=12, labelpad=10)
ax.set_ylabel('Classe réelle', fontsize=12, labelpad=10)
ax.set_title(f"Matrice de confusion — Naive Bayes (ensemble de test, n = {cm.sum()})\nAccuracy = {acc_nb * 100:.2f} %",
             fontsize=12, pad=12)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(ICI, 'matrice_confusion.png'), dpi=200)
plt.close(fig)
print("\nresultats_serie_b.json et matrice_confusion.png écrits dans resultats_rapport/")

