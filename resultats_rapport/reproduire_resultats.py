"""
Reproduit les résultats de projet.ipynb (série A : division 60/20/20, cellules 0 à 40),
dont le Naive Bayes a été sauvegardé dans model_naive_bayes.pkl (cellule 61). Lit ce .pkl sans le modifier. Ne modifie aucun fichier du projet : écrit seulement dans resultats_rapport/.

Le code de préparation et d'entraînement est repris tel quel du notebook.
Seule différence : le notebook calcule l'âge et last_mvt avec pd.Timestamp.today().
On retrouve la date d'exécution d'origine en cherchant celle qui redonne exactement
les paramètres du modèle sauvegardé (model_naive_bayes.pkl).

Lancement (depuis la racine du projet) :  python resultats_rapport/reproduire_resultats.py
"""
import json
import os as _os
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # évite un avertissement de joblib sous Windows
import os
import pickle
import warnings
from datetime import datetime, timedelta

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

warnings.filterwarnings('ignore')

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
DATA = os.path.join(RACINE, 'data.csv')


# ------------------------------------------------------------------
# Préparation : cellules 0 à 31 de projet.ipynb, avec "aujourd'hui" en paramètre
# ------------------------------------------------------------------
def preparer(aujourd_hui):
    bd = pd.read_csv(DATA)
    bd = bd.drop(columns=['age'])
    bd['m-pay_sub_birth_date'] = pd.to_datetime(bd['m-pay_sub_birth_date'], format='%d/%m/%Y')
    bd['age'] = (aujourd_hui - bd['m-pay_sub_birth_date']).dt.days // 365
    bd['m-pay_sub_status_1_last_modif_date'] = pd.to_datetime(bd['m-pay_sub_status_1_last_modif_date'])
    bd['last_mvt'] = (aujourd_hui - bd['m-pay_sub_status_1_last_modif_date']).dt.days
    bd = bd.drop(columns='m-pay_sub_status_1_last_modif_date')
    bd = bd.drop(columns='m-pay_sub_birth_date')

    def surendettement(taux):
        if taux > 1:
            return 1
        else:
            return 0
    bd['surendettement'] = bd['taux_endettement'].apply(surendettement)

    ordre = {'Petite': 0, 'Moyenne': 1, 'Grande': 2}
    bd['taille_entreprise'] = bd['taille_entreprise'].map(ordre)
    bd = pd.get_dummies(bd, columns=['m-pay_sub_city', 'type_contrat', 'm-pay_sub_status_1'])
    colonnes_bool = bd.select_dtypes('bool').columns
    bd[colonnes_bool] = bd[colonnes_bool].astype(int)
    return bd


def diviser(bd_model):
    """Cellule 32 de projet.ipynb."""
    labels = np.array(bd_model['decision'])
    features = bd_model.drop(['decision', 'Unnamed: 0', 'm-pay_sub_msisdn', 'm-pay_sub_client_bankcode'], axis=1)
    feature_list = list(features.columns)
    features = np.array(features)
    idx = np.arange(len(labels))
    train_features, temp_features, train_labels, temp_labels, idx_train, idx_temp = train_test_split(
        features, labels, idx, test_size=0.4, random_state=42)
    val_features, test_features, val_labels, test_labels = train_test_split(
        temp_features, temp_labels, test_size=0.5, random_state=42)
    return feature_list, train_features, val_features, test_features, train_labels, val_labels, test_labels, idx_train


# ------------------------------------------------------------------
# 1. Retrouver la date d'exécution du notebook
# ------------------------------------------------------------------
nb_sauve = pickle.load(open(os.path.join(RACINE, 'model_naive_bayes.pkl'), 'rb'))

brut = pd.read_csv(DATA)
modif = pd.to_datetime(brut['m-pay_sub_status_1_last_modif_date'])
# last_mvt moyen appris (toutes classes) -> estimation grossière de la date
moyenne_last_mvt = (nb_sauve.theta_[:, 9] * nb_sauve.class_count_).sum() / nb_sauve.class_count_.sum()
estimation = modif.mean() + timedelta(days=float(moyenne_last_mvt))

#  Balayage minute par minute sur last_mvt seul (rapide), puis vérification du modèle complet
idx_tr, _, y_tr, _ = train_test_split(np.arange(len(brut)), brut['decision'].values, test_size=0.4, random_state=42)
date_trouvee = None
debut = (estimation - timedelta(days=3)).normalize()
for minutes in range(0, 6 * 24 * 60):
    candidate = debut + timedelta(minutes=minutes)
    lm = (candidate - modif).dt.days.values[idx_tr]
    mu = [lm[y_tr == 0].mean(), lm[y_tr == 1].mean()]
    if not np.allclose(mu, nb_sauve.theta_[:, 9]):
        continue
    bd = preparer(candidate)
    fl, Xtr, *_, ytr, _, _, _ = diviser(bd)
    m = GaussianNB().fit(Xtr, ytr)
    if np.allclose(m.theta_, nb_sauve.theta_) and np.allclose(m.var_, nb_sauve.var_):
        date_trouvee = candidate
        break

if date_trouvee is None:
    raise SystemExit("Impossible de retrouver la date d'exécution : les résultats ne peuvent pas être reproduits à l'identique.")
print(f"Date d'exécution retrouvée : {date_trouvee:%d/%m/%Y %H:%M} (le modèle ré-entraîné est identique au .pkl)")

bd_model = preparer(date_trouvee)
(feature_list, train_features, val_features, test_features,
 train_labels, val_labels, test_labels, idx_train) = diviser(bd_model)

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_scaled = scaler.transform(val_features)
test_features_scaled = scaler.transform(test_features)

# ------------------------------------------------------------------
# 2. Modèles : cellules 33 à 40, mêmes hyperparamètres et mêmes données (brutes ou normalisées)
# ------------------------------------------------------------------
modeles = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42), False),
    ('SVM (noyau RBF)', SVC(probability=True, kernel='rbf'), True),
    ('KNN (k = 5)', KNeighborsClassifier(n_neighbors=5), True),
    ('Naive Bayes (gaussien)', GaussianNB(), False),
    ('Perceptron', Perceptron(random_state=42), True),
    ('SGD Classifier (log_loss)', SGDClassifier(loss='log_loss', random_state=42), True),
    ('Régression logistique', LogisticRegression(), True),
    ('Arbre de décision', DecisionTreeClassifier(), True),
]

# Sorties enregistrées dans le notebook (accuracy et matrices de confusion) pour vérification
NOTEBOOK = {
    'Random Forest': (1.0, [[406, 0], [0, 188]], 1.0, [[395, 0], [0, 199]]),
    'SVM (noyau RBF)': (0.9781144781144782, [[404, 2], [11, 177]], 0.9730639730639731, [[390, 5], [11, 188]]),
    'KNN (k = 5)': (0.9326599326599326, [[396, 10], [30, 158]], 0.9242424242424242, [[373, 22], [23, 176]]),
    'Naive Bayes (gaussien)': (0.9797979797979798, [[399, 7], [5, 183]], 0.9781144781144782, [[387, 8], [5, 194]]),
    'Perceptron': (0.9377104377104377, [[384, 22], [15, 173]], 0.9494949494949495, [[374, 21], [9, 190]]),
    'SGD Classifier (log_loss)': (0.9562289562289562, [[397, 9], [17, 171]], 0.9629629629629629, [[386, 9], [13, 186]]),
    'Régression logistique': (0.9696969696969697, [[402, 4], [14, 174]], 0.9713804713804713, [[388, 7], [10, 189]]),
    'Arbre de décision': (1.0, [[406, 0], [0, 188]], 1.0, [[395, 0], [0, 199]]),
}


def mesures(y, y_pred):
    p, r, f, _ = metrics.precision_recall_fscore_support(y, y_pred, labels=[0, 1], zero_division=0)
    return {
        'accuracy': metrics.accuracy_score(y, y_pred),
        'precision_accord': p[1], 'rappel_accord': r[1], 'f1_accord': f[1],
        'precision_refus': p[0], 'rappel_refus': r[0], 'f1_refus': f[0],
        'precision_macro': p.mean(), 'rappel_macro': r.mean(), 'f1_macro': f.mean(),
        'matrice': metrics.confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
    }


resultats = {}
for nom, modele, normalise in modeles:
    Xtr, Xv, Xte = (train_features_scaled, val_features_scaled, test_features_scaled) if normalise \
        else (train_features, val_features, test_features)
    modele.fit(Xtr, train_labels)
    val = mesures(val_labels, modele.predict(Xv))
    test = mesures(test_labels, modele.predict(Xte))
    acc_v, cm_v, acc_t, cm_t = NOTEBOOK[nom]
    conforme = (np.isclose(val['accuracy'], acc_v) and val['matrice'] == cm_v
                and np.isclose(test['accuracy'], acc_t) and test['matrice'] == cm_t)
    resultats[nom] = {'validation': val, 'test': test, 'normalise': normalise, 'conforme_notebook': bool(conforme)}
    print(f"{nom:28s} val {val['accuracy']:.4f}  test {test['accuracy']:.4f}  "
          f"{'identique au notebook' if conforme else 'DIFFÉRENT du notebook'}")

# ------------------------------------------------------------------
# 3. Données, cible, et lignes de bd_validation.csv vues à l'entraînement
# ------------------------------------------------------------------
cible = brut['decision'].value_counts().sort_index()
bd_val = pd.read_csv(os.path.join(RACINE, 'bd_validation.csv'))
cles = ['Unnamed: 0']
ids_train = set(bd_model.iloc[idx_train]['Unnamed: 0'])
deja_vues = int(bd_val['Unnamed: 0'].isin(ids_train).sum())

infos = {
    'date_execution': date_trouvee.strftime('%Y-%m-%d %H:%M'),
    'fichier': 'data.csv',
    'lignes': int(brut.shape[0]), 'colonnes': int(brut.shape[1]),
    'variables': brut.columns.tolist(),
    'valeurs_manquantes': int(brut.isna().sum().sum()),
    'doublons': int(brut.duplicated().sum()),
    'cible': {'refus (0)': int(cible.get(0, 0)), 'accord (1)': int(cible.get(1, 0))},
    'colonnes_apres_preparation': int(bd_model.shape[1]),
    'variables_modele': feature_list,
    'tailles': {'train': int(len(train_labels)), 'validation': int(len(val_labels)), 'test': int(len(test_labels))},
    'repartition_test': {'refus (0)': int((test_labels == 0).sum()), 'accord (1)': int((test_labels == 1).sum())},
    'repartition_validation': {'refus (0)': int((val_labels == 0).sum()), 'accord (1)': int((val_labels == 1).sum())},
    'surendettement_1': int((bd_model['surendettement'] == 1).sum()),
    'bd_validation_lignes_vues_en_train': deja_vues,
    'resultats': resultats,
}
json.dump(infos, open(os.path.join(ICI, 'resultats_serie_a.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 4. Figures
# ------------------------------------------------------------------
BLEU, VERT, ROUGE = '#1A3A8F', '#17A673', '#E24B4A'
plt.rcParams['font.family'] = 'DejaVu Sans'

# Matrice de confusion du Naive Bayes sur l'ensemble de test
cm = np.array(resultats['Naive Bayes (gaussien)']['test']['matrice'])
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
acc_nb = resultats['Naive Bayes (gaussien)']['test']['accuracy']
ax.set_title(f"Matrice de confusion — Naive Bayes (ensemble de test, n = {cm.sum()})\nAccuracy = {acc_nb * 100:.2f} %",
             fontsize=12, pad=12)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(ICI, 'matrice_confusion_serie_a.png'), dpi=200)
plt.close(fig)

# Répartition de la variable cible
n_refus, n_accord = int(cible.get(0, 0)), int(cible.get(1, 0))
total = n_refus + n_accord
fig, ax = plt.subplots(figsize=(6.4, 4.6))
barres = ax.bar(['Refus (0)', 'Accord (1)'], [n_refus, n_accord], color=[ROUGE, VERT], width=0.55)
for b, n in zip(barres, [n_refus, n_accord]):
    ax.text(b.get_x() + b.get_width() / 2, n + total * 0.012, f"{n}  ({n / total * 100:.1f} %)",
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#14213D')
ax.set_ylabel('Nombre de demandes', fontsize=11)
ax.set_title(f"Répartition de la variable cible « decision » (data.csv, n = {total})", fontsize=12, pad=12)
ax.set_ylim(0, max(n_refus, n_accord) * 1.15)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#ECEFF6')
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(ICI, 'repartition_cible.png'), dpi=200)
plt.close(fig)

print("\nFichiers écrits dans resultats_rapport/ : resultats_serie_a.json, matrice_confusion_serie_a.png, repartition_cible.png")
