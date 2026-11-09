"""
Explicabilité SHAP calculée exactement comme dans app.py, pour le rapport.

Reprend ligne pour ligne le code SHAP de app.py (lignes 102-129 et fonction evaluer_client),
sans importer app.py, pour ne toucher à aucun fichier du projet (ni bank.db).
Lit : model_naive_bayes.pkl, bd_validation.csv, static/summary_plot.png (comparaison).
Écrit uniquement dans resultats_rapport/xai/.

Lancement (depuis la racine du projet) :  python resultats_rapport/xai/xai.py
"""
import json
import os
import pickle
import warnings

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # même backend que app.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(ICI))

# --- Copié de app.py ---
FEATURE_ORDER = [
    'm-pay_sub_solde', 'SoldeCashBack', 'revenu', 'anciennete', 'credits', 'retard',
    'taille_entreprise', 'taux_endettement', 'age', 'last_mvt', 'surendettement',
    'm-pay_sub_city_Nabeul', 'm-pay_sub_city_Sousse', 'm-pay_sub_city_Tunis',
    'type_contrat_CDD', 'type_contrat_CDI', 'type_contrat_Freelance',
    'm-pay_sub_status_1_Désinscrit', 'm-pay_sub_status_1_Enregistré', 'm-pay_sub_status_1_Inscrit'
]
FEATURE_LABELS = {
    'm-pay_sub_solde': 'Solde du compte',
    'SoldeCashBack': 'Solde cashback',
    'revenu': 'Revenu',
    'anciennete': 'Ancienneté',
    'credits': 'Crédits en cours',
    'retard': 'Retards de paiement',
    'taille_entreprise': "Taille de l'entreprise",
    'taux_endettement': "Taux d'endettement",
    'age': 'Âge',
    'last_mvt': 'Dernier mouvement (jours)',
    'surendettement': 'Surendettement',
    'm-pay_sub_city_Nabeul': 'Ville : Nabeul',
    'm-pay_sub_city_Sousse': 'Ville : Sousse',
    'm-pay_sub_city_Tunis': 'Ville : Tunis',
    'type_contrat_CDD': 'Contrat CDD',
    'type_contrat_CDI': 'Contrat CDI',
    'type_contrat_Freelance': 'Contrat Freelance',
    'm-pay_sub_status_1_Désinscrit': 'Statut : Désinscrit',
    'm-pay_sub_status_1_Enregistré': 'Statut : Enregistré',
    'm-pay_sub_status_1_Inscrit': 'Statut : Inscrit',
}

clf_nb = pickle.load(open(os.path.join(RACINE, 'model_naive_bayes.pkl'), 'rb'))
_df_background = pd.read_csv(os.path.join(RACINE, 'bd_validation.csv'))
background_features = _df_background[FEATURE_ORDER].values
explainer = shap.KernelExplainer(clf_nb.predict_proba, background_features)

# ------------------------------------------------------------------
# 1. Explication globale (app.py, lignes 116-129)
# ------------------------------------------------------------------
np.random.seed(42)
_shap_values_bg = explainer.shap_values(background_features, silent=True)

plt.figure()
shap.summary_plot(_shap_values_bg[:, :, 1], background_features, feature_names=FEATURE_ORDER, show=False)
plt.tight_layout()
chemin_global = os.path.join(ICI, 'shap_global.png')
plt.savefig(chemin_global, dpi=120, bbox_inches='tight')
plt.close()

# Comparaison avec l'image mise en cache par l'application
image_app = os.path.join(RACINE, 'static', 'summary_plot.png')
identique = None
if os.path.exists(image_app):
    a, b = plt.imread(image_app), plt.imread(chemin_global)
    identique = bool(a.shape == b.shape and np.array_equal(a, b))

# Classement : summary_plot trie les variables par moyenne des valeurs absolues
moyenne_abs = np.abs(_shap_values_bg[:, :, 1]).mean(axis=0)
moyenne_signee = _shap_values_bg[:, :, 1].mean(axis=0)
ordre = np.argsort(-moyenne_abs)
classement = [{
    'rang': r + 1,
    'variable': FEATURE_ORDER[i],
    'libelle': FEATURE_LABELS[FEATURE_ORDER[i]],
    'moyenne_abs_shap': float(moyenne_abs[i]),
    'moyenne_shap': float(moyenne_signee[i]),
} for r, i in enumerate(ordre)]

# ------------------------------------------------------------------
# 2. Explications locales (fonction evaluer_client de app.py)
# ------------------------------------------------------------------
proba = clf_nb.predict_proba(background_features)[:, 1]
pred = clf_nb.predict(background_features)
reel = _df_background['decision'].values


def expliquer(position, nom_fichier):
    features = background_features[position:position + 1]          # client_to_features -> (1, 20)
    prediction = clf_nb.predict(features)[0]
    probabilite = clf_nb.predict_proba(features)[0]

    np.random.seed(42)
    shap_values = explainer.shap_values(features, silent=True)
    contributions = shap_values[0, :, 1]

    explication = shap.Explanation(
        values=contributions,
        base_values=explainer.expected_value[1],
        data=features[0],
        feature_names=FEATURE_ORDER
    )
    plt.figure()
    shap.plots.waterfall(explication, max_display=8, show=False)
    plt.savefig(os.path.join(ICI, nom_fichier), format='png', dpi=120, bbox_inches='tight')
    plt.close()

    ordre_local = np.argsort(-np.abs(contributions))
    return {
        'position_dans_bd_validation': int(position),
        'identifiant_Unnamed_0': int(_df_background['Unnamed: 0'].iloc[position]),
        'decision_reelle': 'Accepté' if reel[position] == 1 else 'Refusé',
        'decision_modele': 'Accepté' if prediction == 1 else 'Refusé',
        'probabilite_accord': float(probabilite[1]),
        'valeur_de_base': float(explainer.expected_value[1]),
        'somme_shap': float(contributions.sum()),
        'controle_additivite': float(explainer.expected_value[1] + contributions.sum()),
        'top5': [{
            'variable': FEATURE_ORDER[i],
            'libelle': FEATURE_LABELS[FEATURE_ORDER[i]],
            'valeur_variable': float(features[0][i]),
            'valeur_shap': float(contributions[i]),
            'sens': "vers l'accord" if contributions[i] > 0 else 'vers le refus',
        } for i in ordre_local[:5]],
        'toutes_valeurs_shap': {FEATURE_ORDER[i]: float(contributions[i]) for i in range(len(FEATURE_ORDER))},
    }


# Premier dossier (ordre du fichier) prédit refusé, et premier prédit accepté
pos_refus = int(np.where(pred == 0)[0][0])
pos_accord = int(np.where(pred == 1)[0][0])
local_refus = expliquer(pos_refus, 'shap_local_refus.png')
local_accord = expliquer(pos_accord, 'shap_local_accord.png')

resultat = {
    'explainer': 'shap.KernelExplainer(clf_nb.predict_proba, background_features)',
    'modele': type(clf_nb).__name__,
    'modele_lignes_entrainement': int(clf_nb.class_count_.sum()),
    'background_fichier': 'bd_validation.csv',
    'background_lignes': int(background_features.shape[0]),
    'background_repartition_reelle': {'refus': int((reel == 0).sum()), 'accord': int((reel == 1).sum())},
    'valeur_de_base_accord': float(explainer.expected_value[1]),
    'versions': {'shap': shap.__version__},
    'global_identique_a_static_summary_plot': identique,
    'classement_global': classement,
    'predictions_bd_validation': [{
        'position': int(k), 'identifiant': int(_df_background['Unnamed: 0'].iloc[k]),
        'reel': int(reel[k]), 'predit': int(pred[k]), 'probabilite_accord': float(proba[k]),
    } for k in range(len(reel))],
    'local_refus': local_refus,
    'local_accord': local_accord,
}
json.dump(resultat, open(os.path.join(ICI, 'resultats_xai.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# Affichage
# ------------------------------------------------------------------
print(f"Modèle : {resultat['modele']} ({resultat['modele_lignes_entrainement']} lignes d'entraînement)")
print(f"Background : bd_validation.csv, {resultat['background_lignes']} lignes "
      f"({resultat['background_repartition_reelle']})")
print(f"Valeur de base E[P(accord)] : {resultat['valeur_de_base_accord']:.4f}")
print(f"shap_global.png identique à static/summary_plot.png : {identique}")
print("\nClassement global (moyenne des |SHAP|, classe accord) :")
for c in classement:
    print(f"  {c['rang']:2d}. {c['variable']:32s} {c['moyenne_abs_shap']:.6f}")
for nom, loc in [('REFUS', local_refus), ('ACCORD', local_accord)]:
    print(f"\nDossier {nom} : position {loc['position_dans_bd_validation']}, id {loc['identifiant_Unnamed_0']}, "
          f"réel {loc['decision_reelle']}, modèle {loc['decision_modele']}, P(accord) = {loc['probabilite_accord']:.6f}")
    print(f"  base {loc['valeur_de_base']:.4f} + somme SHAP {loc['somme_shap']:.4f} = {loc['controle_additivite']:.4f}")
    for t in loc['top5']:
        print(f"  {t['variable']:32s} valeur={t['valeur_variable']:<10g} SHAP={t['valeur_shap']:+.4f}  {t['sens']}")
