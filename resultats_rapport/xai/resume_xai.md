# Explicabilité du modèle avec SHAP

Tous les chiffres et toutes les figures de ce dossier viennent de `xai.py`, qui reprend ligne pour ligne le code SHAP de `app.py` (lignes 102 à 129 et fonction `evaluer_client`). Contrôle : l'importance globale recalculée (`shap_global.png`) est **identique pixel par pixel** à l'image utilisée par l'application (`static/summary_plot.png`).

Environnement : shap 0.52.0, scikit-learn 1.9.0, matplotlib 3.11.1.

---

## 1. Comment l'application utilise SHAP

### Explainer et modèle

| Élément | Valeur |
|---|---|
| Explainer | `shap.KernelExplainer(clf_nb.predict_proba, background_features)` (`app.py`, ligne 109) |
| Modèle expliqué | Naive Bayes gaussien (`model_naive_bayes.pkl`, entraîné sur 1 782 lignes) |
| Sortie expliquée | `predict_proba`, colonne 1 : **probabilité d'accord** (entre 0 et 1) |
| Nombre de variables | 20, dans l'ordre `FEATURE_ORDER` |

`KernelExplainer` est la méthode SHAP « agnostique » : elle ne regarde pas l'intérieur du modèle, elle l'interroge sur des combinaisons de variables. Une valeur SHAP positive pousse la probabilité d'accord vers le haut (vers l'accord) ; une valeur négative la pousse vers le bas (vers le refus).

### Données de référence (background)

- Fichier : **`bd_validation.csv`**, avec les 20 colonnes de `FEATURE_ORDER` (`app.py`, lignes 105 à 107).
- **20 lignes** : 12 refus et 8 accords. Ce sont les 20 dossiers mis de côté dans `projet.ipynb` (cellule 44).
- **Valeur de base** `E[f(X)]` = **0,3871**. C'est la probabilité d'accord moyenne du modèle sur ces 20 dossiers, le point de départ de chaque explication.

### Explication locale d'un dossier (fonction `evaluer_client`)

1. Les données du client sont transformées en un vecteur de 20 variables (`client_to_features`).
2. `np.random.seed(42)`, puis `explainer.shap_values(features)` : on obtient une valeur SHAP par variable. Seules les valeurs de la classe Accord (`shap_values[0, :, 1]`) sont gardées.
3. **Graphique** : un **waterfall plot** (`shap.plots.waterfall`, `max_display=8`). Il part de la valeur de base (0,387), ajoute la contribution de chaque variable, et arrive à la probabilité d'accord du dossier `f(x)`. Les 7 variables les plus influentes sont détaillées, les 13 autres sont regroupées. En rouge, les contributions vers l'accord ; en bleu, vers le refus. L'image est enregistrée en PNG (120 dpi) dans la base de données (`Credit.shap_image`, `Client.derniere_shap_image`).
4. **« Détail des facteurs principaux »** : les 6 variables avec la plus grande valeur SHAP absolue, sous forme de barres horizontales.
   - Le libellé est en français (`FEATURE_LABELS`).
   - Une étiquette indique « Favorable » si la valeur est positive, « Défavorable » si elle est négative.
   - L'impact affiché est `|SHAP| × 100`, c'est-à-dire des points de probabilité.
   - La longueur de chaque barre est relative à la plus grande des 6.
5. Affichage : page de décision (`credit.html`) et formulaire client après évaluation (`client_form.html`).

### Explication globale

- Calculée **une seule fois**, au démarrage de l'application, si `static/summary_plot.png` n'existe pas encore (`app.py`, lignes 116 à 129). L'image est ensuite réutilisée telle quelle.
- **Données** : les **20 lignes du background** (`bd_validation.csv`), qui jouent donc à la fois le rôle de référence et de dossiers expliqués. `np.random.seed(42)`, puis `explainer.shap_values(background_features)`, classe Accord.
- **Graphique** : `shap.summary_plot`, en mode « beeswarm » (nuage de points). Chaque point est un dossier. Sa position horizontale est la valeur SHAP, sa couleur la valeur de la variable (bleu = faible, rouge = élevée).
- **Mesure d'importance** : les variables sont **triées par moyenne des valeurs SHAP absolues** sur les 20 dossiers. C'est le tri par défaut de `summary_plot` ; l'application n'affiche pas ces valeurs, seulement l'ordre.
- Affichage : page Statistiques (`statistics.html`) et page de décision (`credit.html`).

---

## 2. Importance globale des 20 variables

Moyenne des |SHAP| (classe Accord) sur les 20 dossiers de `bd_validation.csv`, en points de probabilité. Figure : `shap_global.png`.

| Rang | Variable | Libellé | Moyenne des \|SHAP\| |
|---|---|---|---|
| 1 | `retard` | Retards de paiement | 0,3158 |
| 2 | `taux_endettement` | Taux d'endettement | 0,1242 |
| 3 | `revenu` | Revenu | 0,0171 |
| 4 | `credits` | Crédits en cours | 0,0135 |
| 5 | `age` | Âge | 0,0019 |
| 6 | `taille_entreprise` | Taille de l'entreprise | 0,0016 |
| 7 | `m-pay_sub_status_1_Désinscrit` | Statut : Désinscrit | 0,0016 |
| 8 | `last_mvt` | Dernier mouvement (jours) | 0,0014 |
| 9 | `m-pay_sub_city_Nabeul` | Ville : Nabeul | 0,0014 |
| 10 | `anciennete` | Ancienneté | 0,0013 |
| 11 | `type_contrat_CDI` | Contrat CDI | 0,0013 |
| 12 | `m-pay_sub_solde` | Solde du compte | 0,0009 |
| 13 | `m-pay_sub_status_1_Enregistré` | Statut : Enregistré | 0,0008 |
| 14 | `m-pay_sub_city_Sousse` | Ville : Sousse | 0,0005 |
| 15 | `type_contrat_Freelance` | Contrat Freelance | 0,0004 |
| 16 | `type_contrat_CDD` | Contrat CDD | 0,0003 |
| 17 | `m-pay_sub_city_Tunis` | Ville : Tunis | 0,0003 |
| 18 | `m-pay_sub_status_1_Inscrit` | Statut : Inscrit | 0,0003 |
| 19 | `surendettement` | Surendettement | 0,0000 |
| 20 | `SoldeCashBack` | Solde cashback | 0,0000 |

Les valeurs exactes, avec 6 décimales et l'ordre utilisé pour départager les ex æquo, sont dans `resultats_xai.json`.

**Lecture** : deux variables expliquent presque toute la décision. Les **retards de paiement** pèsent en moyenne 31,6 points de probabilité, et le **taux d'endettement** 12,4 points. Viennent ensuite le revenu et le nombre de crédits en cours (1 à 2 points). Les 16 autres variables ont un effet inférieur à 0,2 point. Sur le graphique, les retards élevés (points rouges) sont tous à gauche, vers le refus, et les retards faibles (points bleus) à droite, vers l'accord. C'est la même chose pour le taux d'endettement.

---

## 3. Deux dossiers expliqués

Le modèle classe correctement les 20 dossiers de `bd_validation.csv` (12 refus et 8 accords). Les deux dossiers retenus sont **le premier dossier refusé et le premier dossier accepté**, dans l'ordre du fichier. Leurs explications sont calculées exactement comme dans l'application (même graine, même graphique).

Dans les deux cas, la valeur de base plus la somme des 20 valeurs SHAP redonne exactement la probabilité d'accord (propriété d'additivité de SHAP).

### Dossier refusé : `bd_validation.csv`, ligne 1 (identifiant 2399)

- Décision réelle : Refus. **Décision du modèle : Refus.**
- **Probabilité d'accord : 4,24 %** (0,0424).
- Calcul : 0,3871 (base) − 0,3447 (somme des SHAP) = 0,0424.

| Rang | Variable | Valeur du dossier | Valeur SHAP | Sens |
|---|---|---|---|---|
| 1 | Retards de paiement (`retard`) | 2 | −0,3445 | vers le refus |
| 2 | Taux d'endettement (`taux_endettement`) | 0,168 | +0,0267 | vers l'accord |
| 3 | Revenu (`revenu`) | 1 789 | −0,0263 | vers le refus |
| 4 | Crédits en cours (`credits`) | 1 | +0,0086 | vers l'accord |
| 5 | Âge (`age`) | 46 | −0,0023 | vers le refus |

**Lecture** : ce client a un endettement faible et peu de crédits, ce qui joue en sa faveur. Mais ses **2 retards de paiement** font à eux seuls chuter la probabilité d'accord de 34 points, et son revenu assez bas l'abaisse encore un peu. Figure : `shap_local_refus.png`.

### Dossier accepté : `bd_validation.csv`, ligne 6 (identifiant 1912)

- Décision réelle : Accord. **Décision du modèle : Accord.**
- **Probabilité d'accord : 93,68 %** (0,9368).
- Calcul : 0,3871 (base) + 0,5497 (somme des SHAP) = 0,9368.

| Rang | Variable | Valeur du dossier | Valeur SHAP | Sens |
|---|---|---|---|---|
| 1 | Retards de paiement (`retard`) | 1 | +0,3562 | vers l'accord |
| 2 | Taux d'endettement (`taux_endettement`) | 0,074 | +0,1507 | vers l'accord |
| 3 | Revenu (`revenu`) | 4 043 | +0,0272 | vers l'accord |
| 4 | Crédits en cours (`credits`) | 1 | +0,0208 | vers l'accord |
| 5 | Taille de l'entreprise (`taille_entreprise`) | 0 (Petite) | +0,0048 | vers l'accord |

**Lecture** : les 5 variables les plus influentes poussent toutes vers l'accord. Un seul retard et un endettement très faible (7,4 %) font monter la probabilité de 51 points, complétés par un revenu élevé. Figure : `shap_local_accord.png`.

---

## 4. Points à mentionner dans le rapport

1. **Cohérence avec les données.** Dans `data.csv`, la décision suit exactement la règle « accord si `retard ≤ 1` et `taux_endettement < 0,40` ». SHAP retrouve bien ces deux variables en tête, très loin devant les autres. C'est un bon argument pour la fiabilité de l'explication.
2. **`surendettement` et `SoldeCashBack` à zéro.** Leur importance nulle ne veut pas dire que le modèle les ignore. Ces deux variables valent toujours 0 dans les 20 dossiers de référence, alors que `data.csv` contient 66 dossiers surendettés et 89 dossiers avec un solde cashback non nul. KernelExplainer mesure l'effet d'une variable par rapport au background : une variable constante dans le background et chez le dossier expliqué a forcément une valeur SHAP nulle. Pour un client réel surendetté, l'explication locale montrerait bien un effet.
3. **Background réduit.** La référence ne compte que 20 dossiers. La valeur de base et l'importance globale dépendent donc de cet échantillon : un background plus large (par exemple 100 à 200 dossiers tirés de l'entraînement) donnerait une vue globale plus représentative.
4. **Valeurs approchées.** KernelExplainer estime les valeurs SHAP par échantillonnage aléatoire. L'application fixe la graine (`np.random.seed(42)`), ce qui rend les résultats reproductibles.
5. **Libellés des graphiques.** Comme dans l'application, le waterfall et le summary plot affichent les noms techniques des variables (`retard`, `m-pay_sub_solde`…) et les titres d'axes en anglais (bibliothèque SHAP). Seule la liste « Détail des facteurs principaux » utilise les libellés français.

---

## Fichiers

| Fichier | Contenu |
|---|---|
| `shap_global.png` | Importance globale (summary plot), identique à `static/summary_plot.png` |
| `shap_local_refus.png` | Waterfall du dossier refusé (identifiant 2399) |
| `shap_local_accord.png` | Waterfall du dossier accepté (identifiant 1912) |
| `resultats_xai.json` | Toutes les valeurs : classement global, prédictions des 20 dossiers, 20 valeurs SHAP de chaque dossier |
| `xai.py` | Script qui produit tout ce qui précède (`python resultats_rapport/xai/xai.py`) |
