# Résultats du modèle de décision de crédit

**Méthode** : deuxième série d'essais de `projet.ipynb` (cellules 44 à 58). 20 dossiers sont mis de côté, puis les 2 950 restants sont divisés en 75 % d'entraînement et 25 % de test.

Les chiffres viennent de `serie_b.py`, qui ré-exécute ces cellules dans l'ordre avec la préparation des cellules 0 à 31. Pour les 8 modèles, l'accuracy, la matrice de confusion et le rapport de classification sont **identiques aux sorties enregistrées dans les cellules 48 à 58 du notebook**. Les précisions, rappels et F1-scores sont donnés avec 4 décimales, alors que le notebook les arrondit à 2. Les scores sur les 20 dossiers mis de côté ne figurent pas dans ces cellules : ils ont été calculés par `serie_b.py` avec les mêmes modèles.

---

## 1. Données

| Élément | Valeur |
|---|---|
| Fichier | `data.csv` |
| Nombre de lignes | 2 970 |
| Nombre de colonnes | 18 |
| Valeurs manquantes | 0 (`bd.isna().sum()`) |
| Lignes en double | 0 (`bd.duplicated().sum()`) |

### Variables du fichier brut (18)

| # | Variable | Type | Utilisation |
|---|---|---|---|
| 1 | `Unnamed: 0` | entier | index, exclu du modèle |
| 2 | `m-pay_sub_msisdn` | entier | identifiant (numéro), exclu du modèle |
| 3 | `m-pay_sub_client_bankcode` | entier | code banque, exclu du modèle |
| 4 | `m-pay_sub_status_1` | texte (Désinscrit, Enregistré, Inscrit) | encodé en one-hot |
| 5 | `m-pay_sub_status_1_last_modif_date` | date | transformé en `last_mvt`, puis supprimé |
| 6 | `m-pay_sub_birth_date` | date | transformé en `age`, puis supprimé |
| 7 | `m-pay_sub_city` | texte (Nabeul, Sousse, Tunis) | encodé en one-hot |
| 8 | `m-pay_sub_solde` | entier | conservé |
| 9 | `SoldeCashBack` | réel | conservé |
| 10 | `age` | entier | supprimé puis recalculé depuis la date de naissance |
| 11 | `revenu` | entier | conservé |
| 12 | `type_contrat` | texte (CDD, CDI, Freelance) | encodé en one-hot |
| 13 | `anciennete` | entier | conservé |
| 14 | `credits` | entier | conservé |
| 15 | `retard` | entier | conservé |
| 16 | `taille_entreprise` | texte (Petite, Moyenne, Grande) | encodé en ordinal |
| 17 | `taux_endettement` | réel | conservé ; sert aussi à créer `surendettement` |
| 18 | `decision` | entier (0 / 1) | **variable cible** |

### Variable cible : `decision`

| Classe | Signification | Nombre | Pourcentage |
|---|---|---|---|
| 0 | Refus | 2 013 | 67,8 % |
| 1 | Accord | 957 | 32,2 % |
| **Total** | | **2 970** | 100 % |

Les classes sont déséquilibrées (environ 2 refus pour 1 accord). Aucun rééquilibrage n'a été fait. Figure : `repartition_cible.png`.

---

## 2. Préparation des données (cellules 6 à 31)

1. **Contrôle qualité** : aucune valeur manquante et aucun doublon, donc aucune imputation ni suppression de lignes n'a été nécessaire.
2. **Âge** : la colonne `age` d'origine est supprimée, puis recalculée à partir de `m-pay_sub_birth_date` : `(date de référence − date de naissance) en jours // 365`.
3. **Ancienneté du dernier mouvement** : nouvelle variable `last_mvt` = nombre de jours entre la date de référence et `m-pay_sub_status_1_last_modif_date`.
4. **Suppression des colonnes de dates** `m-pay_sub_birth_date` et `m-pay_sub_status_1_last_modif_date`.
5. **Création de `surendettement`** : 1 si `taux_endettement > 1`, sinon 0. **66 dossiers** valent 1.
6. **Encodage ordinal** de `taille_entreprise` : Petite = 0, Moyenne = 1, Grande = 2.
7. **Encodage one-hot** (`pd.get_dummies`) de `m-pay_sub_city`, `type_contrat` et `m-pay_sub_status_1`, puis conversion des booléens en 0/1. La table passe à **24 colonnes**.
8. **Choix des variables explicatives** : toutes sauf `decision`, `Unnamed: 0`, `m-pay_sub_msisdn` et `m-pay_sub_client_bankcode`, soit **20 variables** :
   `m-pay_sub_solde`, `SoldeCashBack`, `revenu`, `anciennete`, `credits`, `retard`, `taille_entreprise`, `taux_endettement`, `age`, `last_mvt`, `surendettement`, `m-pay_sub_city_Nabeul`, `m-pay_sub_city_Sousse`, `m-pay_sub_city_Tunis`, `type_contrat_CDD`, `type_contrat_CDI`, `type_contrat_Freelance`, `m-pay_sub_status_1_Désinscrit`, `m-pay_sub_status_1_Enregistré`, `m-pay_sub_status_1_Inscrit`.
9. **Normalisation** (`StandardScaler`, ajusté sur l'entraînement seulement) pour SVM, KNN, Perceptron, régression logistique et SGD. Random Forest, arbre de décision et Naive Bayes utilisent les données non normalisées.

> **Date de référence** pour l'âge et `last_mvt` : **6 août 2026, 9 h 13**. Le notebook utilise `pd.Timestamp.today()` ; cette date est celle de l'exécution d'origine de la série, c'est-à-dire l'heure de création de `bd_validation.csv`. Elle redonne exactement l'âge et le `last_mvt` enregistrés dans `bd_validation.csv`.

---

## 3. Division des données (cellules 44 à 47)

1. **20 dossiers mis de côté** : `bd_model.sample(n=20, random_state=42)`, enregistrés dans `bd_validation.csv` (cellule 49). Ce sont bien les mêmes 20 dossiers (vérifié).
2. **Division des 2 950 restants** : `train_test_split(test_size=0.25, random_state=42)`, sans stratification.

| Ensemble | Part | Lignes | Refus (0) | Accord (1) |
|---|---|---|---|---|
| Entraînement | 75 % des 2 950 | 2 212 | 1 501 | 711 |
| Test | 25 % des 2 950 | 738 | 500 | 238 |
| Dossiers mis de côté | — | 20 | 12 | 8 |

---

## 4. Modèles testés et résultats (cellules 48 à 58)

8 modèles, entraînés sur les 2 212 lignes d'entraînement :

| Modèle | Paramètres | Données | Cellule |
|---|---|---|---|
| Random Forest | `n_estimators=100, random_state=42` | brutes | 48 |
| SVM | `kernel='rbf'` | normalisées | 51 |
| KNN | `n_neighbors=5` | normalisées | 52 |
| Arbre de décision | `random_state=42` | brutes | 54 |
| Naive Bayes gaussien | paramètres par défaut | brutes | 55 |
| Perceptron | `random_state=42` | normalisées | 56 |
| Régression logistique | paramètres par défaut | normalisées | 57 |
| SGD Classifier | `random_state=42` (perte par défaut : hinge) | normalisées | 58 |

La cellule 53 refait une Random Forest sans `random_state`, avec le même résultat (accuracy 1,0000) : c'est un doublon, il n'est pas compté.

Précision, rappel et F1 sont calculés pour la **classe positive Accord (1)**.

### Ensemble de test (738 lignes) et dossiers mis de côté (20 lignes)

| Modèle | Accuracy (test) | Précision | Rappel | F1-score | 20 dossiers |
|---|---|---|---|---|---|
| Random Forest | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 20 / 20 |
| Arbre de décision | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 20 / 20 |
| SVM | 0,9770 | 0,9585 | 0,9706 | 0,9645 | 20 / 20 |
| Régression logistique | 0,9743 | 0,9582 | 0,9622 | 0,9602 | 20 / 20 |
| **Naive Bayes** | **0,9661** | **0,9651** | **0,9286** | **0,9465** | **20 / 20** |
| Perceptron | 0,9566 | 0,9440 | 0,9202 | 0,9319 | 20 / 20 |
| SGD Classifier | 0,9566 | 0,9187 | 0,9496 | 0,9339 | 20 / 20 |
| KNN | 0,9404 | 0,9042 | 0,9118 | 0,9079 | 19 / 20 |

---

## 5. Naive Bayes sur l'ensemble de test

| Classe | Précision | Rappel | F1-score | Effectif |
|---|---|---|---|---|
| Refus (0) | 0,9666 | 0,9840 | 0,9752 | 500 |
| Accord (1) | 0,9651 | 0,9286 | 0,9465 | 238 |
| Moyenne macro | 0,9658 | 0,9563 | 0,9608 | 738 |

**Accuracy : 0,9661 (713 bonnes prédictions sur 738)**

### Matrice de confusion (test)

|  | Prédit Refus (0) | Prédit Accord (1) |
|---|---|---|
| **Réel Refus (0)** | 492 (vrais négatifs) | 8 (faux positifs) |
| **Réel Accord (1)** | 17 (faux négatifs) | 221 (vrais positifs) |

- 8 refus réels ont été prédits « accord » (1,1 % du test). C'est l'erreur la plus coûteuse pour la banque : un crédit accordé à tort.
- 17 accords réels ont été prédits « refus » (2,3 % du test).

Figure : `matrice_confusion.png`.

### Dossiers mis de côté (20 lignes)

Les 20 dossiers sont bien classés (12 refus, 8 accords). Le résultat est bon, mais il porte sur un petit échantillon.

---

## 6. Points d'attention pour le rapport

1. **La décision suit une règle exacte.** Sur les 2 970 lignes, `decision = 1` si et seulement si `retard ≤ 1` et `taux_endettement < 0,40`. Un arbre de décision de profondeur 2 retrouve exactement cette règle, d'où ses 100 %, comme ceux de la Random Forest. Ces scores ne prouvent donc pas que ces modèles sont meilleurs en général. Ils montrent que les données suivent une règle déterministe, probablement parce qu'elles ont été générées. Il vaut mieux le dire dans le rapport, et justifier le choix du Naive Bayes par d'autres critères : probabilités et explicabilité avec SHAP.
2. **`last_mvt` dépend de la date de référence.** À l'entraînement, ses valeurs vont de 1 519 à 1 581 jours environ. Une valeur saisie très différente dans l'application (par exemple 5 ou 30) sort de la plage connue du modèle et peut changer la décision.
3. **`model_logistic_regression.pkl` contient un SVM** (cellule 62 du notebook).

---

## Annexe : première série (division 60 / 20 / 20)

Le notebook contient aussi une première série d'essais (cellules 32 à 40) : 1 782 lignes d'entraînement, 594 de validation, 594 de test, `random_state=42`. Le fichier `model_naive_bayes.pkl` contient le Naive Bayes de cette série (sauvegardé par la cellule 61, entraîné sur 1 782 lignes). Les résultats sont reproduits à l'identique par `reproduire_resultats.py` (détails dans `resultats_serie_a.json`, matrice dans `matrice_confusion_serie_a.png`).

| Modèle | Accuracy (validation) | Accuracy (test) | Précision (test) | Rappel (test) | F1 (test) |
|---|---|---|---|---|---|
| Random Forest | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| Arbre de décision | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| Naive Bayes | 0,9798 | 0,9781 | 0,9604 | 0,9749 | 0,9676 |
| SVM | 0,9781 | 0,9731 | 0,9741 | 0,9447 | 0,9592 |
| Régression logistique | 0,9697 | 0,9714 | 0,9643 | 0,9497 | 0,9570 |
| SGD Classifier (log_loss) | 0,9562 | 0,9630 | 0,9538 | 0,9347 | 0,9442 |
| Perceptron | 0,9377 | 0,9495 | 0,9005 | 0,9548 | 0,9268 |
| KNN | 0,9327 | 0,9242 | 0,8889 | 0,8844 | 0,8866 |

Matrice de confusion du Naive Bayes de cette série sur le test : `[[387, 8], [5, 194]]`.

**Résultat impossible à retrouver** : les AUC et courbes ROC. La cellule 42 s'arrête sur une erreur (`NameError: name 'clf_dt' is not defined`), et la figure de la cellule 59 mélange des modèles et des données des deux séries.
