"""
Ajoute des données de démonstration dans bank.db (pour les captures du rapport).

- 3 comptes analystes (agences Tunis Centre, Sousse, Nabeul), ~6 dossiers clients chacun, 1 à 3 demandes par dossier
  datées entre juillet et septembre 2026.
- Chaque décision est calculée par le vrai modèle Naive Bayes, avec le même
  prétraitement et la même explication SHAP que l'application
  (fonctions evaluer_client / enregistrer_historique_credit de app.py).
- Le script AJOUTE seulement : aucune donnée existante n'est modifiée ni supprimée.
- Une sauvegarde de la base est faite avant toute écriture.

Lancement :  python seed_demo.py
"""
import contextlib
import io
import os
import random
import shutil
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'bank.db')
BACKUP_PATH = DB_PATH + '.backup'

MOT_DE_PASSE_DEMO = 'Demo@2026'   # respecte les règles de password_utils.py
SEED = 2026                        # mêmes données à chaque exécution sur une base neuve

ANALYSTES = [
    {'nom': 'Yassine Ben Ammar', 'email': 'yassine.benammar@banque.com', 'telephone': '98 412 330',
     'agence': 'Agence Tunis Centre', 'ville': 'Tunis'},
    {'nom': 'Amira Gharbi', 'email': 'amira.gharbi@banque.com', 'telephone': '55 207 914',
     'agence': 'Agence Sousse', 'ville': 'Sousse'},
    {'nom': 'Mehdi Chaabane', 'email': 'mehdi.chaabane@banque.com', 'telephone': '22 863 105',
     'agence': 'Agence Nabeul', 'ville': 'Nabeul'},
]

CLIENTS_PAR_ANALYSTE = 6

PRENOMS = ['Mohamed', 'Ahmed', 'Youssef', 'Khalil', 'Hamza', 'Aymen', 'Skander', 'Wassim', 'Oussama', 'Bilel',
           'Fatma', 'Mariem', 'Rania', 'Ines', 'Salma', 'Asma', 'Olfa', 'Sonia', 'Hela', 'Imen']
NOMS = ['Ben Salah', 'Jlassi', 'Hammami', 'Ayari', 'Mejri', 'Bouazizi', 'Dridi', 'Khemiri', 'Ferchichi',
        'Zouari', 'Mansouri', 'Baccouche', 'Sassi', 'Chebbi', 'Haddad', 'Ksouri', 'Toumi', 'Riahi']
RUES = ['Rue de Marseille', 'Avenue Habib Bourguiba', 'Rue Ibn Khaldoun', 'Avenue de la République',
        'Rue Farhat Hached', 'Avenue Taïeb Mhiri', 'Rue Ali Belhouane', 'Rue de Palestine']

# Villes des clients : uniquement les trois villes connues du modèle,
# surtout celle de l'agence de l'analyste.
VILLES_AUTOUR = {
    'Tunis': ['Tunis', 'Tunis', 'Tunis', 'Tunis', 'Nabeul', 'Sousse'],
    'Sousse': ['Sousse', 'Sousse', 'Sousse', 'Sousse', 'Tunis', 'Nabeul'],
    'Nabeul': ['Nabeul', 'Nabeul', 'Nabeul', 'Nabeul', 'Tunis', 'Sousse'],
}

# Profils de risque. Les plages restent dans celles des données d'entraînement
# du modèle (data.csv), sinon le Naive Bayes extrapole et ses décisions n'ont plus de sens.
PROFILS = {
    'solide':  {'revenu': (3000, 4990), 'credits': (0, 2), 'retard': (0, 0), 'taux': (0.04, 0.20)},
    'moyen':   {'revenu': (1800, 3600), 'credits': (1, 3), 'retard': (0, 2), 'taux': (0.15, 0.40)},
    'fragile': {'revenu': (820, 2300),  'credits': (2, 4), 'retard': (2, 4), 'taux': (0.30, 0.85)},
}

MOIS = {7: 'Juillet 2026', 8: 'Août 2026', 9: 'Septembre 2026'}


# ------------------------------------------------------------------
# 1. Lecture de la structure (lecture seule)
# ------------------------------------------------------------------

COLONNES_REQUISES = {
    'banquier': {'id', 'nom', 'email', 'password_hash', 'telephone', 'agence'},
    'client': {'id', 'banquier_id', 'msisdn', 'bankcode', 'nom', 'prenom', 'adresse', 'telephone', 'agence',
               'statut', 'ville', 'type_contrat', 'solde', 'solde_cashback', 'age', 'revenu', 'anciennete',
               'credits', 'retard', 'taille_entreprise', 'taux_endettement', 'dernier_mvt', 'surendettement',
               'derniere_decision', 'derniere_probabilite', 'derniers_facteurs', 'derniere_shap_image'},
    'credit': {'id', 'client_id', 'date_demande', 'decision', 'statut', 'features_snapshot', 'facteurs_shap',
               'shap_image'},
}


def lire_structure():
    if not os.path.exists(DB_PATH):
        sys.exit(f"Base introuvable : {DB_PATH}")

    con = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        print("Structure de la base :")
        for table, requises in COLONNES_REQUISES.items():
            if table not in tables:
                sys.exit(f"  Table « {table} » absente : lancez l'application une fois, puis relancez ce script.")
            colonnes = {r[1] for r in con.execute(f'PRAGMA table_info({table})')}
            manquantes = requises - colonnes
            if manquantes:
                sys.exit(f"  Colonnes manquantes dans « {table} » : {sorted(manquantes)}")
            nb = con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
            print(f"  - {table:9s} {len(colonnes):2d} colonnes, {nb} ligne(s) existante(s)")

        emails = [a['email'] for a in ANALYSTES]
        deja = con.execute(
            f"SELECT email FROM banquier WHERE email IN ({','.join('?' * len(emails))})", emails
        ).fetchall()
        if deja:
            sys.exit(
                "\nLes comptes de démonstration existent déjà : " + ', '.join(e for (e,) in deja) +
                "\nRien n'a été modifié. Pour repartir de zéro, restaurez d'abord la sauvegarde."
            )
    finally:
        con.close()


# ------------------------------------------------------------------
# 2. Sauvegarde
# ------------------------------------------------------------------

def sauvegarder():
    """Copie cohérente de la base via l'API de sauvegarde SQLite.
    Une sauvegarde déjà présente n'est jamais écrasée."""
    destination = BACKUP_PATH
    if os.path.exists(destination):
        horodatage = datetime.now().strftime('%Y%m%d-%H%M%S')
        destination = f'{DB_PATH}.{horodatage}.backup'

    source = sqlite3.connect(DB_PATH)
    cible = sqlite3.connect(destination)
    try:
        source.backup(cible)
    finally:
        cible.close()
        source.close()
    print(f"\nSauvegarde créée : {os.path.basename(destination)}")
    return destination


# ------------------------------------------------------------------
# 3. Génération des données
# ------------------------------------------------------------------

def date_ouvree(rng, mois):
    """Date et heure de bureau aléatoires (lundi-vendredi, 8h30-17h30) dans le mois donné."""
    while True:
        jour = rng.randint(1, 30 if mois == 9 else 31)
        d = datetime(2026, mois, jour, rng.randint(8, 17), rng.choice([0, 10, 15, 20, 30, 40, 45, 50]))
        if d.weekday() < 5 and (d.hour, d.minute) >= (8, 30):
            return d


def generer_profil_client(rng, profil):
    p = PROFILS[profil]
    credits = rng.randint(*p['credits'])
    # Le taux d'endettement suit le nombre de crédits en cours
    taux = rng.uniform(*p['taux'])
    if credits == 0:
        taux = round(rng.uniform(0.0, 0.08), 3)
    return {
        'revenu': float(round(rng.uniform(*p['revenu']), -1)),
        'credits': float(credits),
        'retard': float(rng.randint(*p['retard'])),
        'taux_endettement': round(min(taux, 0.95), 3),
        'age': rng.randint(36, 56),
        'anciennete': float(rng.randint(1, 19)),
        'solde': float(rng.choice([1000, 1500, 2000, 2500, 3000, 3000])),
        'solde_cashback': float(rng.choice([0, 0, 0, 0, 15, 40])),
        'taille_entreprise': float(rng.choice([0, 1, 1, 2])),
        'dernier_mvt': float(rng.randint(1520, 1580)),
        'type_contrat': rng.choice(['CDI', 'CDI', 'CDD', 'Freelance']) if profil == 'solide'
                        else rng.choice(['CDI', 'CDD', 'Freelance']),
        'statut': rng.choice(['Inscrit'] * 6 + ['Enregistré', 'Désinscrit']),
    }


def faire_evoluer(rng, client, decision_precedente):
    """Situation du client à la demande suivante, cohérente avec la précédente."""
    if decision_precedente == 'Accepté':
        # Un nouveau crédit s'ajoute : plus de crédits en cours, endettement plus élevé
        client.credits = min(client.credits + 1, 4)
        client.taux_endettement = round(min(client.taux_endettement + rng.uniform(0.06, 0.15), 0.95), 3)
        if rng.random() < 0.3:
            client.retard = min(client.retard + 1, 4)
    else:
        # Après un refus, le client assainit sa situation avant de redemander
        client.retard = max(client.retard - rng.choice([1, 1, 2]), 0)
        client.taux_endettement = round(max(client.taux_endettement - rng.uniform(0.05, 0.15), 0.02), 3)
        client.revenu = float(round(client.revenu * rng.uniform(1.03, 1.10), -1))
    client.dernier_mvt = float(min(client.dernier_mvt + rng.randint(5, 25), 1595))
    client.surendettement = 1 if client.taux_endettement > appli.SEUIL_SURENDETTEMENT else 0


def main():
    lire_structure()
    sauvegarder()

    # Import de l'application APRÈS la sauvegarde (son import prépare la base)
    print("Chargement du modèle et de l'explainer SHAP…")
    global appli
    import app as appli
    from models import db, Banquier, Client, Credit

    rng = random.Random(SEED)
    noms_pris = set()
    msisdn_pris = set()
    resume = defaultdict(lambda: {'Accepté': 0, 'Refusé': 0})
    nb_clients = nb_demandes = 0

    with appli.app.app_context():
        for i, infos in enumerate(ANALYSTES):
            analyste = Banquier(
                nom=infos['nom'], email=infos['email'], telephone=infos['telephone'],
                agence=infos['agence'], role='analyste',
                date_creation=datetime(2026, 6, rng.randint(15, 28), 9, 0),
            )
            analyste.set_password(MOT_DE_PASSE_DEMO)   # même hachage que l'application
            db.session.add(analyste)
            db.session.commit()
            print(f"\n{analyste.nom} ({analyste.agence})")

            profils = ['solide', 'solide', 'moyen', 'moyen', 'fragile', 'fragile']
            rng.shuffle(profils)
            derniere_activite = None

            for k in range(CLIENTS_PAR_ANALYSTE):
                while True:
                    prenom, nom = rng.choice(PRENOMS), rng.choice(NOMS)
                    if (prenom, nom) not in noms_pris:
                        noms_pris.add((prenom, nom))
                        break
                while True:
                    msisdn = str(rng.choice([2, 5, 9])) + ''.join(str(rng.randint(0, 9)) for _ in range(7))
                    if msisdn not in msisdn_pris:
                        msisdn_pris.add(msisdn)
                        break

                valeurs = generer_profil_client(rng, profils[k])
                ville = rng.choice(VILLES_AUTOUR[infos['ville']])
                client = Client(
                    banquier_id=analyste.id,
                    msisdn=msisdn,
                    bankcode=str(rng.randint(103, 114)),
                    prenom=prenom,
                    nom=nom,
                    adresse=f"{rng.randint(3, 120)} {rng.choice(RUES)}, {ville}",
                    telephone=f"{msisdn[:2]} {msisdn[2:5]} {msisdn[5:]}",
                    agence=infos['agence'],
                    ville=ville,
                    **valeurs,
                )
                client.surendettement = 1 if client.taux_endettement > appli.SEUIL_SURENDETTEMENT else 0
                db.session.add(client)
                db.session.commit()
                nb_clients += 1

                # 1 à 3 demandes, dans des mois différents et croissants
                nb = rng.choice([1, 1, 2, 2, 3])
                mois_demandes = sorted(rng.sample([7, 8, 9], nb))
                decision = None
                for n, mois in enumerate(mois_demandes):
                    if n > 0:
                        faire_evoluer(rng, client, decision)
                        db.session.commit()

                    # Décision + explication SHAP calculées par le modèle, comme dans l'application
                    with contextlib.redirect_stderr(io.StringIO()):   # masque la barre de progression SHAP
                        resultat = appli.evaluer_client(client)
                    appli.enregistrer_historique_credit(client, resultat)
                    decision = resultat['decision']

                    credit = (Credit.query.filter_by(client_id=client.id)
                              .order_by(Credit.id.desc()).first())
                    credit.date_demande = date_ouvree(rng, mois)
                    db.session.commit()

                    resume[mois][decision] += 1
                    nb_demandes += 1
                    derniere_activite = max(filter(None, [derniere_activite, credit.date_demande]))
                    print(f"  {prenom + ' ' + nom:22s} {ville:10s} {profils[k]:8s} "
                          f"{credit.date_demande:%d/%m/%Y} -> {decision} ({resultat['proba_accepte']}% accord)")

            # Dernière connexion = jour de sa dernière évaluation
            analyste.derniere_connexion = derniere_activite + timedelta(minutes=rng.randint(5, 90))
            db.session.commit()

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------
    print("\n" + "=" * 52)
    print("Données ajoutées")
    print("=" * 52)
    print(f"  Analystes : {len(ANALYSTES)}")
    print(f"  Dossiers  : {nb_clients}")
    print(f"  Demandes  : {nb_demandes}")
    print(f"\n  {'Mois':16s}{'Accords':>9s}{'Refus':>8s}{'Total':>8s}")
    for mois, libelle in MOIS.items():
        a, r = resume[mois]['Accepté'], resume[mois]['Refusé']
        print(f"  {libelle:16s}{a:>9d}{r:>8d}{a + r:>8d}")
    total_a = sum(v['Accepté'] for v in resume.values())
    total_r = sum(v['Refusé'] for v in resume.values())
    print(f"  {'Total':16s}{total_a:>9d}{total_r:>8d}{total_a + total_r:>8d}")
    print(f"\nMot de passe des 3 comptes : {MOT_DE_PASSE_DEMO}")
    for a in ANALYSTES:
        print(f"  - {a['email']}")


if __name__ == '__main__':
    main()
