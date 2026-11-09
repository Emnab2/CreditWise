import os
import pickle
import io
import base64
import json
from datetime import datetime, timedelta
from functools import wraps
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend sans interface graphique, obligatoire pour un serveur
import matplotlib.pyplot as plt
import shap
import click
from flask import Flask, render_template, redirect, url_for, request, flash, abort
from sqlalchemy import inspect, text
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, Banquier, Client, Credit
from password_utils import valider_mot_de_passe

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def charger_env(chemin):
    """Lit un fichier .env (CLE=valeur) sans écraser les variables déjà définies."""
    if not os.path.exists(chemin):
        return
    with open(chemin, encoding='utf-8') as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne and not ligne.startswith('#') and '=' in ligne:
                cle, valeur = ligne.split('=', 1)
                os.environ.setdefault(cle.strip(), valeur.strip().strip('"').strip("'"))


# Secrets (clé de session, identifiants SMTP) : dans .env, jamais dans le code.
charger_env(os.path.join(BASE_DIR, '.env'))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-moi-en-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'bank.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'sandbox.smtp.mailtrap.io')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 2525))
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@creditwise.com')
from flask_mail import Mail, Message

mail = Mail(app)
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
db.init_app(app)

# --- Authentification ---
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.login_message = None
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return Banquier.query.get(int(user_id))


def admin_required(view):
    """Réserve une page aux comptes administrateur."""
    @wraps(view)
    @login_required
    def wrapper(*args, **kwargs):
        if not current_user.is_admin:
            abort(403)
        return view(*args, **kwargs)
    return wrapper


# Pages accessibles à un administrateur en dehors de l'espace /admin
PAGES_COMMUNES = {'login', 'logout', 'static', 'settings'}


@app.before_request
def separer_espaces_admin_analyste():
    """Un administrateur ne manipule pas de dossiers clients : il reste
    dans son espace (supervision + gestion des comptes)."""
    if (
        current_user.is_authenticated
        and current_user.is_admin
        and request.endpoint
        and request.endpoint not in PAGES_COMMUNES
        and not request.endpoint.startswith('admin_')
    ):
        return redirect(url_for('admin_dashboard'))


def page_accueil(utilisateur):
    return url_for('admin_dashboard') if utilisateur.is_admin else url_for('dashboard')


# --- Chargement du modèle Naive Bayes (une seule fois, au démarrage) ---
MODEL_PATH = os.path.join(BASE_DIR, 'model_naive_bayes.pkl')
clf_nb = pickle.load(open(MODEL_PATH, 'rb'))

# Seuil au-delà duquel on considère un client en surendettement
SEUIL_SURENDETTEMENT = 0.5

# Ordre EXACT des colonnes attendu par le modèle
FEATURE_ORDER = [
    'm-pay_sub_solde', 'SoldeCashBack', 'revenu', 'anciennete', 'credits', 'retard',
    'taille_entreprise', 'taux_endettement', 'age', 'last_mvt', 'surendettement',
    'm-pay_sub_city_Nabeul', 'm-pay_sub_city_Sousse', 'm-pay_sub_city_Tunis',
    'type_contrat_CDD', 'type_contrat_CDI', 'type_contrat_Freelance',
    'm-pay_sub_status_1_Désinscrit', 'm-pay_sub_status_1_Enregistré', 'm-pay_sub_status_1_Inscrit'
]

# ============================================================
# SHAP : chargement du background + création de l'explainer
# ============================================================
BACKGROUND_PATH = os.path.join(BASE_DIR, 'bd_validation.csv')
_df_background = pd.read_csv(BACKGROUND_PATH)
background_features = _df_background[FEATURE_ORDER].values

explainer = shap.KernelExplainer(clf_nb.predict_proba, background_features)

# --- Summary plot global (généré une seule fois au démarrage, mis en cache sur disque) ---
STATIC_DIR = os.path.join(BASE_DIR, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)
SUMMARY_PLOT_PATH = os.path.join(STATIC_DIR, 'summary_plot.png')

if not os.path.exists(SUMMARY_PLOT_PATH):
    np.random.seed(42)
    _shap_values_bg = explainer.shap_values(background_features)

    plt.figure()
    shap.summary_plot(
        _shap_values_bg[:, :, 1],
        background_features,
        feature_names=FEATURE_ORDER,
        show=False
    )
    plt.tight_layout()
    plt.savefig(SUMMARY_PLOT_PATH, dpi=120, bbox_inches='tight')
    plt.close()

# Libellés lisibles en français pour l'affichage
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

TAILLE_ENTREPRISE_MAP = {'Petite': 0, 'Moyenne': 1, 'Grande': 2}
TAILLE_ENTREPRISE_MAP_INVERSE = {v: k for k, v in TAILLE_ENTREPRISE_MAP.items()}


def client_to_features(client):
    """Transforme un objet Client en tableau numpy dans le format attendu par le modèle."""
    row = {
        'm-pay_sub_solde': client.solde,
        'SoldeCashBack': client.solde_cashback,
        'revenu': client.revenu,
        'anciennete': client.anciennete,
        'credits': client.credits,
        'retard': client.retard,
        'taille_entreprise': client.taille_entreprise,
        'taux_endettement': client.taux_endettement,
        'age': client.age,
        'last_mvt': client.dernier_mvt,
        'surendettement': client.surendettement,
        'm-pay_sub_city_Nabeul': 1 if client.ville == 'Nabeul' else 0,
        'm-pay_sub_city_Sousse': 1 if client.ville == 'Sousse' else 0,
        'm-pay_sub_city_Tunis': 1 if client.ville == 'Tunis' else 0,
        'type_contrat_CDD': 1 if client.type_contrat == 'CDD' else 0,
        'type_contrat_CDI': 1 if client.type_contrat == 'CDI' else 0,
        'type_contrat_Freelance': 1 if client.type_contrat == 'Freelance' else 0,
        'm-pay_sub_status_1_Désinscrit': 1 if client.statut == 'Désinscrit' else 0,
        'm-pay_sub_status_1_Enregistré': 1 if client.statut == 'Enregistré' else 0,
        'm-pay_sub_status_1_Inscrit': 1 if client.statut == 'Inscrit' else 0,
    }
    return np.array([[row[f] for f in FEATURE_ORDER]])


def evaluer_client(client):
    """Calcule la prédiction du modèle + l'explication SHAP pour un client donné,
    et sauvegarde le résultat en base pour un affichage instantané plus tard."""
    features = client_to_features(client)

    prediction = clf_nb.predict(features)[0]
    probabilite = clf_nb.predict_proba(features)[0]

    client.derniere_decision = 'Accepté' if prediction == 1 else 'Refusé'
    client.derniere_probabilite = float(probabilite[1])

    np.random.seed(42)
    shap_values = explainer.shap_values(features)
    contributions = shap_values[0, :, 1]

    explication = shap.Explanation(
        values=contributions,
        base_values=explainer.expected_value[1],
        data=features[0],
        feature_names=FEATURE_ORDER
    )
    plt.figure()
    shap.plots.waterfall(explication, max_display=8, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    force_plot_b64 = base64.b64encode(buf.read()).decode('utf-8')

    facteurs = []
    for nom_feature, valeur_shap in zip(FEATURE_ORDER, contributions):
        facteurs.append({
            'label': FEATURE_LABELS.get(nom_feature, nom_feature),
            'valeur_shap': round(float(valeur_shap), 4),
            'impact_pct': round(abs(float(valeur_shap)) * 100, 1),
            'sens': 'positif' if valeur_shap > 0 else 'negatif',
        })
    facteurs_tries = sorted(facteurs, key=lambda f: abs(f['valeur_shap']), reverse=True)[:6]
    max_impact = max(f['impact_pct'] for f in facteurs_tries) if facteurs_tries else 1
    for f in facteurs_tries:
        f['largeur_barre'] = round((f['impact_pct'] / max_impact) * 100, 1) if max_impact else 0

    # --- Sauvegarde en base pour un affichage instantané au prochain chargement ---
    client.derniers_facteurs = json.dumps(facteurs_tries)
    client.derniere_shap_image = force_plot_b64
    db.session.commit()

    return {
        'decision': client.derniere_decision,
        'proba_accepte': round(probabilite[1] * 100, 1),
        'proba_refuse': round(probabilite[0] * 100, 1),
        'facteurs': facteurs_tries,
        'force_plot_b64': force_plot_b64,
    }


def resultat_depuis_client(client):
    """Reconstruit le dict 'resultat' à partir des données déjà sauvegardées en base,
    sans refaire de calcul SHAP. Retourne None si le client n'a jamais été évalué."""
    if not client.derniere_decision or not client.derniers_facteurs:
        return None

    return {
        'decision': client.derniere_decision,
        'proba_accepte': round(client.derniere_probabilite * 100, 1),
        'proba_refuse': round((1 - client.derniere_probabilite) * 100, 1),
        'facteurs': json.loads(client.derniers_facteurs),
        'force_plot_b64': client.derniere_shap_image,
    }


def remplir_client_depuis_formulaire(client, form):
    """Copie les champs du formulaire dans l'objet Client, et calcule
    automatiquement le surendettement à partir du taux d'endettement."""
    client.msisdn = form['msisdn']
    client.bankcode = form['bankcode']
    client.nom = form['nom']
    client.prenom = form['prenom']

    # Informations personnelles (affichage uniquement, hors modèle)
    client.adresse = form.get('adresse')
    client.telephone = form.get('telephone')
    client.agence = form.get('agence')

    client.statut = form['statut']
    # Format uniforme ("TUNIS", " tunis " -> "Tunis") : le modèle compare la ville exacte.
    client.ville = form['ville'].strip().title()
    client.type_contrat = form['type_contrat']

    client.solde = float(form['solde'])
    client.solde_cashback = float(form['solde_cashback'])
    client.age = int(form['age'])
    client.revenu = float(form['revenu'])
    client.anciennete = float(form['anciennete'])
    client.credits = float(form['credits'])
    client.retard = float(form['retard'])
    client.taille_entreprise = TAILLE_ENTREPRISE_MAP[form['taille_entreprise']]
    client.taux_endettement = float(form['taux_endettement'])
    client.dernier_mvt = float(form['dernier_mvt'])

    client.surendettement = 1 if client.taux_endettement > SEUIL_SURENDETTEMENT else 0


def construire_snapshot_features(client):
    """Copie lisible des données du client au moment de la décision,
    indépendante de ce que deviendra le client plus tard."""
    return {
        'Ville': client.ville,
        'Statut du compte': client.statut,
        'Type de contrat': client.type_contrat,
        'Taille de l\'entreprise': TAILLE_ENTREPRISE_MAP_INVERSE.get(client.taille_entreprise, '—'),
        'Âge': client.age,
        'Revenu': client.revenu,
        'Solde': client.solde,
        'Solde cashback': client.solde_cashback,
        'Ancienneté (années)': client.anciennete,
        'Crédits en cours (nb)': client.credits,
        'Retards de paiement (nb)': client.retard,
        'Taux d\'endettement': client.taux_endettement,
        'Dernier mouvement (jours)': client.dernier_mvt,
        'Surendettement': 'Oui' if client.surendettement else 'Non',
    }


def enregistrer_historique_credit(client, resultat):
    """Crée une nouvelle entrée d'historique à partir du résultat d'une évaluation,
    avec une copie figée des données utilisées pour cette décision précise."""
    nouveau_credit = Credit(
        client_id=client.id,
        decision=resultat['decision'],
        statut='en cours' if resultat['decision'] == 'Accepté' else 'refusé',
        features_snapshot=json.dumps(construire_snapshot_features(client)),
        facteurs_shap=json.dumps(resultat['facteurs']),
        shap_image=resultat['force_plot_b64'],
    )
    db.session.add(nouveau_credit)
    db.session.commit()


# ============================================================
# Authentification
# ============================================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        nom = request.form['nom']
        email = request.form['email']
        password = request.form['password']

        if Banquier.query.filter_by(email=email).first():
            flash("Un compte existe déjà avec cet email.", "danger")
            return redirect(url_for('register'))

        banquier = Banquier(nom=nom, email=email)
        banquier.set_password(password)
        db.session.add(banquier)
        db.session.commit()

        flash("Compte créé avec succès, vous pouvez vous connecter.", "success")
        return redirect(url_for('login'))

    return render_template('login.html', active_tab='register')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        banquier = Banquier.query.filter_by(email=email).first()

        if banquier and banquier.check_password(password):
            login_user(banquier, remember=bool(request.form.get('remember')))
            banquier.derniere_connexion = datetime.utcnow()
            db.session.commit()
            return redirect(page_accueil(banquier))

        flash("Email ou mot de passe incorrect.", "danger")

    return render_template('login.html', active_tab='login')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        banquier = Banquier.query.filter_by(email=email).first()

        if banquier:
            token = serializer.dumps(email, salt='reset-password')
            lien = url_for('reset_password', token=token, _external=True)

            msg = Message(
                subject="Réinitialisation de votre mot de passe",
                recipients=[email],
                body=(
                    f"Bonjour {banquier.nom},\n\n"
                    f"Cliquez sur ce lien pour réinitialiser votre mot de passe "
                    f"(valable 1 heure) :\n{lien}\n\n"
                    f"Si vous n'êtes pas à l'origine de cette demande, ignorez cet email."
                )
            )
            mail.send(msg)

        flash("Si cet email existe, un lien de réinitialisation vous a été envoyé.", "info")
        return redirect(url_for('login'))

    return render_template('forgot_password.html')


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = serializer.loads(token, salt='reset-password', max_age=3600)
    except SignatureExpired:
        flash("Le lien de réinitialisation a expiré. Merci de refaire une demande.", "danger")
        return redirect(url_for('forgot_password'))
    except BadSignature:
        flash("Lien de réinitialisation invalide.", "danger")
        return redirect(url_for('forgot_password'))

    banquier = Banquier.query.filter_by(email=email).first_or_404()

    if request.method == 'POST':
        password = request.form['password']
        password_confirm = request.form['password_confirm']

        valide, message_erreur = valider_mot_de_passe(password)
        if not valide:
            flash(message_erreur, "danger")
            return render_template('reset_password.html', token=token)

        if password != password_confirm:
            flash("Les mots de passe ne correspondent pas.", "danger")
            return render_template('reset_password.html', token=token)

        banquier.set_password(password)
        db.session.commit()
        flash("Votre mot de passe a été réinitialisé avec succès.", "success")
        return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)


# ============================================================
# Tableau de bord / liste des clients
# ============================================================

@app.route('/')
@login_required
def dashboard():
    clients = Client.query.filter_by(banquier_id=current_user.id).all()

    # Attache à chaque client sa toute dernière demande de crédit réelle
    # (ou None s'il n'en a aucune), pour que le bouton "Voir décision"
    # renvoie vers une demande figée de l'historique plutôt que de
    # relancer une évaluation à la volée non enregistrée.
    for client in clients:
        client.dernier_credit = (
            Credit.query
            .filter_by(client_id=client.id)
            .order_by(Credit.date_demande.desc())
            .first()
        )

    return render_template('dashboard.html', clients=clients)


@app.route('/client/delete/<int:client_id>', methods=['POST'])
@login_required
def delete_client(client_id):
    client = Client.query.filter_by(id=client_id, banquier_id=current_user.id).first_or_404()
    db.session.delete(client)
    db.session.commit()
    flash("Client supprimé.", "info")
    return redirect(url_for('dashboard'))


@app.route('/credit/<int:client_id>')
@login_required
def credit(client_id):
    client = Client.query.filter_by(id=client_id, banquier_id=current_user.id).first_or_404()
    resultat = evaluer_client(client)
    return render_template('credit.html', client=client, **resultat)


# ============================================================
# Fiche client (vue légère : infos perso + historique)
# ============================================================

@app.route('/client/<int:client_id>')
@login_required
def client_fiche(client_id):
    client = Client.query.filter_by(id=client_id, banquier_id=current_user.id).first_or_404()
    return render_template('client_fiche.html', client=client)


# ============================================================
# Ajout / modification client (formulaire)
# ============================================================

@app.route('/client/add', methods=['GET', 'POST'])
@login_required
def add_client():
    if request.method == 'POST':
        client = Client(banquier_id=current_user.id)
        client.prenom = request.form['prenom']
        client.nom = request.form['nom']
        client.msisdn = request.form.get('msisdn')
        client.bankcode = request.form.get('bankcode')
        client.telephone = request.form.get('telephone')
        client.agence = request.form.get('agence')
        client.adresse = request.form.get('adresse')
        db.session.add(client)
        db.session.commit()

        flash("Client ajouté avec succès.", "success")
        return redirect(url_for('client_fiche', client_id=client.id))

    return render_template('client_new.html')


@app.route('/client/edit/<int:client_id>', methods=['GET', 'POST'])
@login_required
def edit_client(client_id):
    client = Client.query.filter_by(id=client_id, banquier_id=current_user.id).first_or_404()
    is_new_credit = request.args.get('new_credit') == '1' or request.form.get('is_new_credit') == '1'

    if request.method == 'POST':
        remplir_client_depuis_formulaire(client, request.form)
        db.session.commit()

        if request.form.get('action') == 'save_and_evaluate':
            resultat = evaluer_client(client)
            enregistrer_historique_credit(client, resultat)
            return render_template(
                'client_form.html',
                client=client,
                is_new_credit=False,
                taille_actuelle=TAILLE_ENTREPRISE_MAP_INVERSE.get(client.taille_entreprise),
                **resultat
            )

        flash("Client modifié avec succès.", "success")
        return render_template(
            'client_form.html',
            client=client,
            is_new_credit=False,
            taille_actuelle=TAILLE_ENTREPRISE_MAP_INVERSE.get(client.taille_entreprise)
        )

    return render_template(
        'client_form.html',
        client=client,
        is_new_credit=is_new_credit,
        taille_actuelle=TAILLE_ENTREPRISE_MAP_INVERSE.get(client.taille_entreprise)
    )


# ============================================================
# Historique des crédits
# ============================================================

@app.route('/credits')
@login_required
def credits_historique():
    clients = Client.query.filter_by(banquier_id=current_user.id).all()
    client_ids = [c.id for c in clients]

    credits = (
        Credit.query
        .filter(Credit.client_id.in_(client_ids))
        .order_by(Credit.date_demande.desc())
        .all()
    )

    # Une seule ligne par client : sa demande la plus récente (les crédits sont
    # déjà triés du plus récent au plus ancien), avec le nombre total de demandes.
    derniers_credits = {}
    nb_demandes = {}
    for c in credits:
        derniers_credits.setdefault(c.client_id, c)
        nb_demandes[c.client_id] = nb_demandes.get(c.client_id, 0) + 1

    return render_template(
        'credits_historique.html',
        credits=list(derniers_credits.values()),
        nb_demandes=nb_demandes,
    )


@app.route('/credits/<int:credit_id>')
@login_required
def credit_historique_detail(credit_id):
    credit_entry = Credit.query.join(Client).filter(
        Credit.id == credit_id,
        Client.banquier_id == current_user.id
    ).first_or_404()

    snapshot = json.loads(credit_entry.features_snapshot) if credit_entry.features_snapshot else {}
    facteurs = json.loads(credit_entry.facteurs_shap) if credit_entry.facteurs_shap else []

    return render_template(
        'credit_historique_detail.html',
        credit_entry=credit_entry,
        client=credit_entry.client,
        snapshot=snapshot,
        facteurs=facteurs,
        tous_credits=credit_entry.client.credits_historique,
    )


@app.route('/credits/<int:credit_id>/statut', methods=['POST'])
@login_required
def update_statut_credit(credit_id):
    credit_entry = Credit.query.join(Client).filter(
        Credit.id == credit_id,
        Client.banquier_id == current_user.id
    ).first_or_404()

    credit_entry.statut = request.form.get('statut', credit_entry.statut)
    db.session.commit()
    flash("Statut du crédit mis à jour.", "success")
    return redirect(url_for('credit_historique_detail', credit_id=credit_id))


@app.route('/credits/<int:credit_id>/delete', methods=['POST'])
@login_required
def delete_credit(credit_id):
    credit_entry = Credit.query.join(Client).filter(
        Credit.id == credit_id,
        Client.banquier_id == current_user.id
    ).first_or_404()

    client_id = credit_entry.client_id
    db.session.delete(credit_entry)
    db.session.commit()

    flash("Entrée supprimée de l'historique.", "info")
    return redirect(url_for('edit_client', client_id=client_id))


# ============================================================
# Statistiques / Reporting
# ============================================================

@app.route('/statistics')
@login_required
def statistics():
    clients = Client.query.filter_by(banquier_id=current_user.id).all()

    total = len(clients)
    acceptes = [c for c in clients if c.derniere_decision == 'Accepté']
    refuses = [c for c in clients if c.derniere_decision == 'Refusé']
    non_evalues = total - len(acceptes) - len(refuses)
    taux_acceptation = round(len(acceptes) / total * 100, 1) if total else 0

    villes = {}
    for c in clients:
        ville = (c.ville or '').strip().title() or 'Non renseignée'
        villes[ville] = villes.get(ville, 0) + 1

    contrats = {}
    for c in clients:
        contrats[c.type_contrat] = contrats.get(c.type_contrat, 0) + 1

    return render_template(
        'statistics.html',
        total=total,
        acceptes=len(acceptes),
        refuses=len(refuses),
        non_evalues=non_evalues,
        taux_acceptation=taux_acceptation,
        villes=villes,
        contrats=contrats,
    )


# ============================================================
# Paramètres / Profil
# ============================================================

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'update_profile':
            nouvel_email = request.form['email']

            if (
                nouvel_email != current_user.email
                and Banquier.query.filter_by(email=nouvel_email).first()
            ):
                flash("Cet email est déjà utilisé par un autre compte.", "danger")
                return redirect(url_for('settings'))

            current_user.nom = request.form['nom']
            current_user.email = nouvel_email
            current_user.agence = request.form.get('agence')
            current_user.telephone = request.form['telephone']

            db.session.commit()

            flash("Profil mis à jour avec succès.", "success")

        elif action == 'change_password':
            ancien = request.form['ancien_password']
            nouveau = request.form['nouveau_password']
            confirmation = request.form['confirmation_password']

            if not current_user.check_password(ancien):
                flash("Ancien mot de passe incorrect.", "danger")
                return redirect(url_for('settings'))

            valide, message_erreur = valider_mot_de_passe(nouveau)

            if not valide:
                flash(message_erreur, "danger")
                return redirect(url_for('settings'))

            if nouveau != confirmation:
                flash("Les mots de passe ne correspondent pas.", "danger")
                return redirect(url_for('settings'))

            current_user.set_password(nouveau)
            db.session.commit()

            flash("Mot de passe modifié avec succès.", "success")

        return redirect(url_for('settings'))

    nb_clients = Client.query.filter_by(
        banquier_id=current_user.id
    ).count()

    return render_template(
        'settings.html',
        nb_clients=nb_clients,
        seuil_surendettement=SEUIL_SURENDETTEMENT,
    )


# ============================================================
# Administration : supervision + gestion des comptes analystes
# ============================================================

MOIS_FR = ['janv.', 'févr.', 'mars', 'avr.', 'mai', 'juin',
           'juil.', 'août', 'sept.', 'oct.', 'nov.', 'déc.']


def stats_analyste(analyste):
    """Indicateurs d'activité d'un analyste."""
    credits = (
        Credit.query.join(Client)
        .filter(Client.banquier_id == analyste.id)
        .order_by(Credit.date_demande.desc())
        .all()
    )
    acceptes = sum(1 for c in credits if c.decision == 'Accepté')
    return {
        'nb_clients': Client.query.filter_by(banquier_id=analyste.id).count(),
        'nb_credits': len(credits),
        'acceptes': acceptes,
        'refuses': len(credits) - acceptes,
        'taux_acceptation': round(acceptes / len(credits) * 100, 1) if credits else None,
        'derniere_demande': credits[0].date_demande if credits else None,
        'credits': credits,
    }


def lire_formulaire_compte(compte=None):
    """Valide le formulaire de création/modification d'un compte.
    Retourne (donnees, message_erreur)."""
    donnees = {
        'nom': request.form.get('nom', '').strip(),
        'email': request.form.get('email', '').strip().lower(),
        'telephone': request.form.get('telephone', '').strip() or None,
        'agence': request.form.get('agence', '').strip() or None,
        'role': request.form.get('role', 'analyste'),
        'password': request.form.get('password', ''),
    }
    if not donnees['nom'] or not donnees['email']:
        return donnees, "Le nom et l'email sont obligatoires."
    if donnees['role'] not in ('analyste', 'admin'):
        return donnees, "Rôle invalide."

    existant = Banquier.query.filter_by(email=donnees['email']).first()
    if existant and (compte is None or existant.id != compte.id):
        return donnees, "Un compte existe déjà avec cet email."

    # Mot de passe obligatoire à la création, facultatif en modification
    if compte is None or donnees['password']:
        valide, message_erreur = valider_mot_de_passe(donnees['password'])
        if not valide:
            return donnees, message_erreur
    return donnees, None


@app.route('/admin')
@admin_required
def admin_dashboard():
    analystes = Banquier.query.filter_by(role='analyste').order_by(Banquier.nom).all()
    credits = Credit.query.order_by(Credit.date_demande.desc()).all()
    acceptes = sum(1 for c in credits if c.decision == 'Accepté')

    lignes = []
    for a in analystes:
        st = stats_analyste(a)
        st.pop('credits')
        lignes.append({'analyste': a, **st})
    lignes.sort(key=lambda l: l['nb_credits'], reverse=True)

    # Demandes par mois sur les 6 derniers mois
    aujourd_hui = datetime.utcnow()
    mois = []
    annee, m = aujourd_hui.year, aujourd_hui.month
    for _ in range(6):
        mois.insert(0, (annee, m))
        m -= 1
        if m == 0:
            annee, m = annee - 1, 12
    par_mois = {cle: {'acceptes': 0, 'refuses': 0} for cle in mois}
    for c in credits:
        cle = (c.date_demande.year, c.date_demande.month)
        if cle in par_mois:
            par_mois[cle]['acceptes' if c.decision == 'Accepté' else 'refuses'] += 1

    il_y_a_30_jours = aujourd_hui - timedelta(days=30)
    actifs_30j = sum(1 for a in analystes if a.derniere_connexion and a.derniere_connexion >= il_y_a_30_jours)

    return render_template(
        'admin/dashboard.html',
        nb_analystes=len(analystes),
        actifs_30j=actifs_30j,
        nb_clients=Client.query.count(),
        nb_credits=len(credits),
        acceptes=acceptes,
        refuses=len(credits) - acceptes,
        taux_acceptation=round(acceptes / len(credits) * 100, 1) if credits else 0,
        lignes=lignes,
        derniers_credits=credits[:8],
        mois_labels=[f"{MOIS_FR[m - 1]} {str(a)[2:]}" for a, m in mois],
        mois_acceptes=[par_mois[cle]['acceptes'] for cle in mois],
        mois_refuses=[par_mois[cle]['refuses'] for cle in mois],
    )


@app.route('/admin/analystes')
@admin_required
def admin_analystes():
    recherche = request.args.get('q', '').strip()
    requete = Banquier.query
    if recherche:
        motif = f'%{recherche}%'
        requete = requete.filter(
            Banquier.nom.ilike(motif) | Banquier.email.ilike(motif) | Banquier.agence.ilike(motif)
        )
    comptes = requete.order_by(Banquier.role.desc(), Banquier.nom).all()
    stats = {c.id: stats_analyste(c) for c in comptes}
    return render_template('admin/analystes.html', comptes=comptes, stats=stats, recherche=recherche)


@app.route('/admin/analystes/nouveau', methods=['GET', 'POST'])
@admin_required
def admin_analyste_create():
    if request.method == 'POST':
        donnees, erreur = lire_formulaire_compte()
        if erreur:
            flash(erreur, 'danger')
            return render_template('admin/analyste_form.html', compte=None, donnees=donnees)

        compte = Banquier(
            nom=donnees['nom'], email=donnees['email'], telephone=donnees['telephone'],
            agence=donnees['agence'], role=donnees['role'],
        )
        compte.set_password(donnees['password'])
        db.session.add(compte)
        db.session.commit()
        flash(f"Le compte de {compte.nom} a été créé.", 'success')
        return redirect(url_for('admin_analyste_detail', compte_id=compte.id))

    return render_template('admin/analyste_form.html', compte=None, donnees={})


@app.route('/admin/analystes/<int:compte_id>')
@admin_required
def admin_analyste_detail(compte_id):
    compte = Banquier.query.get_or_404(compte_id)
    clients = Client.query.filter_by(banquier_id=compte.id).order_by(Client.nom).all()
    return render_template('admin/analyste_detail.html', compte=compte, clients=clients, stats=stats_analyste(compte))


@app.route('/admin/analystes/<int:compte_id>/modifier', methods=['GET', 'POST'])
@admin_required
def admin_analyste_edit(compte_id):
    compte = Banquier.query.get_or_404(compte_id)

    if request.method == 'POST':
        donnees, erreur = lire_formulaire_compte(compte)
        if not erreur and compte.id == current_user.id and donnees['role'] != 'admin':
            erreur = "Vous ne pouvez pas retirer votre propre rôle d'administrateur."
        if erreur:
            flash(erreur, 'danger')
            return render_template('admin/analyste_form.html', compte=compte, donnees=donnees)

        compte.nom = donnees['nom']
        compte.email = donnees['email']
        compte.telephone = donnees['telephone']
        compte.agence = donnees['agence']
        compte.role = donnees['role']
        if donnees['password']:
            compte.set_password(donnees['password'])
        db.session.commit()
        flash("Les modifications ont été enregistrées.", 'success')
        return redirect(url_for('admin_analyste_detail', compte_id=compte.id))

    donnees = {'nom': compte.nom, 'email': compte.email, 'telephone': compte.telephone,
               'agence': compte.agence, 'role': compte.role}
    return render_template('admin/analyste_form.html', compte=compte, donnees=donnees)


@app.route('/admin/analystes/<int:compte_id>/supprimer', methods=['POST'])
@admin_required
def admin_analyste_delete(compte_id):
    compte = Banquier.query.get_or_404(compte_id)
    if compte.id == current_user.id:
        flash("Vous ne pouvez pas supprimer votre propre compte.", 'danger')
        return redirect(url_for('admin_analyste_detail', compte_id=compte.id))

    nom = compte.nom
    db.session.delete(compte)  # supprime aussi ses clients et leurs crédits (cascade)
    db.session.commit()
    flash(f"Le compte de {nom} a été supprimé.", 'success')
    return redirect(url_for('admin_analystes'))


# ============================================================
# Base de données : création + ajout des colonnes récentes
# ============================================================

def migrer_base():
    """Crée les tables manquantes et ajoute aux tables existantes les
    colonnes apparues depuis (SQLite ne le fait pas tout seul)."""
    db.create_all()
    colonnes = {c['name'] for c in inspect(db.engine).get_columns('banquier')}
    ajouts = {
        'role': "VARCHAR(20) NOT NULL DEFAULT 'analyste'",
        'date_creation': 'DATETIME',
        'derniere_connexion': 'DATETIME',
    }
    with db.engine.begin() as conn:
        for nom, definition in ajouts.items():
            if nom not in colonnes:
                conn.execute(text(f'ALTER TABLE banquier ADD COLUMN {nom} {definition}'))


with app.app_context():
    migrer_base()


@app.cli.command('create-admin')
@click.option('--nom', prompt='Nom complet')
@click.option('--email', prompt='Email')
@click.password_option('--password', prompt='Mot de passe')
def create_admin(nom, email, password):
    """Crée un compte administrateur (ou promeut un compte existant)."""
    compte = Banquier.query.filter_by(email=email).first()
    if compte:
        compte.role = 'admin'
        db.session.commit()
        click.echo(f"Le compte {email} est maintenant administrateur.")
        return

    valide, message_erreur = valider_mot_de_passe(password)
    if not valide:
        raise click.ClickException(message_erreur)

    compte = Banquier(nom=nom, email=email, role='admin')
    compte.set_password(password)
    db.session.add(compte)
    db.session.commit()
    click.echo(f"Administrateur {email} créé.")


if __name__ == '__main__':
    app.run(debug=True)