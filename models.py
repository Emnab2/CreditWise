from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class Banquier(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    nom = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    telephone = db.Column(db.String(30))

    agence = db.Column(db.String(120), nullable=True)

    # 'analyste' (utilisateur métier) ou 'admin' (supervision + gestion des comptes)
    role = db.Column(db.String(20), nullable=False, default='analyste')
    date_creation = db.Column(db.DateTime, default=datetime.utcnow)
    derniere_connexion = db.Column(db.DateTime, nullable=True)

    clients =db.relationship('Client', backref='banquier', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self):
        return self.role == 'admin'


class Client(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    banquier_id = db.Column(db.Integer, db.ForeignKey('banquier.id'), nullable=False)

    msisdn = db.Column(db.String(30))
    bankcode = db.Column(db.String(30))
    nom = db.Column(db.String(100), nullable=False)
    prenom = db.Column(db.String(100), nullable=False)

    # --- Informations personnelles (purement informatives, ne servent PAS
    # au modèle de scoring — à ne pas ajouter à FEATURE_ORDER dans app.py) ---
    adresse = db.Column(db.String(255), nullable=True)
    telephone = db.Column(db.String(30), nullable=True)
    agence = db.Column(db.String(120), nullable=True)

    statut = db.Column(db.String(50), default='Autre')
    ville = db.Column(db.String(50), default='Autre')
    type_contrat = db.Column(db.String(50), default='Autre')

    solde = db.Column(db.Float, default=0)
    solde_cashback = db.Column(db.Float, default=0)
    age = db.Column(db.Integer, default=0)
    revenu = db.Column(db.Float, default=0)
    anciennete = db.Column(db.Float, default=0)
    credits = db.Column(db.Float, default=0)
    retard = db.Column(db.Float, default=0)
    taille_entreprise = db.Column(db.Float, default=0)
    taux_endettement = db.Column(db.Float, default=0)
    dernier_mvt = db.Column(db.Float, default=0)

    surendettement = db.Column(db.Integer, default=0)

    derniere_decision = db.Column(db.String(20))
    derniere_probabilite = db.Column(db.Float)
    derniers_facteurs = db.Column(db.Text)
    derniere_shap_image = db.Column(db.Text)


class Credit(db.Model):
    """
    Une ligne = une demande de crédit évaluée par le modèle, à un instant T.
    Le client peut avoir plusieurs demandes dans le temps : chaque demande garde
    une copie (snapshot) des données utilisées pour CETTE décision précise,
    indépendamment de l'état actuel du client (qui peut avoir changé depuis).
    """
    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.Integer, db.ForeignKey('client.id'), nullable=False)

    date_demande = db.Column(db.DateTime, default=datetime.utcnow)

    decision = db.Column(db.String(20))          # Accepté / Refusé
    statut = db.Column(db.String(50), default='en cours')  # en cours / soldé / en retard / refusé

    # Snapshot des données au moment de cette décision (JSON), pour pouvoir
    # comprendre/rejouer une décision passée même après modification du client.
    features_snapshot = db.Column(db.Text)
    facteurs_shap = db.Column(db.Text)
    shap_image = db.Column(db.Text)

    client = db.relationship(
        'Client',
        backref=db.backref(
            'credits_historique',
            lazy=True,
            order_by='Credit.date_demande.desc()',
            cascade='all, delete-orphan'
        )
    )