# Sonar Rock vs Mine Prediction

Ce projet est une application de **Machine Learning** capable de prédire si un objet détecté par un sonar est une **Roche (Rock)** ou une **Mine (Mine)**.

L'application utilise un modèle de **Régression Logistique** entraîné sur le jeu de données *Sonar* (208 échantillons, 60 fréquences). Elle est déployée via **Streamlit** pour offrir une interface utilisateur interactive.

##  Structure du Projet

- `app.py` : Le code principal de l'application Streamlit.
- `sonar_data.csv` : Le jeu de données utilisé pour l'entraînement (copie de `sonar.all-data`).
- `requirements.txt` : La liste des dépendances Python nécessaires.
- `.streamlit/config.toml` : Configuration du thème .

## Installation et Lancement

### Pré-requis
Assurez-vous d'avoir **Python** installé sur votre machine.

### 1. Cloner ou télécharger le projet
Placez tous les fichiers dans un dossier local.

### 2. Installer les dépendances
Ouvrez un terminal dans le dossier du projet et exécutez :
```bash
pip install -r requirements.txt
