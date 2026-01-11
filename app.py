import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Sonar Rock vs Mine",
    page_icon="🚢",
    layout="wide"
)

# --- FONCTIONS (CACHÉES) ---
@st.cache_data
def load_data():
    # Chargement des données
    # Assurez-vous que le fichier csv est dans le même dossier
    df = pd.read_csv('sonar_data.csv', header=None)
    return df

@st.cache_resource
def train_model(df):
    X = df.drop(columns=60, axis=1)
    Y = df[60]
    
    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
    
    # Train
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    
    # Evaluate
    train_acc = accuracy_score(model.predict(X_train), Y_train)
    test_acc = accuracy_score(model.predict(X_test), Y_test)
    
    return model, train_acc, test_acc

def get_mean_signatures(df):
    # Calcul des moyennes pour visualisation
    means = df.groupby(60).mean()
    return means.loc['R'], means.loc['M']

# --- CHARGEMENT ---
try:
    df = load_data()
    model, train_acc, test_acc = train_model(df)
    mean_rock, mean_mine = get_mean_signatures(df)
except FileNotFoundError:
    st.error("Erreur : Le fichier 'sonar_data.csv' est introuvable. Veuillez le placer dans le même dossier que ce script.")
    st.stop()

# --- INTERFACE UTILISATEUR ---

# Sidebar (Menu latéral)
with st.sidebar:
    st.header("🔍 À propos du modèle")
    st.write(f"**Type :** Régression Logistique")
    st.write(f"**Précision (Train) :** {train_acc:.2%}")
    st.write(f"**Précision (Test) :** {test_acc:.2%}")
    st.markdown("---")
    st.write("Ceci est une démonstration de détection d'objets sous-marins basée sur les fréquences sonar.")
    
    # Bouton pour remplir avec un exemple
    st.subheader("Tester rapidement")
    example_type = st.radio("Choisir un exemple de données :", ("Aucun", "Exemple Roche", "Exemple Mine"))

# Titre Principal
st.title("🚢 Prédicteur Sonar : Roche ou Mine ?")
st.markdown("""
Cette application permet de classifier les échos sonar. 
Les mines et les roches ont des signatures fréquentielles différentes.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Entrée des Données")
    st.write("Entrez les 60 valeurs fréquentielles (séparées par des virgules).")
    
    # Valeurs par défaut pour les exemples
    default_val = ""
    if example_type == "Exemple Roche":
        # Une vraie ligne "Roche" du dataset
        default_val = "0.02, 0.0371, 0.0428, 0.0207, 0.0954, 0.0986, 0.1539, 0.1601, 0.3109, 0.2111, 0.1609, 0.1582, 0.2238, 0.0645, 0.066, 0.2273, 0.31, 0.2999, 0.5078, 0.4797, 0.5783, 0.5071, 0.4328, 0.555, 0.6711, 0.6415, 0.7104, 0.808, 0.6791, 0.3857, 0.1307, 0.2604, 0.5121, 0.7547, 0.8537, 0.8507, 0.6692, 0.6097, 0.4943, 0.2744, 0.051, 0.2834, 0.2825, 0.4256, 0.2641, 0.1386, 0.1051, 0.1343, 0.0383, 0.0324, 0.0232, 0.0027, 0.0065, 0.0159, 0.0072, 0.0167, 0.018, 0.0084, 0.009, 0.0032"
    elif example_type == "Exemple Mine":
        # Une vraie ligne "Mine" du dataset
        default_val = "0.0491, 0.0279, 0.0592, 0.127, 0.1772, 0.1908, 0.2217, 0.0768, 0.1246, 0.2028, 0.0947, 0.2497, 0.2209, 0.3195, 0.334, 0.3323, 0.278, 0.2975, 0.2948, 0.1729, 0.3264, 0.3834, 0.3523, 0.541, 0.5228, 0.4475, 0.534, 0.5323, 0.3907, 0.3456, 0.4091, 0.4639, 0.558, 0.5727, 0.6355, 0.7563, 0.6903, 0.6176, 0.5379, 0.5622, 0.6508, 0.4797, 0.3757, 0.3051, 0.1995, 0.1073, 0.0588, 0.0251, 0.0174, 0.025, 0.0081, 0.0129, 0.0161, 0.0063, 0.0119, 0.0194, 0.014, 0.0332, 0.0439, 0.0198"

    input_text = st.text_area("Données brutes :", value=default_val, height=200)
    predict_btn = st.button("Lancer l'analyse 🚀", type="primary")

with col2:
    st.subheader("2. Analyse & Visualisation")
    
    if predict_btn and input_text:
        try:
            # Conversion
            data_list = [float(x.strip()) for x in input_text.split(',')]
            
            if len(data_list) != 60:
                st.error(f"⚠️ Erreur : On attend 60 valeurs, vous en avez fourni {len(data_list)}.")
            else:
                # Prédiction
                input_np = np.asarray(data_list).reshape(1, -1)
                prediction = model.predict(input_np)
                prob = model.predict_proba(input_np) # Probabilité
                
                # --- RESULTAT ---
                if prediction[0] == 'R':
                    st.success(f"### Résultat : 🪨 ROCHE (Rock)")
                    confidence = prob[0][1] # Proba d'être Rock (si 'R' est la classe 1, à vérifier, sinon classe 0)
                    # En sklearn binary, classes_[0] est souvent la classe negative (M) et [1] la positive (R). 
                    # On vérifie ça :
                    label = "Sûr à {:.1f}%".format(np.max(prob)*100)
                    st.metric("Confiance du modèle", label)
                else:
                    st.error(f"### Résultat : 💣 MINE (Mine)")
                    label = "Sûr à {:.1f}%".format(np.max(prob)*100)
                    st.metric("Confiance du modèle", label)

                # --- GRAPHIQUE ---
                st.markdown("#### Comparaison du signal")
                st.write("Le graphique ci-dessous compare votre entrée avec la signature moyenne des Mines et des Roches.")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(mean_rock, label="Signature Moyenne ROCHE", color='green', linestyle='--', alpha=0.7)
                ax.plot(mean_mine, label="Signature Moyenne MINE", color='red', linestyle='--', alpha=0.7)
                ax.plot(data_list, label="VOTRE SIGNAL", color='blue', linewidth=2.5)
                
                ax.set_title("Analyse Spectrale du Signal")
                ax.set_xlabel("Fréquence (Index 0-60)")
                ax.set_ylabel("Énergie")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)

        except ValueError:
            st.warning("Erreur de format : Assurez-vous que les données sont des nombres séparés par des virgules.")
    
    elif not input_text:
        st.info("👈 En attente de données dans le panneau de gauche.")