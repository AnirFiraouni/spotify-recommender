import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go

# 1. Configuration de la page
st.set_page_config(page_title="Spotify Recommender", page_icon="🎵")
 
# Barre latérale (Sidebar) pour tes infos
with st.sidebar:
    st.header("À propos")
    st.write("J'ai devloppé cette aplplication en tant que projet personnel en **MAM3 à Polytech Nice Sophia**.")
    st.info("💡 **But du projet :** Appliquer des algorithmes de KNN (Voisins les plus proches) pour la recommandation musicale.")
    st.write("---")
    st.write("📧 **Contact :** [anir.firaouni05@gmail.com]")
    st.write("🔗 **LinkedIn :** [https://www.linkedin.com/in/firaounianir/]")
st.title("🎵 Le Recommendateur d'Ambiance")
st.markdown("Choisis une chanson, je t'en trouve 5 autres mathématiquement proches !")

# 2. Chargement des données (Mets ton fichier csv au même endroit)
# Pour l'exemple, assure-toi d'avoir un fichier avec ces colonnes
@st.cache_data
def load_data():
    # Remplace par le vrai nom de ton fichier Kaggle
    df = pd.read_csv("spotify_data.csv") 
    # On garde un échantillon pour que ça aille vite si le fichier est gros
    return df.sample(n=10000).reset_index(drop=True)

try:
    df = load_data()
except:
    st.error("Erreur: Télécharge un dataset Spotify sur Kaggle et nomme-le spotify_data.csv")
    st.stop()

# 3. Préparation du Machine Learning
features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']
X = df[features]

# On entraîne le modèle (C'est super rapide sur des données numériques)
model = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
model.fit(X)

# 4. Interface Utilisateur
song_choice = st.selectbox("Rechercher une chanson :", df['track_name'].unique())

if st.button("Recommander"):
    # Trouver l'index de la chanson choisie
    song_idx = df[df['track_name'] == song_choice].index[0]
    
    # Récupérer les features de cette chanson
    song_vec = [df.loc[song_idx, features]]
    
    # Trouver les voisins (le 1er est la chanson elle-même, donc on en prend 6)
    distances, indices = model.kneighbors(song_vec)
    
    st.write("### 🎧 Si tu aimes ça, tu aimeras aussi :")
    
    # Affichage des résultats
    for i in range(1, 6): # On commence à 1 pour exclure la chanson originale
        idx = indices[0][i]
        recommended_song = df.iloc[idx]
        st.success(f"{recommended_song['track_name']} - {recommended_song['artists']}")

    # 5. La Visualisation Radar Chart (La touche pro)
    categories = features
    
    fig = go.Figure()
    
    # La chanson choisie
    fig.add_trace(go.Scatterpolar(
          r=df.loc[song_idx, features].values,
          theta=categories,
          fill='toself',
          name='Chanson choisie'
    ))
    
    # La première recommandation (pour comparer)
    rec_idx = indices[0][1]
    fig.add_trace(go.Scatterpolar(
          r=df.loc[rec_idx, features].values,
          theta=categories,
          fill='toself',
          name='Recommandation #1'
    ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    st.plotly_chart(fig)

