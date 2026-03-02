import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go

# --- 1. CONFIGURATION DE LA PAGE ET SIDEBAR ---
st.set_page_config(page_title="Spotify Recommender", page_icon="🎵", layout="centered")

with st.sidebar:
    st.header("À propos")
    st.write("J'ai développé cette application en tant que projet personnel en **MAM3 à Polytech Nice Sophia**.")
    st.info("💡 **But du projet :** Appliquer l'algorithme KNN sur une base de données Big Data pour la recommandation musicale.")
    st.write("---")
    st.write("📧 **Contact :** [anir.firaouni05@gmail.com]")
    st.write("🔗 **LinkedIn :** [https://www.linkedin.com/in/firaounianir/]")

st.title("🎵 Le Recommandateur d'Ambiance")
st.markdown("Choisis une chanson, l'IA t'en trouve 5 autres mathématiquement proches parmi **des milliers** de titres !")

# --- 2. CHARGEMENT DES DONNÉES (CACHE) ---
@st.cache_data
def load_data():
    # ⚠️ REMPLACE ICI PAR LE NOM DE TON NOUVEAU GROS FICHIER CSV 
    # Astuce : Si tu le mets sur GitHub, compresse-le en .zip et mets : pd.read_csv("spotify_data.zip", compression='zip')
    df = pd.read_csv("spotify_600k.zip", compression='zip')
    
    # On définit les variables mathématiques
    features_knn = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'instrumentalness', 'liveness']
    
    # Nettoyage rapide (on enlève les lignes vides pour éviter les bugs)
    df_clean = df.dropna(subset=features_knn).copy()
    
    # Si la colonne s'appelle 'track_name' au lieu de 'name', on standardise :
    if 'track_name' in df_clean.columns:
        df_clean = df_clean.rename(columns={'track_name': 'name'})
    if 'artist_names' in df_clean.columns:
         df_clean = df_clean.rename(columns={'artist_names': 'artists'})
         
    return df_clean, features_knn

@st.cache_resource
def train_model(df, features):
    # Le scaler est obligatoire pour que le Tempo n'écrase pas l'Energy !
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])
    
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(df_scaled)
    return scaler, knn

try:
    df, features_knn = load_data()
    scaler, knn = train_model(df, features_knn)
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")
    st.stop()

st.divider()

# --- 3. MOTEUR DE RECHERCHE ---
st.markdown("### 🔍 Recherche ta musique")
mot_cle = st.text_input("Tape le titre de la musique (ex: 'Get Lucky') et valide avec Entrée :")

if mot_cle:
    mot_cle = mot_cle.lower()
    masque = df['name'].fillna('').str.lower().str.contains(mot_cle)
    resultats = df[masque].head(15)

    if resultats.empty:
        st.warning("😕 Aucune musique ne contient ce titre.")
    else:
        options = []
        for index, row in resultats.iterrows():
            options.append(f"{row['name']} - par {row['artists']} (Index: {index})")
            
        choix_utilisateur = st.selectbox("Choisis la version exacte :", options)
        
        if st.button("Trouver des recommandations", type="primary"):
            
            # --- 4. RECOMMANDATION KNN ---
            index_choisi = int(choix_utilisateur.split("(Index: ")[1].replace(")", ""))
            chanson_cible = df.loc[index_choisi]
            
            st.success(f"🎵 Analyse mathématique de : **{chanson_cible['name']}**")
            
            donnees_cible = pd.DataFrame([chanson_cible[features_knn]], columns=features_knn)
            caracteristiques_cible = scaler.transform(donnees_cible)
            
            distances, indices = knn.kneighbors(caracteristiques_cible)
            
            st.subheader("### 🎧 Si tu aimes ça, tu aimeras aussi :")
            
            # Affichage des recommandations avec lien Spotify
            for i in range(1, 6):
                idx_voisin = indices[0][i]
                voisin = df.iloc[idx_voisin]
                similarite = (1 - distances[0][i]) * 100
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{i}. {voisin['name']}** - {voisin['artists']}")
                    
                    # AJOUT DU LIEN SPOTIFY (On vérifie si la colonne s'appelle 'id' ou 'track_id')
                    id_col = 'id' if 'id' in df.columns else 'track_id' if 'track_id' in df.columns else None
                    if id_col:
                        st.markdown(f"[▶️ Écouter sur Spotify](https://open.spotify.com/track/{voisin[id_col]})")
                with col2:
                    st.caption(f"{similarite:.1f}% match")
            
            st.divider()

            # --- 5. LE RADAR CHART (Adapté pour les valeurs 0 à 1) ---
            st.subheader("🕸️ Comparaison visuelle (Musique cible vs Top 1)")
            
            # On ne prend QUE les variables qui vont de 0 à 1 pour le graphique !
            # (Le Tempo est autour de 120, ça casserait l'affichage du graphique)
            radar_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness']
            
            fig = go.Figure()
            
            # La chanson choisie
            fig.add_trace(go.Scatterpolar(
                  r=chanson_cible[radar_features].values,
                  theta=radar_features,
                  fill='toself',
                  name=f"Cible : {chanson_cible['name']}"
            ))
            
            # La première recommandation (Index 1)
            rec_idx = indices[0][1]
            premiere_reco = df.iloc[rec_idx]
            fig.add_trace(go.Scatterpolar(
                  r=premiere_reco[radar_features].values,
                  theta=radar_features,
                  fill='toself',
                  name=f"Recommandation : {premiere_reco['name']}"
            ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])), 
                showlegend=True,
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)