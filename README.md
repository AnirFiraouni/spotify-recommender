# 🎵 Spotify Mood Recommender (Big Data Edition)

Une application web interactive de Data Science qui recommande des musiques basées sur leurs signatures audio mathématiques. Développé dans le cadre de mon portfolio en **MAM3 à Polytech Nice Sophia**.

**[👉 TESTER L'APPLICATION ICI]** *(Remplace ce texte par le lien de ton site Streamlit !)*

## 🧠 L'Intelligence Artificielle derrière le projet
Ce projet utilise un algorithme de **K-Nearest Neighbors (KNN)** couplé à un `StandardScaler` pour analyser et comparer les *Audio Features* (tempo, énergie, danceability, etc.) des musiques.

### 🚀 Défis techniques relevés :
* **Traitement de Big Data :** Le modèle d'origine a été entraîné sur un dataset massif de **plus d'1,2 million de titres**, puis optimisé à 600 000 lignes pour le déploiement Cloud, garantissant une diversité de recommandation exceptionnelle.
* **Moteur de recherche local :** Implémentation d'une fonction de recherche textuelle instantanée via Pandas, contournant la dépréciation récente de l'API Spotify.
* **Data Visualisation :** Génération de *Radar Charts* interactifs (Plotly) pour expliquer visuellement et prouver mathématiquement la pertinence des recommandations à l'utilisateur.

## 🛠️ Technologies Utilisées
* **Python** (Pandas, Scikit-Learn)
* **Streamlit** (Interface Web et mise en cache des données)
* **Plotly** (Graphiques interactifs)
* **Git / GitHub** (Déploiement)
