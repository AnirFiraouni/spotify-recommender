# 🎵 Spotify Mood Recommender

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://spotify-recommender-anirfiraouni.streamlit.app])

**Une application de Data Science interactive qui recommande des musiques basées sur leurs signatures audio mathématiques.**

---

## 📋 Présentation du Projet

Ce projet a été réalisé dans le cadre de ma formation en **MAM3 (Mathématiques Appliquées et Modélisation)** à Polytech Nice Sophia.
L'objectif était de passer de la théorie mathématique à une application réelle ("Data to Production") en construisant un moteur de recommandation musical.

Contrairement aux recommandations basées sur l'historique d'écoute, ce moteur utilise le **Content-Based Filtering**. Il analyse les caractéristiques audio intrusèques des morceaux (tempo, énergie, "dansabilité") pour trouver des similarités vectorielles.

### 🚀 Fonctionnalités
* **Moteur de Recommandation :** Suggestion de 5 morceaux similaires à partir d'un titre choisi.
* **Visualisation Avancée :** Comparaison graphique (Radar Chart) des empreintes audio entre la chanson source et la recommandation.
* **Interface Web :** Application interactive déployée via Streamlit.

---

## 🧠 L'Approche Mathématique (Le cœur du projet)

Le problème de recommandation est traité ici comme un problème de **géométrie vectorielle** en dimension $N$.

1.  **Espace Vectoriel :** Chaque chanson est représentée comme un vecteur $V$ dans un espace à 5 dimensions correspondant aux "Audio Features" de Spotify :
    * $x_1$ : Danceability
    * $x_2$ : Energy
    * $x_3$ : Valence (Positivité)
    * $x_4$ : Acousticness
    * $x_5$ : Instrumentalness

2.  **Algorithme :** J'utilise l'algorithme des **K-Nearest Neighbors (K-NN)** (K-Plus Proches Voisins).

3.  **Métrique de Distance :** La similarité entre deux chansons $A$ et $B$ est calculée via la **Distance Euclidienne** :
    $$d(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}$$
    Plus la distance $d$ est faible, plus les chansons sont "proches" musicalement.

---

## 🛠️ Stack Technique

* **Langage :** Python 3.9+
* **Interface :** Streamlit
* **Machine Learning :** Scikit-Learn (NearestNeighbors)
* **Manipulation de Données :** Pandas
* **Visualisation :** Plotly Graph Objects

---

## 💻 Installation Locale

Si vous souhaitez faire tourner le projet sur votre machine :

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/](https://github.com/)[TON-PSEUDO]/spotify-recommender.git
    cd spotify-recommender
    ```

2.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Lancer l'application :**
    ```bash
    streamlit run app.py
    ```

---

## 👤 Auteur & Contact

**[Anir] [Firaouni]**
* 🎓 Étudiant en 3ème année (MAM) à **Polytech Nice Sophia**.
* 🔭 En recherche active d'une **Alternance en Data Science / Data Analysis** (Début : Septembre 2026).
* 📫 **Email :** [anir.firaouni05@gmail.com]
* 🔗 **LinkedIn :** [https://www.linkedin.com/in/firaounianir/]

