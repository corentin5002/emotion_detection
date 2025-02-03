# Emotion Detection

## Projet Fil Rouge - 5EMOT - Reconnaissance Emotionnelle

![Reconnaissance Emotionnel](/Assets/Emotion-Recognition.jpg "Reconnaissance Emotionnel")

## Objectif

Analyser les émotions d'une personne grâce à un appareil de capture vidéo en direct.

## Ressources

- **Streamlit** : Support visuel de l'application (UI/UX)
- **[Modèle d'apprentissage](model_MobilNetV3.h5)** : Fichier de notre modèle d'IA pour la reconnaissance émotionnelle
- **[Conda](https://www.anaconda.com/download)** : Support d'application pour le fonctionnement du projet

## Étapes de fonctionnement

### Prérequis

Avant de lancer l'application, assurez-vous d'avoir les éléments suivants installés :

- **Python** (>=3.10 recommandé)
- **Conda** ou **Virtualenv** pour gérer les dépendances
- **Streamlit** pour l'interface utilisateur
- **TensorFlow/Keras** pour charger et exécuter le modèle de reconnaissance émotionnelle

### Installation

1. **Cloner le projet**  
   ```bash
   git clone https://github.com/corentin5002/emotion_detection.git
   cd emotion_detection
   ```

2. **Créer et activer un environnement virtuel**  
   - Avec Conda :
     ```bash
     conda env create -f environment.yml
     conda activate emotion_detection
     ```
   - Avec Virtualenv :
     ```bash
     python -m venv venv
     source venv/bin/activate  # Sur macOS/Linux
     venv\Scripts\activate  # Sur Windows
     ```

3. **Installer les dépendances**  
   ```bash
   pip install -r requirements.txt
   ```

### Lancement de l'Application

Une fois l'installation terminée, vous pouvez exécuter l'application Streamlit :

```bash
streamlit run app.py
```

### Structure du Projet

```
emotion_detection/
│── app.py                 # Application principale Streamlit
│── functions.py           # Fonctions utilitaires pour la détection d’émotions
│── test.py                # Scripts de test
│── model_MobilNetV3.h5    # Modèle d’apprentissage pré-entraîné
│── requirements.txt       # Liste des dépendances Python
│── environment.yml        # Configuration de l'environnement Conda
│── command-launch.sh      # Script de lancement (optionnel)
│── README.md              # Documentation du projet
│── .gitignore             # Fichiers à ignorer dans Git
```

### Fonctionnalités de l'Application

- **Détection d'émotions en direct** via webcam
- **Affichage des résultats en temps réel** avec Streamlit
- **Modèle basé sur MobilNetV3** optimisé pour la classification rapide

### Tests

Pour exécuter les tests unitaires :

```bash
pytest test.py
```

### Auteurs et Contributeurs

Projet réalisé dans le cadre du **Projet Fil Rouge - 5EMOT** par :

- **Correntin SABIER**
- **Théo MOURGUES**
- **Valentin POIGT**
- **Jonah DUVILLARD**
```
