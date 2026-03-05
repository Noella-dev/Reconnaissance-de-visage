# 🤖 ManouRecognizer : Système Hybride de Reconnaissance Faciale

Ce projet combine la rapidité des algorithmes classiques et la puissance du Deep Learning pour créer un système de surveillance capable de détecter et d'identifier des individus en temps réel via webcam.

---

## 🧠 Architecture du Système

Le projet repose sur deux piliers technologiques complémentaires :

### 1. Détection de Visage : Algorithme de Viola-Jones (Haar Cascade)
C'est le module responsable de la localisation spatiale du visage.
* **Rôle :** Analyser le flux vidéo pour isoler les coordonnées du visage humain.
* **Fonctionnement :** Utilise des **caractéristiques de Haar** (petits filtres de contraste). Il repère par exemple que la zone des yeux est naturellement plus sombre que les pommettes ou le pont du nez.
* **Avantage :** Une latence extrêmement faible, permettant un suivi fluide à 30+ FPS sur une webcam standard.



### 2. Identification : Convolutional Neural Networks (CNN)
C'est le "cerveau" de l'application (modèle **ManouRecognizer**).
* **Rôle :** Distinguer et authentifier les traits spécifiques d'un utilisateur par rapport à une base de données de visages.
* **Fonctionnement :** * **Couches de Convolution :** Agissent comme des loupes pour extraire les textures et les bords.
    * **Couches Profondes :** Assemblent ces détails pour reconnaître des formes complexes (structure du nez, distance inter-oculaire, commissures des lèvres).
* **Précision :** Contrairement à la détection simple, le CNN apprend la signature biométrique unique de l'individu.



---

## ⚙️ Pipeline de Traitement

1.  **Capture :** Lecture du flux vidéo brut via OpenCV.
2.  **Localisation (Haar Cascade) :** Détection de la présence d'un visage et recadrage (crop).
3.  **Prétraitement :** Redimensionnement et normalisation pour l'entrée du réseau de neurones.
4.  **Inférence (CNN) :** Analyse du visage recadré par le modèle ManouRecognizer.
5.  **Résultat :** Affichage du nom de la personne identifiée et score de confiance.

---

## 🛠️ Technologies Utilisées
* **Python 3.10+**
* **OpenCV :** Pour la manipulation d'images et l'implémentation de Haar Cascade.
* **Numpy :** Pour les calculs matriciels rapides.
* **PyTorch :** Framework de Deep Learning utilisé pour l'entraînement et l'inférence du modèle CNN (`.pth`).
* **Torchvision :** Pour les transformations d'images (Resize, Normalization).

---

## 🚀 Installation & Lancement

```bash
# Installation des dépendances
pip install opencv-python tensorflow numpy

# Lancement de l'application
python main.py