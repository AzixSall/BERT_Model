"# Classificateur BERT pour l'Origine des Films Basé sur leur Synopsis" 

Ce projet implémente un classificateur qui utilise un modèle BERT pour prédire l'origine ou l'ethnicité d'un film à partir de sa description ou synopsis.

Ce projet utilise un modèle BERT (Bidirectional Encoder Representations from Transformers) pré-entraîné et l'adapte à la tâche de classification des origines/ethnicités de films basées sur leurs synopsis. BERT est particulièrement efficace pour comprendre le contexte et les nuances linguistiques présentes dans les descriptions de films.

Le projet comprend:

Un modèle PyTorch basé sur BERT
Un pipeline d'entraînement complet
Une interface de démonstration interactive créée avec Gradio

# Description des données

Le modèle est conçu pour fonctionner avec un jeu de données de films contenant au minimum:

Titre du film
Origine/Ethnicité (la cible de classification)
Synopsis/Plot (texte utilisé pour la prédiction)

Dans notre implémentation, nous utilisons un jeu de données qui contient des films de diverses origines/ethnicités:

Américain
Français
Allemand
Japonais
Brésilien
Espagnol
...

# Architecture du modèle

Notre classificateur s'appuie sur l'architecture BERT avec des ajustements spécifiques:

Couche de base:

Utilisation du modèle pré-entraîné bert-base-uncased de Hugging Face
Conservation des poids du modèle pré-entraîné pour bénéficier du transfert d'apprentissage


Modifications pour la classification:

Ajout d'une couche de dropout (0.3) pour la régularisation
Ajout d'une couche linéaire de classification qui projette la sortie de BERT vers le nombre de classes (origines/ethnicités)


Paramètres techniques:

Longueur maximale de séquence: 256 tokens
Dimension cachée: 768 (standard pour bert-base)

# Processus d'entraînement

L'entraînement du modèle se déroule comme suit:

Prétraitement des données:

Tokenisation des synopsis avec le tokenizer BERT
Conversion des étiquettes d'origine/ethnicité en indices numériques
Division des données en ensembles d'entraînement et de validation (80/20)


Configuration d'entraînement:

Optimiseur: AdamW avec learning rate de 2e-5
Fonction de perte: Cross Entropy
Nombre d'époques: 5 (ajustable)
Taille de batch: 16 (ajustable selon les ressources)
Écrêtage de gradient: 1.0 pour éviter l'explosion des gradients


Boucle d'entraînement:

Calcul des pertes sur l'ensemble d'entraînement
Évaluation sur l'ensemble de validation après chaque époque
Calcul de métriques: accuracy, classification report


Sauvegarde du modèle:

Le modèle entraîné est sauvegardé sous bert_movie_plot_classifier.pt

# Captures d'écran

