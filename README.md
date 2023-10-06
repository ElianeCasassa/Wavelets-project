# Wavelets-project
 
Projet réalisé et implémenté par Eliane Casassa (MSIAM 2) pour le cours de wavelets en 3ème année à l'ENSIMAG en 2021/2022.

Objectif du projet : Choisir et implémenter un papier de recherche utilisant les ondelettes

Papier utilisé :
An edge detection approach based on directional wavelet transform publié en 2009 par Zhen Zhang, Siliang Ma, Hui Liu et Yuexin Gong.

Introduction : 
La détection de contours constitue souvent une étape préliminaire à de nombreuses applications en traitement d’images comme la reconnaissance d’objets, de formes ainsi que la compression d’images. En effet les contours sont les parties les plus informatives d’une image. Il existe beaucoup d’algorithmes permettant la détection des contours mais les plus utilisés sont les algorithmes de Prewitt, de Sobel et de Canny. Ce sont des algorithmes basés sur le calcul des gradients des images. Mallat s’est également penché sur la question en utilisant la transformée en ondelettes.

Pour ma part, j’ai choisi d’étudier le papier An edge detection approach based on directional wavelet transform publié en 2009 par Zhen Zhang, Siliang Ma, Hui Liu et Yuexin Gong [2]. Ce n’est que plus tard que j’ai compris que ce papier s’est très fortement inspiré des papiers de Vladan Velisavljevic sur les Directionlets [1] et qu’il manquait d’explications. Malgré cela j’ai décidé de l’implémenter mais en y apportant des changements ainsi que davantage de contenu.

Les modifications apportés par ce papier consistent principalement à augmenter le nombre de directions pour lesquelles on calcule la transformée en ondelettes tout en gardant le caractère séparable de la transformée en ondelettes 2D (pour la simplicité des calculs). La suite de ce rapport est constitué de deux parties. On développera dans un premier temps l’algorithme en lui-même ainsi que son implémentation informatique puis dans un second temps on comparera cet algorithme avec l’algorithme de Canny ainsi que la transformée en ondelettes classique (suivant les directions horizontales et verticales seulement).





Afin de lancer l'exécution du projet veuillez choisir les paramètres voulu ainsi que l'image utilisée dans le fichier main puis exécuter :
                          python3 main.py

Les images à utiliser se trouvent dans le dossier "Images" et les résultats s'enregistrent dans le dossier associé à la méthode choisie (Canny, standard_WT, directional) avec chacunes des étapes de l'algorithme.
