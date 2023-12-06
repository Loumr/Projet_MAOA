
# Problème de métro circulaire

### Binôme:
Lou Moulin-Roussel<br />
Maxence Maire<br />

### Méthode Méta-Heuristique:
1. Choisir l'ensemble de villes où l'on va mettre des stations
2. Recherche locale:
- Voyageur de commerce rapide pour trouver l'ordre des stations
- Evaluation des solutions, et suppression des mauvaises solutions
- Changer une station pour chercher une nouvelle solution et génération de X nouvelles configurations pour l'itération suivante de la boucle
3. Voyageur de commerce poussé pour trouver une bonne solution

### Méthode exacte (formulation compacte ou non-compacte)
1. Choisir l'ensemble de villes où l'on va mettre des stations
2. Voyageur de commerce exact pour trouver la meilleure solution

<br />
<br />

+ visualisation de chaque point à chaque itération pour la méta heuristique

### A FAIRE
- ALGO GENERAL
&ensp;Choix des villes<br />
&ensp;Itération  => uniquement pour la méta heuristique<br />
&ensp;&ensp;Voyageur de commerce<br />
&ensp;&ensp;Evaluation<br />
&ensp;&ensp;Changement d'une station<br />
&ensp;Voyageur de commerce<br />
- Visualisation des données + Comparaison des résultats
- Heuristique:
&ensp;évaluation d'une solution
- Méta heuristique cf Critères<br />
&ensp;algo de choix des stations => Heuristique randomisée répétée du problème du médian<br />
&ensp;algo de changement de station => cf Changement de station<br />
&ensp;voyageur de commerce heuristique => cf Heuristique du proche voisin<br />
&ensp;voyageur de commerce avancé => résolution exacte (Gurobi/CPlex)
- Formulation compact
- Formulation non-compacte
- Mini-rapport + Analyse critique

TSP => projet Concorde<br />

PLNE -> TSP Généralisé avec P-median quelconque<br />
(ou Anneau Etoile)<br />

### Changement de station
On remplace une station par une de ses villes voisines qui n'est pas une station.<br />
Générer X (100?) nouvelles configurations à partir des paramètres suivants:
- meilleure station 20% des configurations
- pire station 20% des configurations
- stations aléatoire 60% des configurations

### Critères:
- Coût de construction du métro (nombre de stations et longueur des tronçons) (le nombre de stations est fixé)
- Temps de trajets moyen d'une ville à une autre
- Ration moyen entre marche à pied et trajet en métro<br />
L'évalutation d'une solution se fait en n².
<br />
<br />

### Formulations PLNE
- Formulation PLNE compacte => P-Median puis TSP Concorde
- Formulation PLNE non compacte => P-Median (cluster) puis TSP Généralisé

> PL Formulation non compacte = nombre exponentiel de contraintes et de variables

## TO-DO LIST

### Fait

### En cours
- Lou : Formulation Compacte (P Median + TSP Concorde) => méthodes exactes
- Maxence : Visualisation des données <br />

### A faire
- Comparaison des résultats (temps de calcul, complexité, qualité des solutions)
- Récupérer l'évaluation du prof (temps de trajets moyens)
- Tester automatiquement sur des instances
- Heuristique
- Méta-Heuristique
- Formulation non compacte (k-partition + TSP Généralisé) => méthodes exactes
- Mini-rapport (3 pages)
