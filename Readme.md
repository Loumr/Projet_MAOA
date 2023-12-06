
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

### A FAIRE
- ALGO GENERAL
&ensp;Choix des villes          <<<<<>>>>><br />
&ensp;Itération  => uniquement pour la méta heuristique<br />
&ensp;&ensp;Voyageur de commerce<br />
&ensp;&ensp;Evaluation<br />
&ensp;&ensp;Changement d'une station<br />
&ensp;Voyageur de commerce      <<<<<>>>>>><br />

- Visualisation des données + Comparaison des résultats

- Heuristique:
&ensp;évaluation d'une solution

- Méta heuristique
&ensp;algo de choix des stations
&ensp;algo de changement de station
&ensp;voyageur de commerce heuristique
&ensp;voyageur de commerce avancé
- Formulation compact
- Formulation non-compacte
- Mini-rapport + Analyse critique


### Critères:
- Coût de construction du métro (nombre de stations et longueur des tronçons)
- Temps de trajets moyen d'une ville à une autre
- Ration moyen entre marche à pied et trajet en métro
> L'évalutation d'une solution se fait en n².
