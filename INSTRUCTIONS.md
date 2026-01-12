Projet à faire en binôme ou trinôme

L'objectif du projet est de concevoir et de mettre en œuvre de bout en bout une application innovante qui réponde à un cas d’usage. Vous pouvez par exemple reprendre et étendre votre projet sur le SI d'un opérateur de randonnées à vélo. Dans tous les cas prévoir des utilisateurs qui ont des avis sur des items. 

Vous devez démontrer vos compétences acquises dans les 3 modules : web de données, ingénierie des connaissances et web sémantique. Vous devez réaliser les tâches suivants :

- Modélisation ontologique en OWL et SKOS en fonction du type des données. L'ontologie OWL doit être riche (hiérarchies de classes et de propriétés, définitions de classes, propriétés algébriques des propriétés). L'ontologie et le ou les thesauri doivent être conçus et réalisés spécifiquement pour les données du projet (ne pas réutiliser un vocabulaire existant pour le cœur de votre modèle). Elle peut être enrichie par une base de règles d'inférence implémentées avec SPARQL. 

- Caractérisation de la structure du graphe de connaissances attendu par un ensemble de contraintes SHACL (possiblement incluant des contraintes de respect des définitions des classes de l’ontologie).

- Population de l'ontologie (création et description d'instances) et alimentation du thésaurus à partir de données hétérogènes :

    - Construire un graphe de connaissances à partir de données non structurées (textes en language naturel) disponibles sur le web et pouvant être relatives à un ou plusieurs sujets d'intérêt pour l'application. Ceci en appliquant les techniques d'extraction de l'information (IE) les plus convenables et adaptées parmi celles étudiées en TD, allant des techniques manuelles aux modèles de langage avancés comme les transformers et LLMs.

    - Enrichir le graphe de connaissances et/ou les thesauri à l’aide de données structurées : données CSV, JSON ou XML "liftées" en RDF avec le langage de mapping RML et RMLmapper, de la même manière que ce qui a été réalisé en TD. Vous pouvez réutiliser des données utilisées lors du TD, mais le modèle ontologique cible doit être celui que vous avez défini dans ce projet, à la différence des TDs.


- Toutes les données RDF produites à partir des données hétérogènes structurées et non structurées doivent être intégrées selon le modèle ontologique développé.

- Alignement (automatique ou semi-automatique) de votre ontologie et thésaurus avec des vocabulaires du web de données liées (e.g. ontologie DBpedia, Wikidata, Schema.org …), et liage (automatique ou semi-automatique) des resources créées avec des ressources du web de données liées (e.g. DBpedia, Wikidata, …).

- Exploitation du graphe de connaissances :

    - Implémenter des questions de compétences avec des requêtes SPARQL (complexes et intéressantes) dont une requête fédérée (utilisant la clause SPARQL SERVICE) : les principales fonctionnalités de l'application doivent être implémentées avec des requêtes SPARQL appliquées au graphe construit.

    - Implémenter une recommandation d'item (par exemple une randonné vélo) via la tâche de link prediction.

    - Implémenter et comparer deux approches GraphRAG :

        - pour répondre à des questions en langage naturel en les transformant en requêtes SPARQL sur le graphe et exécutant les requêtes obtenues ;

        - pour répondre en langage naturel à des questions en langage naturel en calculant des embeddings.

- L'application doit avoir une interface (minimale) pour communiquer avec ses utilisateurs cibles, illustrant le cas d’usage. Il est recommandé d'utiliser une interface web, mais d'autres interfaces (e.g. en ligne de commande) peuvent être acceptées. Dans tous les cas, cette partie est démonstrative, mais ne doit pas vous demander beaucoup de temps comparé au reste du projet.
 
Vous devez rédiger un rapport et préparer une présentation orale avec démo décrivant le cas d'usage considéré, les spécifications de votre application, votre modélisation, votre graphe de connaissances, la façon dont vous l'avez construit à partir de sources hétérogènes, les fonctionnalités/questions de compétence implémentées. Svp, pas de généralités ChatGPT, des informations utiles et réfléchies dans votre rapport.