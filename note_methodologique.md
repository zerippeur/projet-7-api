# Projet d'implémentation d'un modèle de scoring
### <i>Projet réalisé dans le cadre de la formation 'Data Scientist' d'Openclassrooms</i>

Liens utiles :   
- Compétition Kaggle à l'origine du projet : <a>https://www.kaggle.com/c/home-credit-default-risk</a>
- Lien vers les données : <a>https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip</a>
- Feature engineering utilisé : <a>https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features</a>

***

## Méthodologie d'entraînement du modèle  

#### Objectif et contexte du projet
  L'objectif de ce projet est de créer une api de prédiction pour déterminer l'octroi d'un crédit à la consommation à un client donné.  

  Cette api sera accompagnée d'un dashboard qui permettra de sélectionner le client en question, de requêter l'api et de donner une interprétation du résultat ainsi que visualiser le positionnement du client par rapport aux autres clients.
  Dans ce cadre, nous proposons une méthodologie classique d'entraînement de modèle, avec tracking via mlflow.
  Nous sommes partis d'un feature engineering existant, afin de nous concentrer sur la partie modélisation (voir lien utile n°3)
  
#### Description des données
  Les données sont décrites sur la page du projet kaggle dont est issu le projet (voir lien utile n°1, onglet <i>data</i>) et téléchargeables (voir lien utile n°2). La cible à prédire correspond au défaut de remboursement du crédit (1 : le client a eu des difficultés à rembourser son crédit).

#### Prétraitement des données
  Nous avons conservé le feature engineering existant, mis en place pour un Light GBM Classifier (LGBMC), en ajoutant quelques étapes supplémentaires afin de l'adapter pour qu'il soit testé sur un Random Forest Classier (RFC) et un XGBoost Classifier (XGBC).
  Nous avons notamment inputé les NaN, en prenant soin d'imputer le jeu test à partir du train.
  Les valeurs infinies ont ainsi été transformées en NaN, puis imputées à la médiane.
  Les dernières étapes concernent la séparation en train et test, l'uniformisation des colonnes entre les deux jeux de données et leur sauvegarde dans une base de données.

#### Sélection des modèles
  Après observation du leaderboard de la compétition, nous avons choisi de travailler avec trois types de classifieurs : LGBMC, XGBC et RFC.
  C'est ce choix qui nous a amené à revenir sur le feature engineering afin de l'adapter aux RFC qui n'ont pas de système de gestion intégré des valeurs manquantes.

#### Entraînement et validation
##### Validation croisée
  Nous avons choisi d'ajuster le modèle en utilisant une validation croisée randomisée à 5 plis avec 50 itérations.   

  Cela permet d'étendre l'exploration des hyperparamètres dans l'espace de recherche et de gagner du temps lors de la recherche par rapport à une simple recherche sur grille (GridSearchCV).

##### Score d'ajustement et seuil d'optimisation du coût métier
  Nous avons opté pour une optimisation du modèle grâce à un score custom (fonction de coût métier).   
  
  Nous avons laissé la fonction de perte par défaut du modèle (log loss pour LGBMC et XGBC, Gini impurity pour RFC) et choisi le score custom pour le fit de la validation croisée.
  Ces choix sont délibérés, car les fonctions de pertes implémentées par défaut sont optimisées pour leurs capacités de convergence.  

#### Tracking
  Nous avons utilisé mlflow via la plateforme dagshub afin d'enregistrer les entraînements des modèles et les différents indicateurs.   

  Nous avons choisi d'enregistrer le seuil optimal et le graphique associé, ainsi que les scores suivants : auc et accuracy sur le jeu train (avec équilibrage et en conditions réelles), auc et accuracy sur le jeu test (conditions réelles), score custom de la validation croisée (moyenne sur les plis). Nous avons aussi choisi d'enregistrer un shap explainer entraîné sur le jeu de train équilibré afin de gagner du temps lors du déploiement de l'api de prédiction. Enfin, le seuil, issu du second processus d'optimisation, ainsi que le graphique de recherche d'optimum, ont aussi été enregistrés via mlflow.  

#### Sélection du modèle

  Ayant des performances similaires entre LBGMC et XGBC (RFC en retrait), nous avons choisi un modèle LGBMC pour plusieurs raisons :
  - Modèle plus rapide à entraîner (temps de calcul au moins 2x plus rapide)
  - Modèle plus léger à mettre en place lors du déploiement (librairie moins volumineuse)
  - Modèle permettant plus de flexibilité lors du feature engineering (utile pour des travaux futurs et l'ajout éventuel de nouvelles variables)

***

## Traitement du déséquilibre des classes

#### Description du problème

  Les données utilisées présentent un fort déséquilibre des classes, la classe positive (clients ayant eu des difficultés à rembourser leur crédit) représentant moins de 10% du total de la population.  

#### Traitement appliqué

  Nous avons mis en place un randomundersampler afin de traiter ce déséquilibre. Cela nous permet d'obtenir un jeu équilibré (ratio 1:1) sans créer de fausses données sur la classe à prédire, tout en conservant suffisamment d'information, étant donnée la taille conséquente du jeu initial. Nous avons cependant tenu à calculer les métriques d'accuracy et de roc_auc sur le jeu équilibré et sur le jeu non équilibré, afin de mesurer les performances du modèle en conditions réelles. Par ailleurs, les métriques de test (sauf cas de la validation croisée), sont effectués sur le jeu test non équilibré.

***

## Fonction coût métier 

  Lors de la phase d'optimisation du modèle, le score métier est calculé pour toutes les valeurs de seuil possibles (probabilités du vecteur y_pred). Nous nous commes placés dans une optique de gain et pertes réelles avec des frais de crédit correspondant à 10% du montant du crédit (ce qui revient au ratio demandé de 1 pour 10 entre les faux négatifs et les faux positifs). On obtient les coûts suivants :   
  - Faux négatif : -10 <i>(On prête 10 à un client qui ne va pas rembourser son crédit : on perd tout le crédit prêté, soit 10 unités de coût arbitraires)</i>
  - Vrai négatif : +1 <i>(On prête à un client qui va rembourser son crédit : on gagne les frais du crédit, soit 1 unité de coût arbitraire)</i>   

#### Indicateurs complémentaires

  Une fois la recherche randomisée effectuée, nous retrouvons le seuil grâce à une nouvelle recherche d'optimum de la fonction de coût métier avec les hyperparamètres obtenus sur le jeu test.
  Nous avons aussi calculé et enregistré les indicateurs suivants pour les différentes valeurs de seuil: précision, rappel, accuracy, fonction fbeta (beta = np.sqrt(5)) et tracé les courbes correspondantes sur le jeu test. Le graphique permet de repérer facilement le seuil optimal.

  ![Graphique d'optimisation du seuil à l'aide de la courbe de coût](images/f_beta_plot_LGBMClassifier-tuning_v7.png)

***

## Tableau de synthèse des résultats

Les résultats des différents modèles sont regroupés dans les tableaux suivants :

![Tableau détaillé des runs](images/Run_table_duration.png)

![Tableau détaillé des métriques](images/Run_table_metrics.png)

À résultats similaires, la version 3 du LGBMC est retenue.

***

## Interprétabilité globale et locale du modèle

#### Utilisation de SHAP pour l'interprétabilité du modèle  

  Nous avons décidé d'utiliser la librairie SHAP afin d'interpréter les résultats du modèles à l'échelle globale et locale. L'explainer a été initialisé sur le jeu train équilibré afin d'avoir comme référence, les données sur lesquelles le modèle a été entraîné.

  Les valeurs de SHAP accessibles sur le dashboard pour l'échelle globale ont été calculées à partir d'un échantillon de 2000 individus du jeu train non équilibré. Cette limite de 2000 individus a été mis en place pour éviter un temps de réponse trop long entre le dashboard et l'api (Heroku retournant un timeout à partir d'un délai de 30 secondes et plus).

#### Présentation des résultats

##### Échelle globale

![Graphique SHAP échelle globale](images/Shap_global_scale.png)

##### Échelle locale

  Pour un crédit accordé :

![Graphique SHAP échelle locale crédit accordé](images/Shap_local_scale_100003_SAFE.png)

  Pour un crédit refusé :

![Graphique SHAP échelle locale crédit refusé](images/Shap_local_scale_100002_NOPE.png)

***

## Limites et améliorations possibles

#### Ajout séquentiel des features

  Afin de trouver un modèle optimal minimal, nous pourrions ajouter les features au fur et à mesure et calculer le gain de performance. Cela permettrait de sélectionner seulement les features nécessaires et suffisantes à la capture de la variabilité et d'obtenir un modèle plus synthétique. Cette approche permettrait de diminuer la taille du modèle ainsi que celle du jeu de données nécessaire à sa mise en place. Elle est toutefois plus longue à mettre en place en termes de code.

#### Autres modèles  

  D'autres modèles sont utilisés à des fins de classification sur des données tabulaires. C'est le cas des Deep Neural Netwporks (Multilayer Perceptron) ou encore des TabTransformers. Nous aurions pu essayer ces modèles, mais les coûts en termes de temps de calcul sont plus élevés et demanderaient plus de moyens matériels.

#### Déséquilibre des classes

  Une autre approche aurait pu consister à limiter le changement du ratio (passage de 1:10 à 1:5 et non 1:1). Cela aurait permis de conserver plus de données et donc de perdre moins d'information. Nous aurions pu alors implémenter des poids lors de la validation croisée afin de traiter le déséquilibre (poids de 5 pour la classe minoritaire contre 1 pour la classe majoritaire).

***

## Analyse du Data Drift

#### Création du rapport de drift  

  Afin de produire le rapport de drift, nous avons utilisé la librairie evidently. Nous avons utilisé le jeu application_train comme jeu de référence et application_test comme jeu courant. Nous créé un rapport en utilisant le préréglage DatDriftPreset, afin de détecter la dérive sur les features. Le jeu courant (test) ne contenant pas la cible, nous l'avons écartée du jeu de référence afin d'avoir la même structure de données entre le jeu de référence et le jeu courant. Le rapport généré a été sauvegardé sous format html.

#### Présentation des résultats  

  L'analyse a détecté une dérive pour 7.44% des featurs (9 sur 121). Parmi elles, 4 sont à retenir car elles se retrouvent dans le top 100 de la feature importance intrinsèque du modèle :
  - AMT_ANNUITY : 6ème feature (score de gain = 74, 5ème position avec SHAP)
  - AMT_CREDIT : 13ème feature (score de gain = 53, 13ème position avec SHAP)
  - DAYS_LAST_PHONE_CHANGE : 14ème feature (score de gain = 51, 41ème position avec SHAP)
  - AMT_GOODS_PRICE : 17ème feature (score de gain = 46, 6ème position avec SHAP)

  En regardant de plus près la distribution des données pour les features concernées, on peut voir que le jeu courant ne comporte pas de valeurs extrêmes qui ne seraient pas présentes dans le jeu de référence. Cela limite les inquiétudes quand à l'inexploitabilité éventuelle du jeu test.