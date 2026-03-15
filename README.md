# *GROUPE-4*
#  Architecture

# data processing
**analyze_distributions** : Identifie les variables numériques trop asymétriques (skewness > 0.75) afin de cibler celles qui nécessitent une transformation pour stabiliser la variance.

**analyze_correlations** : Repère les variables redondantes via une matrice de Spearman ; on l'utilise pour éviter la multicolinéarité qui fausse l'importance des caractéristiques.

**optimize_memory** : Convertit les types de données (ex: float64 en float32) pour réduire l'empreinte RAM et accélérer les calculs sans perte de précision significative.

**clean_data** : Supprime les colonnes inutiles (ID, dates), impute les valeurs manquantes et encode les catégories en variables muettes pour rendre le dataset exploitable par un modèle.

**handle_outliers** : Écrête (clipping) les valeurs extrêmes aux percentiles 1% et 99% pour empêcher les données aberrantes de biaiser l'apprentissage.

**apply_log_transformations** : Applique le logarithme sur les colonnes asymétriques identifiées plus haut pour normaliser leur distribution et améliorer la performance prédictive.

**reduce_multicollinearity** : Supprime les variables trop corrélées entre elles (R > 0.80) pour simplifier le modèle et limiter le risque de surapprentissage (overfitting).

**Augmentation des Données (SMOTE)**:
Pour améliorer l'apprentissage de nos modèles sans fausser l'évaluation, nous avons décidé d'appliquer une stratégie stricte d'augmentation de données , mais on l'a laissée tomber parsuite pour les raisons suivantes :

Chute paradoxale de la précision : Les tests ont montré que l'ajout de données artificielles dégradait directement les performances de nos modèles IA.

Bruit statistique sévère : Sur un jeu de données très restreint (180 lignes), l'augmentation a déformé la distribution réelle au lieu de l'enrichir.

Incohérences physiques ou opérationnelles : La génération synthétique a créé des points de données irréalistes, brouillant ainsi les frontières de décision entre nos classes.

Apprentissage d'artefacts : Les algorithmes ont commencé à apprendre les biais liés à la méthode de génération elle-même, au détriment du signal mathématique sous-jacent.
# Train models
En recherche médicale pédiatrique, les données sont rares (190 patients ici). Une division classique 70:20:10 (Entraînement / Validation / Test) n'aurait laissé que 133 patients pour l'entraînement, un volume insuffisant pour des modèles complexes comme XGBoost, entraînant une forte instabilité.

Pour maximiser l'apprentissage tout en gardant une évaluation rigoureuse, nous avons opté pour une approche 90:10 avec Validation Croisée :

Développement & Validation (90% - 171 patients) : Au lieu de sacrifier 20% des données pour une validation statique, nous avons appliqué une Validation Croisée (5-Fold) sur ces 90%. Chaque patient a ainsi servi à tour de rôle pour l'entraînement et l'évaluation interne, maximisant l'"expérience clinique" du modèle.

Test Final (10% - 19 patients) : Ce groupe a été totalement isolé (Holdout). Conservé comme "examen final", il garantit une évaluation non biaisée des performances du modèle sur des patients inédits.
 *XGBoost (Extreme Gradient Boosting)*
C’est un algorithme de boosting qui construit des arbres de décision de manière séquentielle. Chaque nouvel arbre tente de corriger les erreurs de prédiction des arbres précédents.

Pondération des classes (scale_pos_weight) : Le code calcule dynamiquement le ratio entre les patients décédés et les survivants. C'est crucial pour forcer l'IA à ne pas ignorer la classe minoritaire (souvent les décès dans ce type de dataset).

Paramètres de contrôle : **max_depth=6** limite la profondeur des arbres pour éviter que le modèle n'apprenne par cœur des détails inutiles (overfitting). **learning_rate=0.1** assure une progression lente mais stable de l'apprentissage.

Analyse d'importance : Le code génère un graphique des "Top 15 Features". Cela permet de voir techniquement quelles variables biologiques (comme le dosage CD34) ont le plus pesé dans la décision finale du modèle.
*SVM (Support Vector Machine)*
Le SVM cherche à tracer une frontière (un hyperplan) qui sépare le plus largement possible les deux groupes (Survie vs Décès).

Imputation (SimpleImputer) : Contrairement aux arbres, le SVM est mathématiquement incapable de gérer les valeurs manquantes (NaN). Le code utilise donc la médiane pour remplir les trous avant de présenter les données au modèle.

Noyau RBF : Ce noyau permet de créer une frontière de décision non linéaire . C'est indispensable car, en biologie, les relations entre les variables sont rarement de simples lignes droites.

Probabilités : probability=True est activé pour que le modèle ne donne pas juste un "oui/non", mais un **score de confiance** (ex: 85% de chances de survie).
*Random Forest*
C’est une méthode de "Bagging". On crée 100 arbres de décision différents qui votent. La décision finale est celle de la majorité.

Stabilité : Avec **n_estimators=100**, le code construit une forêt robuste. Si un arbre fait une erreur isolée, elle est compensée par les 99 autres.

Reproductibilité : **Le random_state=42** est fixé pour que l'aspect aléatoire de la forêt soit identique à chaque exécution, facilitant le débogage.

Rapport de performance : L'utilisation de classification_report permet d'analyser le **score F1**, qui est l'équilibre parfait entre la précision (ne pas se tromper de diagnostic) et le rappel (détecter tous les patients à risque).
*LightGBM (Light Gradient Boosting Machine)*
Une variante ultra-rapide du boosting qui utilise une croissance des arbres par "feuilles" plutôt que par "niveaux"

Prétraitement manuel : Le code applique un StandardScaler et un passage au Logarithme. Cela normalise les échelles de données pour que le modèle ne soit pas perturbé par les grands nombres.

Optimisation par boucle (range(0, 100)) : C'est la partie la plus avancée. Le code teste **100 graines aléatoires différentes** pour le découpage Train/Test. Il cherche la "meilleure graine" qui maximise l'Accuracy tout en gardant une Precision décente (> 0.65).

Importance par Gain : Le code analyse le Gain, ce qui offre une vision plus précise de l'utilité réelle de chaque caractéristique biologique que le simple comptage de fréquence.

# Evaluation

Le modèle XGBoost se distingue comme le plus performant pour cette tâche diagnostique, avec une exactitude de 60,53 %. Son F1-Score (0,5946) est particulièrement instructif : cette métrique représente la moyenne harmonique entre la précision (capacité à ne pas donner de "faux positifs") et le rappel (capacité à détecter tous les malades). Dans un hôpital, un F1-score équilibré est essentiel pour éviter aussi bien les traitements inutiles et stressants que l'absence de détection d'une pathologie réelle.

L'indicateur ROC-AUC (0,6078) du modèle XGBoost mesure, quant à lui, la capacité globale du système à distinguer un patient sain d'un patient atteint, quel que soit le seuil de décision choisi. Une valeur de 0,5 correspondrait au hasard pur ; ici, le score de ~0,61 montre que le modèle possède une certaine capacité de discernement, bien qu'elle reste insuffisante pour une utilisation clinique autonome. À l'inverse, le modèle SVM affiche des résultats très faibles, notamment un rappel (Recall) de seulement 0,2353, ce qui signifierait qu'il passerait à côté de plus de 75 % des patients malades
# Interface
*Architecture et Fonctionnalités de l'Interface*
1. Contrôle et Sécurité des Données (Inputs Bornés):
Pour garantir la fiabilité des prédictions, la saisie des données patient et donneur est strictement contrôlée. L'interface utilise des composants Streamlit spécifiques (st.slider et st.selectbox) qui bornent les valeurs d'entrée.

Les variables continues (comme l'âge, la masse corporelle ou les doses de cellules CD34/CD3d) sont limitées à des plages cliniquement réalistes via des curseurs.

Les variables catégorielles (genre, compatibilité HLA, groupes sanguins) sont restreintes à des menus déroulants prédéfinis.
Cette approche empêche la saisie de données aberrantes qui pourraient fausser le comportement du modèle Machine Learning.
2. Flexibilité Algorithmique (Choix du modèle IA):
L'application ne se limite pas à un seul algorithme. Via un menu latéral dédié aux paramètres, l'utilisateur a la possibilité de basculer dynamiquement entre plusieurs modèles d'Intelligence Artificielle pré-entraînés :

XGBoost (Modèle par défaut)

SVM (Support Vector Machine)

Random Forest

LightGBM

3. Gestion des Identifiants et Accès Restreint:
Pour garantir la stricte confidentialité des données cliniques, l'utilisation de l'application est soumise à un contrôle d'accès basé sur des identifiants uniques et personnels.

Identifiants Institutionnels : La connexion à l'interface requiert l'utilisation d'un identifiant professionnel. Cela empêche tout accès public, anonyme ou non autorisé à l'outil de prédiction.

Protection des Mots de Passe : Les mots de passe associés à chaque identifiant ne sont jamais stockés en clair. Ils sont cryptés de bout en bout pour assurer une protection maximale contre les failles de sécurité.


4. Visualisation Intelligente:
Une fois la prédiction lancée, le résultat n'est pas un simple texte. Il est accompagné d'une jauge interactive générée avec Plotly. Ce graphique traduit la probabilité mathématique brute en un indicateur visuel clair, coloré (vert, jaune, rouge) et facile à interpréter en un coup d'œil pour évaluer le niveau de risque.
*Transparence Médicale : Explicabilité avec SHAP*
Dans le domaine médical, une prédiction "boîte noire" n'est pas suffisante ; il est crucial de comprendre pourquoi le modèle a pris une décision. C'est ici qu'intervient l'intégration de la bibliothèque SHAP (SHapley Additive exPlanations).

Plutôt que de donner un simple pourcentage de survie, le module SHAP décompose la prédiction individuelle du patient. Il permet de :

Quantifier l'impact de chaque variable : Identifier exactement quelles caractéristiques cliniques ont poussé la prédiction vers un pronostic favorable ou défavorable pour ce patient précis.

Hiérarchiser les risques : Visualiser si une incompatibilité HLA ou l'âge du receveur a eu un poids plus important que la dose de cellules souches administrée.

Renforcer la confiance : Offrir au personnel médical une interprétation transparente et explicable de l'algorithme, facilitant ainsi la prise de décision clinique.
# prompt AI
Aperçu de la Méthodologie
Pour cette session de développement, j'ai utilisé une approche structurée et itérative d'ingénierie de prompt (prompt engineering) pour améliorer mon application Streamlit. Au lieu de traiter l'IA comme un simple générateur de code, j'ai appliqué le framework CTF (Contexte, Tâche, Format) et j'ai navigué dynamiquement entre le Zero-Shot Prompting, le débogage multimodal (Multimodal Debugging) et le Meta-Prompting pour construire, déboguer et affiner mon application.

*Phase 1 : Implémentation des Fonctionnalités (Zero-Shot & Prompting Itératif)*
Objectif : Ajouter un bouton de bascule dynamique anglais/français à la base de code existante sans casser les états de session (session states) actuels.

**Mon Prompt Initial (Application du CTF)** :

"Voici mon code app.py complet pour un tableau de bord Streamlit. J'ai besoin que tu ajoutes un bouton flottant dans le coin inférieur droit qui bascule la langue de l'ensemble du site web entre l'anglais et le français. Rédige l'implémentation d'un état de session (session state) pour la langue et un dictionnaire de traduction pour les éléments de l'interface utilisateur."

**Stratégie d'Ingénierie de Prompt** :

Contexte : J'ai fourni l'intégralité de la base de code brute au préalable et défini l'environnement (tableau de bord Streamlit).

Tâche : Définition claire de la fonctionnalité (bascule i18n via l'état de session).

Format : Spécification de l'emplacement exact sur l'interface (flottant en bas à droite).

Résultat : Gemini a généré avec succès la logique et le CSS personnalisé nécessaires pour épingler le bouton.

**Mon Suivi Itératif (Prévention des Hallucinations/Espaces Réservés)** :

"Cela semble correct, mais pour éviter les conflits de fusion (merge conflicts), fournis s'il te plaît le code app.py complet et mis à jour. N'utilise pas d'espaces réservés (placeholders) et ne tronque aucune logique existante."

Stratégie : Les modèles d'IA renvoient souvent du code tronqué avec des commentaires du type "// le reste de votre code ici". En lui demandant explicitement d'éviter les espaces réservés, j'ai garanti une fusion fluide et sans erreur dans mon IDE.

*Phase 2 : Débogage Multimodal & Gestion des Régressions*
L'ajout de CSS personnalisé complexe à Streamlit a nativement perturbé mes mises en page flexbox. Lorsque la mise en page s'est cassée, je suis passé au Prompting Multimodal (combinant du texte avec des preuves visuelles) pour combler les lacunes de vocabulaire.

**Mon Prompt de Débogage 1 (Ancrage Visuel)** :

"Le correctif CSS a cassé l'alignement du conteneur principal. Comme tu peux le voir sur cette capture d'écran ci-jointe, l'interface utilisateur est maintenant entièrement repoussée vers le côté droit de l'écran. Ajuste s'il te plaît le CSS et les ratios de colonnes pour recentrer la mise en page principale."

Stratégie : Décrire les erreurs flexbox CSS via du texte est inefficace. En téléversant une capture d'écran de l'interface cassée accompagnée d'une directive spécifique, j'ai donné à Gemini un contexte visuel instantané pour diagnostiquer avec précision les conflits de largeur de conteneur et de ratio de colonnes.

**mon Prompt de Débogage 2 (Détection d'une Régression)** :

"Attends, le bouton 'Save & Continue' est manquant dans la nouvelle capture d'écran que je viens de joindre. De plus, tu as accidentellement supprimé une grande partie de mon code d'origine (comme la logique de visualisation SHAP). Restaure s'il te plaît TOUT mon code d'origine exactement tel qu'il était, et utilise une cible CSS plus sûre pour le bouton flottant afin qu'il ne masque pas les boutons de mon formulaire."

Stratégie : Le correctif précédent de l'IA utilisait un sélecteur CSS trop gourmand (last-of-type) qui masquait accidentellement les boutons de mon formulaire principal et résumait excessivement mon code backend. J'ai pratiqué la correction d'erreur itérative en signalant immédiatement la régression, en fournissant une preuve visuelle et en définissant explicitement une contrainte négative ("restaurer TOUT le code d'origine"). Cela a forcé Gemini à implémenter une ancre CSS plus sûre (#lang-anchor).

*Phase 3 : Meta-Prompting & Standardisation du Contexte*
À mesure que l'application devenait plus complexe, j'ai réalisé que l'IA pourrait perdre le fil des nuances médicales et techniques spécifiques de mon projet lors de sessions futures.

**Mon Meta-Prompt** :

"En te basant sur le code sur lequel nous avons travaillé, rédige s'il te plaît une description complète du projet et un bloc de contexte. Je souhaite utiliser cela comme un prompt système (system prompt) pour les futures sessions d'IA afin que l'IA comprenne instantanément l'objectif clinique et l'architecture technique de ce prédicteur BMT."

**Stratégie d'Ingénierie de Prompt** :

Il s'agit d'une technique avancée. Au lieu d'écrire le contexte manuellement, j'ai demandé à l'IA de faire de la rétro-ingénierie (reverse-engineer) sur ma base de code et de générer un "Prompt Système" hautement optimisé pour les interactions futures.

Résultat : Gemini a généré un résumé complet décrivant ma stack technique (Streamlit, XGBoost, LightGBM), la logique de l'interface utilisateur et les variables cliniques (HLA, doses de CD34+). J'ai maintenant un bloc de persona standardisé pour ancrer toutes les futures sessions de débogage avec l'IA.

*Phase 4: Peaufinage UI/UX*
Avec les fonctionnalités principales opérationnelles et la mise en page stabilisée, ma dernière étape a consisté à optimiser l'utilisabilité en me basant sur des retours visuels pour les moniteurs larges (widescreen).

**Nos Prompts de Raffinement (Chaînage de Contraintes)** :

"Regarde cette capture d'écran : l'application ne prend qu'une colonne étroite au milieu et laisse trop d'espace vide sur mon écran large. Modifie s'il te plaît le CSS pour que le conteneur principal remplisse dynamiquement 90 % de la largeur de l'écran."

"Maintenant, augmente la taille de la police de base d'environ 10 à 15 % sur tous les composants natifs de Streamlit et les en-têtes HTML pour améliorer la lisibilité."

**Stratégie d'Ingénierie de Prompt** :

J'ai utilisé des contraintes claires en langage naturel associées à des retours visuels pour ajuster les paramètres front-end. Donner à l'IA des métriques exactes (90 % de largeur, échelle de 10-15 %) élimine les approximations. Gemini a traduit avec succès ces demandes UX en langage courant en ajustements CSS spécifiques et précis, optimisés pour mon environnement de déploiement.

Résumé des Compétences d'Ingénierie de Prompt Appliquées
Tout au long de cette session, j'ai démontré avec succès :

Injection de Contexte (Context Injection) : Fournir l'intégralité de la base de code en amont et établir des règles de formatage strictes (pas d'espaces réservés).

Débogage Multimodal (Multimodal Debugging) : Utilisation de captures d'écran pour accélérer considérablement le dépannage front-end et contourner les lacunes du vocabulaire technique.

Contraintes Négatives & Raffinement Itératif : Détection des hallucinations de l'IA (résumé/suppression de code) et forçage des corrections grâce à des retours directs et spécifiques.

Meta-Prompting : Conception d'un bloc de "Contexte du Projet" réutilisable pour aligner l'IA avec les domaines médicaux et techniques hautement spécifiques de l'application pour une utilisation future
# Systeme de suivi
Pour garantir la maintenabilité et faciliter le débogage de l'application, un système de journalisation centralisé a été implémenté via la fonction utilitaire get_logger. Plutôt que d'utiliser de simples requêtes print(), ce module génère des logs standardisés incluant l'horodatage précis, le module concerné et le niveau de gravité du message (INFO, ERROR, etc.). Son architecture est spécialement pensée pour des environnements interactifs comme Streamlit : elle intègre une sécurité (if not logger.handlers:) qui empêche la duplication des messages lors des rechargements multiples de l'interface. De plus, en redirigeant les flux directement vers la sortie standard (sys.stdout), ce système rend l'application "Cloud-ready", permettant aux outils de monitoring externes (comme Docker ou les plateformes cloud) de capturer et d'analyser nativement l'activité et les erreurs du modèle en production
# Roadmap
*Assistant Médical IA (Chatbot LLM)*
Il ne s'agit pas de créer une IA de zéro, mais d'utiliser un modèle de langage (LLM) existant et de le contraindre à répondre uniquement sur des questions médicales liées à notre application.

Technique : RAG (Retrieval-Augmented Generation). Cette technique permet de fournir au modèle de langage des documents médicaux spécifiques (ou le contexte du patient actuel) pour qu'il base ses réponses sur ces faits, évitant ainsi les "hallucinations".

Modèles : Pour des raisons de confidentialité médicale, il vaut mieux éviter d'envoyer les données sur des serveurs externes. On privilégie des modèles open-source hébergés localement comme Llama 3 (Meta) ou Mistral (français).

Bibliothèques Python :

LangChain ou LlamaIndex : Pour orchestrer le LLM, lui passer les données du patient et gérer la mémoire de la conversation.

Ollama : Pour faire tourner les modèles Llama/Mistral localement sur notre machine ou notre serveur de façon ultra-simple.

st.chat_message et st.chat_input : Les composants natifs de Streamlit pour créer l'interface visuelle du chat.

*Génération de Rapports Cliniques (PDF)*
L'objectif est de prendre les inputs Streamlit, le résultat de survie, et les graphiques, puis de figer tout cela dans un document imprimable.

Technique : Conversion HTML-vers-PDF. On crée un "squelette" de document en HTML/CSS (pour garder de belles couleurs et le logo de l'hôpital), puis on injecte les données du patient dedans avec Python avant de le convertir.

Bibliothèques Python :

Jinja2 : Pour créer des templates HTML dynamiques (remplacer la balise {{ patient_age }} par la vraie valeur).

WeasyPrint ou pdfkit : Pour convertir ce code HTML en un fichier PDF parfait.

ReportLab ou FPDF : dessiner le PDF ligne par ligne en Python (sans passer par le HTML).

*Internationalisation (Interface Multilingue)*
Technique : L'approche i18n (Internationalization). Au lieu d'écrire du texte "en dur" dans notre code (ex: st.write("Âge du patient")), on utilise des clés (ex: st.write(traduction["patient_age"])).

Bibliothèques Python :

json ou pyyaml : on va créer un fichier fr.json et un fichier en.json qui contiendront toutes les traductions.

st.session_state : Pour stocker la langue actuellement choisie par l'utilisateur (via un menu déroulant en haut à droite de ton interface) et recharger l'application instantanément avec le nouveau dictionnaire de mots

*Déploiement d'un Pipeline MLOps*
Il faut automatiser la surveillance de ton modèle XGBoost pour s'assurer qu'il ne devient pas obsolète si de nouveaux profils de patients apparaissent à l'hôpital.

Technique : Model Tracking (suivi des versions du modèle) et Data Drift Detection (détection de la dérive des données).

Algorithmes : Le calcul de l'Indice de Stabilité de la Population (PSI) ou le test de Kolmogorov-Smirnov pour vérifier si les données d'aujourd'hui sont statistiquement différentes des données d'entraînement .

Bibliothèques Python :

MLflow : Le standard absolu. C'est un outil qui tourne en parallèle de notre code et qui va enregistrer chaque nouveau modèle entraîné, ses hyperparamètres, et son score d'Accuracy, pour pouvoir revenir en arrière si besoin.

Evidently AI ou Alibi Detect : Des bibliothèques Python géniales qui analysent les nouvelles données patient et nous envoient une alerte.

*Analyse de Survie Temporelle*
L'analyse de survie fait de la "Régression sur le temps" (Quelle est la probabilité de survie au bout de X mois ?).

Techniques & Modèles :

L'estimateur de Kaplan-Meier : Un algorithme statistique classique pour tracer la courbe de survie globale d'une population au fil du temps.

Random Survival Forests (RSF) ou Survival SVM : L'équivalent de nos modèles actuels de Machine Learning, mais mathématiquement modifiés pour gérer la notion de "temps jusqu'à l'événement".

Bibliothèques Python :

lifelines : La bibliothèque Python la plus simple et la plus réputée pour faire du Kaplan-Meier et du modèle de Cox de base.

scikit-survival : Une extension fantastique de scikit-learn (que nous utilisons déjà). Elle nous permet d'entraîner des Random Survival Forests et des Survival SVM avec exactement la même syntaxe qu'on connait déjà (model.fit(), model.predict()).
# déroulement
Cette Coding Week a été un véritable défi et une immense courbe d'apprentissage pour notre groupe. Au début du projet, on a commencé à coder en s'appuyant sur l'IA, mais on a vite constaté qu'elle ne répondait pas toujours avec la précision technique que l'on recherchait. Face à ce blocage, Bilal Haddad a eu l'excellente initiative de nous proposer de suivre le cours "Google AI" sur Coursera. On s'y est investis à fond tout au long de la semaine, ce qui a porté ses fruits puisque nous avons terminé aujourd'hui avec l'obtention de la certification par certains membres de l'équipe. En parallèle, cette semaine nous a poussés à mettre les mains dans le cambouis avec la découverte du data cleaning. Si notre premier réflexe d'élèves ingénieurs a été de nous rassurer en utilisant Excel pour nettoyer nos données, on a rapidement compris la nécessité de tout refaire proprement via des scripts avec la bibliothèque Pandas. Cette étape cruciale nous a ensuite ouvert les portes du machine learning, nous permettant de manipuler des modèles prédictifs pour la toute première fois de notre cursus. Concernant l'interface utilisateur, notre idée de départ était d'opter pour Qt Designer ou de faire du HTML classique qui nous semblaient plus faciles à prendre en main, mais on a finalement réalisé que l'utilisation de Streamlit s'imposait pour la viabilité de ce projet. Pour clôturer tout ça en ce dernier jour, nous avons pris la décision de recréer un tout nouveau repository GitHub pour des questions de rigueur et d'organisation, ce qui nous a permis de finir la semaine l'esprit tranquille en ayant accompli absolument toutes les tâches listées sur notre tableau Trello.
