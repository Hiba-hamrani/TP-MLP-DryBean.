# Réponses aux questions – TP MLP avec TensorFlow/Keras (dataset Dry Bean)

Ce fichier accompagne le notebook `TP_MLP_DryBean.ipynb`. Les réponses s'appuient sur les résultats obtenus lors de l'exécution du notebook (seed = 42).

---

## Questions de lecture (mini-MLP XNOR)

- **Traversée des couches :** dans `forward`, la ligne `H = np.tanh(X @ self.W1 + self.b1)` correspond à la couche cachée, puis `Z = H @ self.W2 + self.b2` et `P` (sigmoïde) à la couche de sortie.
- **Comparaison aux cibles :** la ligne `losses.append(...)` calcule la perte entre les prédictions et les cibles `y`.
- **Transmission de l'erreur vers la couche cachée :** `D1 = (D2 @ self.W2.T) * (1-H**2)`.
- **Modification des poids :** les lignes `self.W1 -= rate*dW1`, `self.b1 -= rate*db1`, `self.W2 -= rate*dW2`, `self.b2 -= rate*db2`.
- **`forward` vs `fit` :** `forward` calcule seulement une prédiction ; `fit` répète forward, perte, rétropropagation et mise à jour pendant 2000 époques.

Les probabilités passent d'environ 0,5 au départ (réponse au hasard) à environ 0,999 / 0,003 / 0,003 / 0,997 après l'entraînement, ce qui correspond à la table de vérité du XNOR (1 / 0 / 0 / 1). La perte diminue jusqu'à presque 0.

---

## Q1 – Les données

Le jeu Dry Bean contient **13 611 observations** et **16 variables**. **68 doublons exacts** ont été supprimés avant la séparation, il reste donc **13 543 haricots**.

Une ligne de X décrit **un grain de haricot** par 16 mesures de forme (surface, périmètre, longueur des axes, rondeur, compacité, facteurs de forme…).

Il s'agit d'une **classification à 7 classes** (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA). Les classes sont des **catégories** : il n'existe aucun ordre ni aucune distance entre deux variétés, ce ne sont donc pas des valeurs continues.

Les classes sont **déséquilibrées** : 3 546 DERMASON contre seulement 522 BOMBAY.

---

## Q2 – Séparation et préparation

Les données sont séparées en **8 125 / 2 709 / 2 709** exemples (60 / 20 / 20).

- **Stratification :** elle conserve les proportions de chaque variété dans les trois ensembles, ce qui est important pour la classe rare BOMBAY. Sans elle, un ensemble pourrait contenir trop peu d'exemples d'une variété et les scores seraient faussés.
- **Scaler ajusté sur le train uniquement :** cela évite une fuite d'information du test vers l'entraînement. La normalisation est indispensable ici car les mesures ont des échelles très différentes (la surface atteint 254 616, alors que ShapeFactor2 vaut environ 0,002).
- **LabelEncoder :** les variétés sont du texte ; elles sont converties en indices 0 à 6, appris sur le train.
- **Validation vs test :** la validation sert à **choisir** les réglages du modèle ; le test sert à mesurer la performance **finale**, une seule fois, sur des données jamais utilisées pour faire des choix.

---

## Q3 – Architecture

Le réseau possède **une seule couche cachée** de 32 neurones ReLU.

Il a **7 sorties** car il y a 7 variétés : la fonction softmax donne une probabilité par variété.

Le résumé affiche **775 paramètres** :
- couche cachée : 16 × 32 poids + 32 biais = 544 ;
- couche de sortie : 32 × 7 poids + 7 biais = 231 ;
- total : 544 + 231 = 775.

Ce sont tous les poids et biais ajustés pendant l'entraînement.

---

## Q4 – Perte et accuracy

- La **perte** (entropie croisée) mesure l'écart entre les probabilités prédites et la vraie classe. C'est ce que le réseau minimise pendant l'entraînement. Elle tient compte de la confiance de la prédiction.
- L'**accuracy** est le pourcentage de bonnes réponses. Elle ne tient pas compte de la confiance.

Combinaisons sortie / perte :
- **classification binaire :** 1 neurone sigmoïde + entropie croisée binaire ;
- **régression non contrainte :** 1 neurone linéaire (sans activation) + erreur quadratique moyenne (MSE).

---

## Q5 – Paramètres de `fit`

- `epochs` : nombre de passages complets sur les données d'entraînement (ici 40).
- `batch_size` : nombre d'exemples traités avant chaque mise à jour des poids (ici 32).
- `validation_data` : ensemble utilisé pour mesurer la performance à la fin de chaque époque.

Pour 8 125 observations et des lots de 32 : 8 125 / 32 ≈ 253,9, soit **254 lots par époque** (donc 254 mises à jour des poids par époque).

Les poids **ne s'ajustent pas** à partir de la validation : elle sert seulement à observer les performances.

---

## Q6 – Courbes d'apprentissage

Les pertes train et validation diminuent fortement pendant les premières époques, puis se stabilisent. En fin d'entraînement, la perte de validation vaut environ **0,17** et l'accuracy de validation environ **94 %**.

Les accuracies progressent rapidement, puis plafonnent. L'écart entre train et validation reste faible et la perte de validation ne remonte pas : il n'y a **pas de surapprentissage marqué**.

Une baisse de la perte train ne garantit pas pour autant une amélioration de la validation : la validation progresse de moins en moins alors que la perte train continue de diminuer.

---

## Q7 – Prédiction et évaluation

- Le tableau de probabilités a la forme **(2709, 7)** : une probabilité par variété pour chaque haricot de validation.
- **argmax plutôt qu'un seuil de 0,5 :** avec 7 classes, aucune probabilité ne dépasse forcément 0,5. On choisit donc la classe la plus probable. Le seuil de 0,5 sert en classification binaire.

Le modèle obtient **94,2 % d'accuracy** et un **F1 macro de 0,951** sur la validation.

- **BOMBAY** est parfaitement reconnue (précision et rappel de 1,000) : sa forme est très différente des autres variétés.
- **SIRA** est la classe la plus difficile (rappel de 0,880) : 12 % des vrais SIRA sont classés dans une autre variété (voir la matrice de confusion du notebook).
- **DERMASON** a la précision la plus faible (0,920) : certains haricots prédits DERMASON appartiennent en réalité à une autre variété.

La **précision** d'une classe mesure la fiabilité d'une prédiction (parmi les haricots prédits dans cette classe, combien le sont vraiment). Le **rappel** mesure la capacité à retrouver tous les exemples de la classe (parmi les vrais exemples, combien sont reconnus). Une case hors diagonale de la matrice de confusion indique une confusion entre la variété réelle (ligne) et la variété prédite (colonne).

---

## Q8 – Taux d'apprentissage

| Taux | CE validation | Accuracy | F1 macro |
|---|---|---|---|
| 0,0001 | 0,221 | 93,5 % | 0,945 |
| 0,001 | **0,174** | **94,2 %** | **0,950** |
| 0,05 | 0,240 | 92,1 % | 0,934 |

- Le taux **0,0001** avance le plus lentement. Il reste correct car chaque époque contient 254 mises à jour.
- Le taux **0,001** donne la meilleure perte de validation : c'est le meilleur compromis entre vitesse et stabilité.
- Le taux **0,05** ne diverge pas, mais il donne une perte de validation plus élevée et une courbe plus irrégulière : les pas sont trop grands pour se stabiliser près du minimum.

Un grand taux ne diverge donc pas forcément, mais il rend l'apprentissage moins stable.

---

## Q9 – Adam et SGD

Seul l'**optimiseur** passé à `compile` change (`keras.optimizers.Adam` ou `keras.optimizers.SGD`).

| Optimiseur | CE validation | Accuracy | F1 macro |
|---|---|---|---|
| Adam | 0,174 | 94,2 % | 0,950 |
| SGD | 0,457 | 89,6 % | 0,910 |

On **ne peut pas conclure** qu'Adam est toujours meilleur à partir de ce seul essai : un taux de 0,001 est trop petit pour SGD sans momentum. Pour une comparaison pertinente, il faudrait **régler le taux d'apprentissage** de chaque optimiseur séparément (et éventuellement ajouter un momentum à SGD).

---

## Q10 – Taille des lots

| Taille du lot | Mises à jour par époque | Accuracy | F1 macro |
|---|---|---|---|
| 1 | 8 125 | 93,9 % | 0,950 |
| 32 | 254 | 93,2 % | 0,942 |
| 8 125 (lot complet) | 1 | 2,3 % | 0,019 |

Avec le lot complet, le modèle n'effectue que 10 mises à jour au total en 10 époques : il n'a presque rien appris (CE de 2,15).

Le **même nombre d'époques ne signifie pas le même nombre de mises à jour** : une époque correspond à un passage sur toutes les données, mais le nombre de corrections des poids dépend de la taille du lot. C'est ce nombre de corrections qui fait progresser l'apprentissage. Le mini-lot de 32 offre le meilleur compromis entre vitesse de calcul et qualité.

---

## Q11 – Couches cachées et activations

| Configuration | Couches cachées | Paramètres | CE validation | Accuracy | F1 macro |
|---|---|---|---|---|---|
| Étroit | 2 (ReLU) | 55 | 0,367 | 91,0 % | 0,917 |
| Référence | 32 (ReLU) | 775 | 0,174 | 94,2 % | 0,950 |
| Deux couches | 64 puis 32 (ReLU) | 3 399 | 0,177 | 94,0 % | 0,951 |
| Tanh | 32 (tanh) | 775 | 0,176 | 94,2 % | 0,951 |

- Le réseau **étroit** est trop petit : il sous-apprend.
- Le réseau à **deux couches** a plus de quatre fois plus de paramètres que la référence, sans gain réel.
- **ReLU et tanh** donnent des résultats équivalents ici.

Un plus grand réseau **n'améliore donc pas toujours** la généralisation : au-delà d'une certaine taille, il n'apporte plus rien et peut même favoriser le surapprentissage.

---

## Q12 – Régularisation

| Option | Paramètres | CE validation (sans pénalité) | Perte `evaluate` | Accuracy | F1 macro |
|---|---|---|---|---|---|
| Référence | 3 399 | 0,177 | 0,177 | 94,1 % | 0,951 |
| L1 | 3 399 | 0,174 | 0,206 | 94,1 % | 0,952 |
| L2 | 3 399 | 0,174 | 0,204 | 94,1 % | 0,950 |
| Dropout | 3 399 | 0,180 | 0,180 | 94,1 % | 0,952 |
| BatchNorm | 3 783 | 0,199 | 0,199 | 92,9 % | 0,940 |

- Avec **L1 et L2**, `evaluate` affiche une perte plus élevée car elle **ajoute la pénalité sur les poids** à l'entropie croisée. C'est pourquoi on compare les modèles avec la CE sans pénalité.
- Le **Dropout** n'est actif que **pendant l'entraînement**. Le diagnostic le confirme : « Évaluation identique : True » et « Entraînement identique : False ».
- On utilise `predict` ou `evaluate` pour une prédiction finale car ils fonctionnent en **mode évaluation** : tous les neurones sont actifs et BatchNorm utilise ses statistiques mobiles, ce qui donne des prédictions stables.
- BatchNorm ajoute des paramètres (3 783 au lieu de 3 399).
- **Aucune option n'améliore systématiquement les résultats** : L1 et L2 apportent un gain très faible, le Dropout n'améliore pas la perte, et BatchNorm dégrade même les scores dans cette expérience.

---

## Q13 – Arrêt précoce

- **36 époques** ont été exécutées.
- La plus faible perte de validation se situe à l'**époque 28**.
- L'écart de 8 correspond à la **patience** : Keras attend 8 époques sans amélioration avant de s'arrêter, puis restaure les poids de la meilleure époque (`restore_best_weights=True`). Le modèle final obtient 94,1 % d'accuracy et une perte de 0,174.
- L'arrêt **n'est pas garanti** avant le maximum : si la perte de validation continuait de baisser, même légèrement, au moins une fois toutes les 8 époques, l'entraînement irait jusqu'à 100 époques.

---

## Q14 – Sauvegarde et utilisation dans une application

- Le fichier `modele_drybean.keras` contient le réseau et ses poids, mais **ni le StandardScaler ni le LabelEncoder**. Si l'on donne au modèle des mesures brutes (par exemple une surface de 50 000), il reçoit des valeurs à une échelle qu'il n'a jamais vue et ses prédictions sont fausses.
- **Avant `predict`**, il faut appliquer `scaler.transform(...)` avec **le même scaler** ajusté sur le train, sans jamais refaire un `fit` sur les nouvelles données. Il faut aussi conserver le **LabelEncoder** (ou la correspondance numéro → variété) pour retraduire l'indice prédit en nom de variété.
- **`fit`** entraîne le modèle : il modifie les poids par rétropropagation. **`predict`** utilise le modèle sans modifier les poids (et le Dropout y est désactivé).
