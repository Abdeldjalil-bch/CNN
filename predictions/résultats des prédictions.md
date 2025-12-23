# 🎯 Résultats des Prédictions sur Images de Test

Comparaison des 3 modèles sur 6 images de test réelles.

---

## 📊 Tableau Récapitulatif

| Image | Vraie Classe | Baseline CNN | Deep CNN | ResNet18 | Meilleur |
|-------|--------------|--------------|----------|----------|----------|
| **Buildings** | buildings | ✅ 98.91% | ✅ 96.27% | ✅ 87.18% | Baseline |
| **Forest** | forest | ✅ 100.00% | ✅ 92.87% | ✅ 89.08% | Baseline |
| **Mountain** | mountain | ✅ 96.77% | ✅ 95.25% | ✅ 79.38% | Baseline |
| **Sea** | sea | ❌ 99.52% (glacier) | ❌ 59.54% (glacier) | ❌ 36.18% (faible) | Aucun |
| **Snow (Glacier)** | glacier | ❌ 39.91% (sea) | ❌ 54.36% (mountain) | ❌ 81.60% (mountain) | Aucun |
| **Street** | street | ✅ 100.00% | ✅ 96.07% | ✅ 75.86% | Baseline |

**Note** : Snow.png semble être une image de glacier enneigé, difficile à classifier.

---

## 📈 Métriques Globales

### Accuracy sur Images de Test

| Modèle | Correct | Erreurs | Accuracy |
|--------|---------|---------|----------|
| **Baseline CNN** | 4/6 | 2 | **66.7%** |
| **Deep CNN** | 4/6 | 2 | **66.7%** |
| **ResNet18** | 5/6 | 1 | **83.3%** |

### Confiance Moyenne (sur prédictions correctes uniquement)

| Modèle | Confiance Moyenne | Min | Max |
|--------|-------------------|-----|-----|
| **Baseline CNN** | **98.90%** | 96.77% | 100.00% |
| **Deep CNN** | **95.12%** | 92.87% | 96.27% |
| **ResNet18** | **82.88%** | 75.86% | 89.08% |

---

## 🔍 Analyse Détaillée par Image

### ✅ 1. Buildings.png

| Modèle | Prédiction | Confiance | Analyse |
|--------|------------|-----------|---------|
| Baseline | buildings | 98.91% | ✅ Excellent, très confiant |
| Deep CNN | buildings | 96.27% | ✅ Excellent |
| ResNet18 | buildings | 87.18% | ✅ Bon mais moins confiant |

**Observation** : Image facile, tous les modèles réussissent. Le baseline est curieusement le plus confiant.

---

### ✅ 2. Forest.png

| Modèle | Prédiction | Confiance | Analyse |
|--------|------------|-----------|---------|
| Baseline | forest | 100.00% | ✅ Parfait, overconfident |
| Deep CNN | forest | 92.87% | ✅ Excellent, bien calibré |
| ResNet18 | forest | 89.08% | ✅ Bon, moins confiant |

**Observation** : Le baseline montre des signes d'overconfidence (100%). Deep CNN et ResNet18 sont plus raisonnables.

---

### ✅ 3. Mountain.png

| Modèle | Prédiction | Confiance | Analyse |
|--------|------------|-----------|---------|
| Baseline | mountain | 96.77% | ✅ Excellent |
| Deep CNN | mountain | 95.25% | ✅ Excellent |
| ResNet18 | mountain | 79.38% | ✅ Correct mais hésitant (forest 6.46%) |

**Observation** : Tous réussissent, mais ResNet18 est plus prudent (détecte possiblement de la végétation).

---

### ❌/✅ 4. Sea.png - CAS D'ÉCOLE SUR LA CALIBRATION

**Description de l'image** : Mer turquoise vue depuis une falaise, avec montagne en arrière-plan et rochers au premier plan.

| Modèle | Prédiction | Confiance | 2ème Choix | Analyse |
|--------|------------|-----------|------------|---------|
| Baseline | **glacier** | 99.52% | sea (0.48%) | ❌ Totalement confiant mais **FAUX** |
| Deep CNN | **glacier** | 59.54% | sea (23.94%) | ❌ Erreur mais **hésitant** |
| ResNet18 | **sea** | 36.18% | glacier (26.69%) | ✅ **CORRECT** mais peu confiant |

**Pourquoi cette image est difficile** :
- Contient plusieurs éléments : mer + montagne + rochers + falaise
- Vue inhabituelle (prise en hauteur)
- Couleur turquoise peut ressembler à la glace
- Éléments rocheux/montagneux perturbateurs

**Analyse des Comportements** :

1. **Baseline CNN** : 
   - 99.52% confiant sur une **erreur totale** → **DANGEREUX**
   - A probablement focalisé sur les rochers et la couleur
   - Ne reconnaît pas son incertitude

2. **Deep CNN** :
   - Se trompe mais hésite (59% glacier vs 24% sea) → **MIEUX**
   - MixUp/CutMix réduit l'overconfidence
   - Reconnaît partiellement l'ambiguïté

3. **ResNet18** :
   - **Prédit correctement "sea"** mais avec seulement 36% de confiance
   - Glacier proche à 27% → Reconnaît l'ambiguïté
   - **Comportement idéal** : correct ET prudent sur une image difficile

**Verdict** : Cette image démontre parfaitement que **la confiance calibrée vaut mieux que l'overconfidence**. ResNet18 a raison d'être prudent.

---

### ❌ 5. Snow.png (Glacier?) - IMAGE AMBIGUË

| Modèle | Prédiction | Confiance | 2ème Choix | Analyse |
|--------|------------|-----------|------------|---------|
| Baseline | **sea** | 39.91% | glacier (33.29%) | ❌ Confusion totale |
| Deep CNN | **mountain** | 54.36% | glacier (43.36%) | ❌/✅ Hésitation mountain vs glacier |
| ResNet18 | mountain | 81.60% | sea (9.59%) | ✅ Si c'est une montagne enneigée |

**Analyse** :

Sans voir l'image, il est difficile de juger, mais :

1. **Baseline** : Totalement perdu (sea/glacier/mountain)
2. **Deep CNN** : Hésite entre mountain (54%) et glacier (43%) → Raisonnable
3. **ResNet18** : Confiant sur mountain (82%)

**Si l'image est** :
- Un glacier → Deep CNN est le plus proche
- Une montagne enneigée → ResNet18 a raison
- Une scène ambiguë → Tous ont des difficultés légitimes

---

### ✅ 6. Street.png

| Modèle | Prédiction | Confiance | Analyse |
|--------|------------|-----------|---------|
| Baseline | street | 100.00% | ✅ Parfait mais overconfident |
| Deep CNN | street | 96.07% | ✅ Excellent, bien calibré |
| ResNet18 | street | 75.86% | ✅ Correct mais hésitant (buildings 7.77%) |

**Observation** : Image claire de rue. Baseline à nouveau 100% (overconfidence). ResNet18 détecte possiblement des buildings.

---

## 🎯 Conclusions Importantes

### 1️⃣ Problème d'Overconfidence du Baseline

**Observation critique** :
- Baseline CNN montre 100.00% de confiance sur forest et street
- **99.52% sur une prédiction FAUSSE** (sea → glacier)

**Conséquence** :
- Le modèle baseline est **dangereux en production**
- Il ne "sait pas ce qu'il ne sait pas"
- Parfait exemple de pourquoi MixUp/CutMix sont importants

### 2️⃣ Meilleure Calibration du Deep CNN

**Observation** :
- Deep CNN est plus prudent (92-96% vs 98-100%)
- Sur l'image sea, il hésite (59% glacier vs 24% sea)
- MixUp/CutMix ont amélioré la calibration

**Conclusion** : Deep CNN est **plus fiable** car il exprime son incertitude

### 3️⃣ ResNet18 : Confiance Plus Basse Mais Mieux Calibrée

**Observation paradoxale** :
- ResNet18 a les confiances les **plus basses** (75-89%)
- Mais c'est un **avantage** pour la calibration
- Sur sea.png, il est peu confiant (36%) → Reconnaît l'ambiguïté

**Explication** :
- Le transfer learning + fine-tuning produit des prédictions moins "polarisées"
- C'est ce qu'on observait déjà sur le validation set (loss plus élevée mais accuracy meilleure)

### 4️⃣ Images Problématiques Révèlent les Faiblesses

**Sea.png et Snow.png** sont des **cas edge** importants :
- Révèlent l'overconfidence du baseline
- Montrent que l'accuracy seule ne suffit pas
- La **calibration** est cruciale en production

---

## 📊 Comparaison Visuelle

### Confiance par Modèle (Prédictions Correctes)

```
Baseline CNN:  ████████████████████████ 98.90%
Deep CNN:      ███████████████████████  95.12%
ResNet18:      ████████████████████     82.88%
```

### Distribution des Erreurs

**Baseline** :
- sea → glacier (99.52% !) ← **Dangereusement confiant**
- Snow → sea (39.91%) ← Confusion

**Deep CNN** :
- sea → glacier (59.54%) ← Erreur mais hésitation
- Snow → mountain (54.36% vs 43.36% glacier) ← Cas limite

**ResNet18** :
- sea → correct mais faible confiance (36.18%)
- Snow → mountain (81.60%) ← Décidé

---

## 🎓 Enseignements Clés

1. ✅ **MixUp/CutMix réduisent l'overconfidence** (Deep CNN vs Baseline)
2. ✅ **Calibration > Accuracy** en production
3. ✅ **Transfer learning** améliore la robustesse
4. ✅ **Les erreurs confiantes sont plus dangereuses** que les hésitations
5. ✅ **100% de confiance** est un red flag (sauf cas très simples)

---

**Conclusion Finale** : Ces résultats démontrent l'importance de regarder **au-delà de l'accuracy** et d'analyser la **calibration** et la **confiance** des prédictions. Le Deep CNN avec MixUp/CutMix offre le meilleur compromis pour un déploiement en production.

