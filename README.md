# 🏞️ Scene Classification avec Deep Learning

Classification de scènes naturelles et urbaines utilisant des CNNs et le transfer learning avec PyTorch.

## 📊 Dataset

**Source**: [Scene Classification - Kaggle](https://www.kaggle.com/datasets/nitishabharathi/scene-classification)

- **6 classes**: Buildings, Forest, Glacier, Mountain, Sea, Street
- **Distribution**: Dataset équilibré (~2400 images par classe)
- **Dimensions**: Images de tailles variables (redimensionnées à 120x120 pour l'entraînement)
- **Split**: Train/Validation

## 🔬 Notebooks

### 1️⃣ Data Exploration
- Distribution des classes (dataset équilibré confirmé)
- Visualisation d'échantillons pour chaque classe
- Analyse des dimensions des images
- Statistiques descriptives

### 2️⃣ Baseline Model
**Architecture**: CNN simple avec 2 blocs convolutifs
```
Conv2D(3→24) → ReLU → MaxPool → Conv2D(24→8) → ReLU → AvgPool
→ Flatten → Linear(4232→220) → GELU → Linear(220→6)
```

**Résultats**:
- ✅ **Meilleure Val Accuracy**: 81.75% (Epoch 1)
- ⚠️ **Problème**: Overfitting sévère
  - Train Loss: 0.1478 → Val Loss: 0.9895 (Epoch 20)
  - Train Acc: 94.97% → Val Acc: 80.87%
- **Conclusion**: Architecture trop simple, manque de régularisation

### 3️⃣ Deep CNN avec RBF Activation
**Architecture**: 4 blocs convolutifs + MLP avec activation RBF

**Blocs Convolutifs**:
```
Bloc 1: Conv(3→16) → BN → GELU → MaxPool
Bloc 2: Conv(16→24) → BN → GELU → MaxPool
Bloc 3: Conv(24→16) → BN → GELU → MaxPool
Bloc 4: Conv(16→8) → BN → GELU → MaxPool
```

**Classificateur MLP**:
```
Linear(392→128) → RBF → Linear(128→64) → ReLU → Linear(64→6)
```

**Augmentation de données**: 
- ✅ MixUp
- ✅ CutMix
- ✅ Random Horizontal Flip
- ✅ Random Rotation (±10°)
- ✅ Color Jitter

**Résultats**:
- ✅ **Best Val Loss**: 0.4214 (Epoch 18)
- ✅ **Best Val Accuracy**: 86.39% (Epoch 18)
- ✅ **Train Accuracy**: 80.26%
- ⏱️ **Temps d'entraînement**: 54 minutes (20 epochs)
- **Force**: Bon équilibre train/val, pas d'overfitting grâce à MixUp/CutMix
- **Innovation**: Utilisation de RBF (Radial Basis Function) comme activation

### 4️⃣ Transfer Learning - ResNet18
**Architecture**: ResNet18 pré-entraîné (ImageNet) avec fine-tuning complet

**Configuration**:
```python
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# Tous les paramètres dégelés
for param in model.parameters():
    param.requires_grad = True
# Adaptation de la tête de classification
model.fc = nn.Linear(512, 6)
```

**Hyperparamètres**:
- Learning Rate: 1e-4 (plus faible pour préserver les poids pré-entraînés)
- Patience: 4 epochs (Early Stopping)

**Résultats**:
- 🏆 **Best Val Accuracy**: 93.19% (Epoch 11)
- ⚠️ **Best Val Loss**: 0.5751 (Epoch 11)
- ✅ **Train Accuracy**: 91.31%
- ⏱️ **Temps d'entraînement**: 227 minutes (15 epochs avant early stopping)

**Observations**:
- ✅ **Meilleure accuracy** de tous les modèles (+6.8% vs Deep CNN)
- ⚠️ **Problème de Loss**: La validation loss ne diminue pas autant qu'attendu (0.5751 vs 0.4214 pour Deep CNN)
- **Hypothèse**: Le modèle apprend bien les classes (accuracy élevée) mais a moins confiance dans ses prédictions (loss plus élevée)
- **Temps**: ~4x plus long que le Deep CNN custom

## 📈 Comparaison des Modèles

| Modèle | Val Accuracy | Val Loss | Train Time | Overfitting |
|--------|--------------|----------|------------|-------------|
| **Baseline CNN** | 81.75% | 0.9895 | ~30 min | ❌ Sévère |
| **Deep CNN + RBF** | 86.39% | 0.4214 | 54 min | ✅ Minimal |
| **ResNet18** | **93.19%** | 0.5751 | 227 min | ⚠️ Léger |

## 🎯 Résultats Clés

### 🥇 Meilleur Modèle: ResNet18
- **+11.44%** vs Baseline
- **+6.80%** vs Deep CNN custom
- Bénéficie du transfer learning d'ImageNet

### 🏅 Meilleur Rapport Performance/Temps: Deep CNN
- 86.39% d'accuracy en seulement 54 minutes
- Loss la plus faible (0.4214)
- Pas d'overfitting grâce à MixUp/CutMix
- Architecture originale avec RBF activation

## 🛠️ Technologies

```
Python 3.10+
PyTorch 2.0+
torchvision
numpy
pandas
matplotlib
seaborn
tqdm
```

## 📦 Installation

```bash
# Cloner le repository
git clone https://github.com/votre-username/scene-classification.git
cd scene-classification

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le dataset depuis Kaggle
# Placer dans ./data/
```

## 🚀 Utilisation

```python
# Charger un modèle pré-entraîné
import torch
from models import CustomCNN  # ou ResNet18

model = CustomCNN(num_classes=6)
model.load_state_dict(torch.load('models/best_deep_cnn.pth'))
model.eval()

# Prédire sur une nouvelle image
# (voir notebooks pour exemple complet)
```

## 📊 Techniques Clés

### Data Augmentation
- **MixUp**: Interpolation linéaire entre paires d'images
- **CutMix**: Remplacement de régions rectangulaires entre images
- Transformations classiques (flip, rotation, color jitter)

### Régularisation
- Batch Normalization dans tous les blocs convolutifs
- Early Stopping (patience = 4-7 epochs)
- ReduceLROnPlateau scheduler
- Dropout implicite via MixUp/CutMix

### Innovation
- **RBF Activation**: Radial Basis Function comme couche d'activation non-linéaire
  ```python
  RBF(x) = exp(-γ(x - center)²)
  ```

## 🔍 Observations et Enseignements

1. **Transfer Learning est puissant**: +6.8% d'amélioration avec ResNet18
2. **MixUp/CutMix réduisent l'overfitting**: Écart train/val minimal
3. **RBF Activation fonctionne**: Alternative intéressante à ReLU/GELU
4. **Trade-off temps/performance**: Deep CNN custom = excellent compromis



---

⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !