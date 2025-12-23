# Intel Scene Classification - Classification de Scènes Naturelles

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

Projet de classification d'images de scènes naturelles utilisant le dataset **Intel Image Classification** (6 classes : buildings, forest, glacier, mountain, sea, street).

Ce repository compare **3 modèles CNN** entraînés from scratch et avec fine-tuning :
- Modèle baseline léger
- Modèle CNN profond personnalisé (avec RBF activation)
- ResNet18 pré-entraîné (fine-tuning)

## 🎯 Résultats sur Validation (Meilleure Accuracy)

| Modèle                  | Input Size | Meilleure Val Accuracy | Notes |
|-------------------------|------------|-------------------------|-------|
| Baseline CNN            | 100x100    | ~80-82%                | Modèle simple, rapide |
| Deep CNN + RBF          | 120x120    | **86.39%**             | Meilleur modèle from scratch |
| ResNet18 (fine-tuning)  | 224x224    | **86.33%** (après phase 1) | Potentiel >92% avec full fine-tune |

## 🧠 Modèles Testés sur Images Exemples

Test sur 6 images représentatives (une par classe) :

| Image          | Baseline CNN | Deep CNN + RBF | ResNet18 Fine-tune |
|----------------|--------------|----------------|---------------------|
| Buildings.png  | 98.91% ✅    | 96.27% ✅      | 87.18% ✅           |
| forest.png     | 100.00% ✅   | 92.87% ✅      | 89.08% ✅           |
| mountain.png   | 96.77% ✅    | 95.25% ✅      | 79.38% ✅           |
| sea.png        | glacier (99.52%) ❌ | glacier (59.54%) ❌ | sea (36.18%) ✅ (mais faible confiance) |
| Snow.png       | sea (39.91%) ❌ | mountain (54.36%) ✅ | mountain (81.60%) ✅ |
| street.png     | 100.00% ✅   | 96.07% ✅      | 75.86% ✅           |

> Le **Deep CNN avec RBF** est le plus équilibré sur ces exemples difficiles (mer/neige/glacier).

