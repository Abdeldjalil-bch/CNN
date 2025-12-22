"""
Scene Classification Package
=============================

Ce package contient tous les modules nécessaires pour entraîner et évaluer
des modèles de classification de scènes.

Modules disponibles:
- models: Architectures CNN (BaselineCNN, CustomCNN, RBFLinear)
- data_loader: Chargement et préparation des données
- trainer: Fonctions d'entraînement
- utils: Utilitaires (MixUp/CutMix, visualisation, configs)

Exemple d'utilisation:
    >>> from src import CustomCNN, get_data_loaders, get_config
    >>> model = CustomCNN(num_classes=6)
    >>> train_loader, val_loader, classes = get_data_loaders('data/train', 'data/val')
    >>> config = get_config('deep_cnn')
"""

__version__ = '1.0.0'
__author__ = 'Votre Nom'

# Import des classes et fonctions principales
try:
    from .models import (
        RBFLinear,
        CustomCNN,
        BaselineCNN
    )
except ImportError:
    pass  # Les modèles seront importés à la demande

try:
    from .data_loader import get_data_loaders
except ImportError:
    pass

try:
    from .utils import (
        get_mixup_cutmix_transform,
        get_optimizer_and_scheduler,
        plot_training_history,
        visualize_augmentation,
        print_model_summary,
        count_parameters,
        get_config,
        CONFIGS
    )
except ImportError:
    pass

try:
    from .trainer import (
        train_model,
        train_with_mixup_cutmix
    )
except ImportError:
    pass

# Liste des exports publics
__all__ = [
    # Models
    'RBFLinear',
    'CustomCNN',
    'BaselineCNN',
    
    # Data
    'get_data_loaders',
    
    # Training
    'train_model',
    'train_with_mixup_cutmix',
    
    # Utils
    'get_mixup_cutmix_transform',
    'get_optimizer_and_scheduler',
    'plot_training_history',
    'visualize_augmentation',
    'print_model_summary',
    'count_parameters',
    'get_config',
    'CONFIGS',
]