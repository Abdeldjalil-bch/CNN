"""
Fonctions utilitaires pour l'augmentation de données et la visualisation.
"""
import torch
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import numpy as np


def get_mixup_cutmix_transform(num_classes, mixup_alpha=0.2, cutmix_alpha=1.0, apply_prob=0.6):
    """
    Crée une transformation combinant MixUp et CutMix.
    
    Args:
        num_classes (int): Nombre de classes dans le dataset
        mixup_alpha (float): Paramètre alpha pour MixUp (0.2 = mélange doux)
        cutmix_alpha (float): Paramètre alpha pour CutMix (1.0 = standard)
        apply_prob (float): Probabilité d'appliquer l'augmentation (0.6 = 60%)
    
    Returns:
        transform: Transformation v2.RandomApply contenant MixUp ou CutMix
    
    Example:
        >>> transform = get_mixup_cutmix_transform(num_classes=6, apply_prob=0.6)
        >>> images, labels = next(iter(train_loader))
        >>> images, labels = transform(images, labels)
    """
    # Définition des deux augmentations
    mixup = v2.MixUp(alpha=mixup_alpha, num_classes=num_classes)
    cutmix = v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes)
    
    # Choisir aléatoirement entre MixUp ou CutMix (50%/50%)
    mixup_or_cutmix = v2.RandomChoice([mixup, cutmix])
    
    # Appliquer avec une probabilité donnée
    transform = v2.RandomApply([mixup_or_cutmix], p=apply_prob)
    
    return transform


def get_optimizer_and_scheduler(model, lr=0.001, patience=4, factor=0.5):
    """
    Crée l'optimizer et le scheduler pour l'entraînement.
    
    Args:
        model: Le modèle PyTorch
        lr (float): Learning rate initial
        patience (int): Nombre d'epochs sans amélioration avant réduction du LR
        factor (float): Facteur de réduction du LR (0.5 = diviser par 2)
    
    Returns:
        optimizer, scheduler: Tuple (optimizer, scheduler)
    
    Example:
        >>> optimizer, scheduler = get_optimizer_and_scheduler(model, lr=0.001)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=patience, 
        factor=factor,
        verbose=True
    )
    return optimizer, scheduler


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Affiche les courbes d'entraînement (Loss et Accuracy).
    
    Args:
        train_losses (list): Liste des train losses par epoch
        val_losses (list): Liste des validation losses par epoch
        train_accs (list): Liste des train accuracies par epoch
        val_accs (list): Liste des validation accuracies par epoch
        save_path (str, optional): Chemin pour sauvegarder la figure
    
    Example:
        >>> plot_training_history(train_losses, val_losses, train_accs, val_accs,
        ...                       save_path='images/training_curves.png')
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    axes[0].plot(train_losses, label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Accuracy
    axes[1].plot(train_accs, label='Train Accuracy', marker='o', linewidth=2)
    axes[1].plot(val_accs, label='Val Accuracy', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure sauvegardée: {save_path}")
    
    plt.show()


def visualize_augmentation(image, label, transform, num_classes, class_names, num_samples=4):
    """
    Visualise l'effet de MixUp/CutMix sur une image.
    
    Args:
        image (tensor): Image d'entrée [C, H, W]
        label (int): Label de l'image
        transform: Transformation MixUp/CutMix
        num_classes (int): Nombre de classes
        class_names (list): Liste des noms de classes
        num_samples (int): Nombre d'échantillons augmentés à générer
    
    Example:
        >>> transform = get_mixup_cutmix_transform(num_classes=6)
        >>> image, label = dataset[0]
        >>> visualize_augmentation(image, label, transform, 6, class_names)
    """
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(20, 4))
    
    # Image originale
    img_np = image.permute(1, 2, 0).numpy()
    img_np = (img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    img_np = np.clip(img_np, 0, 1)
    
    axes[0].imshow(img_np)
    axes[0].set_title(f'Original\n{class_names[label]}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Images augmentées
    for i in range(num_samples):
        # Créer un batch avec 2 images identiques
        batch = image.unsqueeze(0).repeat(2, 1, 1, 1)
        labels = torch.tensor([label, label])
        
        # Appliquer la transformation
        aug_batch, aug_labels = transform(batch, labels)
        
        # Afficher la première image du batch
        aug_img = aug_batch[0].permute(1, 2, 0).numpy()
        aug_img = (aug_img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        aug_img = np.clip(aug_img, 0, 1)
        
        axes[i + 1].imshow(aug_img)
        
        # Titre avec les probabilités des classes
        if aug_labels[0].dim() > 0:  # Soft label
            top_probs, top_indices = torch.topk(aug_labels[0], k=2)
            title = f'{class_names[top_indices[0]]}: {top_probs[0]:.2f}\n'
            title += f'{class_names[top_indices[1]]}: {top_probs[1]:.2f}'
        else:
            title = class_names[aug_labels[0].item()]
        
        axes[i + 1].set_title(f'Augmented {i+1}\n{title}', fontsize=10)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def count_parameters(model):
    """
    Compte le nombre de paramètres entraînables dans un modèle.
    
    Args:
        model: Modèle PyTorch
    
    Returns:
        int: Nombre total de paramètres entraînables
    
    Example:
        >>> model = CustomCNN(num_classes=6)
        >>> num_params = count_parameters(model)
        >>> print(f"Total parameters: {num_params:,}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    """
    Affiche un résumé du modèle (architecture + nombre de paramètres).
    
    Args:
        model: Modèle PyTorch
    
    Example:
        >>> model = CustomCNN(num_classes=6)
        >>> print_model_summary(model)
    """
    print("=" * 80)
    print("📊 MODEL SUMMARY")
    print("=" * 80)
    print(model)
    print("-" * 80)
    
    total_params = count_parameters(model)
    print(f"\n✅ Total trainable parameters: {total_params:,}")
    print(f"💾 Model size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    print("=" * 80)


# Configuration pour différents modèles
CONFIGS = {
    'baseline': {
        'lr': 0.001,
        'patience': 7,
        'mixup_alpha': None,  # Pas d'augmentation avancée
        'cutmix_alpha': None,
        'apply_prob': 0.0,
        'description': 'CNN simple sans MixUp/CutMix'
    },
    'deep_cnn': {
        'lr': 0.001,
        'patience': 7,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
        'apply_prob': 0.6,
        'description': 'Deep CNN avec MixUp/CutMix (p=0.6)'
    },
    'resnet18': {
        'lr': 0.0001,
        'patience': 4,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
        'apply_prob': 0.6,
        'description': 'ResNet18 fine-tuning avec MixUp/CutMix (p=0.6)'
    }
}


def get_config(model_name):
    """
    Récupère la configuration pour un modèle donné.
    
    Args:
        model_name (str): Nom du modèle ('baseline', 'deep_cnn', 'resnet18')
    
    Returns:
        dict: Configuration du modèle
    
    Example:
        >>> config = get_config('deep_cnn')
        >>> print(config['lr'])  # 0.001
    """
    if model_name not in CONFIGS:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(CONFIGS.keys())}")
    return CONFIGS[model_name]