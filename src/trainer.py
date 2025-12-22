"""
Fonctions d'entraînement pour les modèles de classification de scènes.

Ce module contient:
- train_model: Entraînement standard sans MixUp/CutMix
- train_with_mixup_cutmix: Entraînement avec MixUp/CutMix
- plot_training_history: Visualisation des métriques
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, 
                num_epochs=20, device='cuda', early_stopping_patience=5, 
                save_best_model=True, model_path='best_model.pth'):
    """
    Fonction d'entraînement standard (SANS MixUp/CutMix).
    
    Args:
        model: Modèle PyTorch
        train_loader: DataLoader d'entraînement
        val_loader: DataLoader de validation
        optimizer: Optimiseur (Adam, SGD, etc.)
        criterion: Fonction de perte (CrossEntropyLoss)
        scheduler: Scheduler pour le learning rate
        num_epochs: Nombre d'epochs (défaut: 20)
        device: 'cuda' ou 'cpu'
        early_stopping_patience: Patience pour early stopping (défaut: 5)
        save_best_model: Sauvegarder le meilleur modèle (défaut: True)
        model_path: Chemin de sauvegarde du modèle (défaut: 'best_model.pth')
    
    Returns:
        dict: Historique d'entraînement contenant les métriques
    
    Example:
        >>> history = train_model(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     optimizer=optimizer,
        ...     criterion=criterion,
        ...     scheduler=scheduler,
        ...     num_epochs=20,
        ...     device='cuda'
        ... )
    """
    
    # Historique des métriques
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    # Temps d'entraînement
    start_time = time.time()
    
    print("="*70)
    print(f"🚀 DÉBUT DE L'ENTRAÎNEMENT - {num_epochs} epochs")
    print("="*70)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # ==================== ENTRAÎNEMENT ====================
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Barre de progression pour l'entraînement
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                         leave=False, ncols=100)
        
        for data, targets in train_pbar:
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores, targets)
            
            loss.backward()
            
            # Gradient clipping (optionnel, pour stabilité)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # Accuracy
            _, predictions = torch.max(scores, 1)
            correct_train += (predictions == targets).sum().item()
            total_train += targets.size(0)
            
            # Mise à jour de la barre de progression
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct_train/total_train:.2f}%'
            })
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        
        # ==================== VALIDATION ====================
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', 
                       leave=False, ncols=100)
        
        with torch.no_grad():
            for data, targets in val_pbar:
                data = data.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                scores = model(data)
                loss = criterion(scores, targets)
                
                epoch_val_loss += loss.item()
                
                _, predictions = torch.max(scores, 1)
                correct_val += (predictions == targets).sum().item()
                total_val += targets.size(0)
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*correct_val/total_val:.2f}%'
                })
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        
        # Sauvegarder les métriques
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Scheduler step
        scheduler.step(avg_val_loss)
        
        # Temps de l'epoch
        epoch_time = time.time() - epoch_start
        
        # ==================== AFFICHAGE ====================
        print(f'\n📊 Epoch [{epoch+1}/{num_epochs}] - {epoch_time:.2f}s')
        print(f'  Train → Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'  Val   → Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print(f'  LR    → {optimizer.param_groups[0]["lr"]:.6f}')
        
        # ==================== EARLY STOPPING ====================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            if save_best_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_acc': val_accuracy,
                    'history': history
                }, model_path)
                print(f'  ✅ Meilleur modèle sauvegardé! (Val Loss: {best_val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'  ⚠️  Patience: {patience_counter}/{early_stopping_patience}')
            
            if patience_counter >= early_stopping_patience:
                print(f'\n🛑 Early stopping à l\'epoch {epoch+1}')
                print(f'   Meilleur modèle: Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})')
                break
        
        print('-' * 70)
    
    # ==================== FIN DE L'ENTRAÎNEMENT ====================
    total_time = time.time() - start_time
    print('\n' + '='*70)
    print(f'✅ ENTRAÎNEMENT TERMINÉ en {total_time/60:.2f} minutes')
    print(f'   Meilleur epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f}')
    print('='*70)
    
    # ==================== VISUALISATIONS ====================
    plot_training_history(history, best_epoch)
    
    return history


def train_with_mixup_cutmix(model, train_loader, val_loader, optimizer, criterion, 
                            scheduler, mixup_cutmix_transform, num_epochs=20, 
                            device='cuda', early_stopping_patience=5, 
                            save_best_model=True, model_path='best_model.pth'):
    """
    Fonction d'entraînement AVEC MixUp/CutMix.
    
    Args:
        model: Modèle PyTorch
        train_loader: DataLoader d'entraînement
        val_loader: DataLoader de validation
        optimizer: Optimiseur
        criterion: Fonction de perte
        scheduler: Scheduler pour le learning rate
        mixup_cutmix_transform: Transform MixUp/CutMix (de utils.py)
        num_epochs: Nombre d'epochs
        device: 'cuda' ou 'cpu'
        early_stopping_patience: Patience pour early stopping
        save_best_model: Sauvegarder le meilleur modèle
        model_path: Chemin de sauvegarde du modèle
    
    Returns:
        dict: Historique d'entraînement
    
    Example:
        >>> from src.utils import get_mixup_cutmix_transform
        >>> mixup_cutmix = get_mixup_cutmix_transform(num_classes=6)
        >>> history = train_with_mixup_cutmix(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     optimizer=optimizer,
        ...     criterion=criterion,
        ...     scheduler=scheduler,
        ...     mixup_cutmix_transform=mixup_cutmix,
        ...     num_epochs=20
        ... )
    """
    
    # Historique des métriques
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    # Temps d'entraînement
    start_time = time.time()
    
    print("="*70)
    print(f"🚀 DÉBUT DE L'ENTRAÎNEMENT AVEC MIXUP/CUTMIX - {num_epochs} epochs")
    print("="*70)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # ==================== ENTRAÎNEMENT ====================
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                         leave=False, ncols=100)
        
        for data, targets in train_pbar:
            # 🎯 APPLIQUER MIXUP/CUTMIX ICI
            data, targets = mixup_cutmix_transform(data, targets)
            
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # Accuracy avec gestion des soft labels
            _, predictions = torch.max(scores, 1)
            if targets.dim() == 2:  # Soft labels (MixUp/CutMix)
                _, targets_hard = torch.max(targets, 1)
            else:  # Hard labels
                targets_hard = targets
            correct_train += (predictions == targets_hard).sum().item()
            total_train += targets.size(0)
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct_train/total_train:.2f}%'
            })
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        
        # ==================== VALIDATION ====================
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', 
                       leave=False, ncols=100)
        
        with torch.no_grad():
            for data, targets in val_pbar:
                data = data.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                scores = model(data)
                loss = criterion(scores, targets)
                
                epoch_val_loss += loss.item()
                
                _, predictions = torch.max(scores, 1)
                correct_val += (predictions == targets).sum().item()
                total_val += targets.size(0)
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*correct_val/total_val:.2f}%'
                })
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        
        # Sauvegarder les métriques
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Scheduler step
        scheduler.step(avg_val_loss)
        
        # Temps de l'epoch
        epoch_time = time.time() - epoch_start
        
        # ==================== AFFICHAGE ====================
        print(f'\n📊 Epoch [{epoch+1}/{num_epochs}] - {epoch_time:.2f}s')
        print(f'  Train → Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'  Val   → Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print(f'  LR    → {optimizer.param_groups[0]["lr"]:.6f}')
        
        # ==================== EARLY STOPPING ====================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            if save_best_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_acc': val_accuracy,
                    'history': history
                }, model_path)
                print(f'  ✅ Meilleur modèle sauvegardé! (Val Loss: {best_val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'  ⚠️  Patience: {patience_counter}/{early_stopping_patience}')
            
            if patience_counter >= early_stopping_patience:
                print(f'\n🛑 Early stopping à l\'epoch {epoch+1}')
                print(f'   Meilleur modèle: Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})')
                break
        
        print('-' * 70)
    
    # ==================== FIN DE L'ENTRAÎNEMENT ====================
    total_time = time.time() - start_time
    print('\n' + '='*70)
    print(f'✅ ENTRAÎNEMENT TERMINÉ en {total_time/60:.2f} minutes')
    print(f'   Meilleur epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f}')
    print('='*70)
    
    # ==================== VISUALISATIONS ====================
    plot_training_history(history, best_epoch)
    
    return history


def plot_training_history(history, best_epoch=None):
    """
    Génère des graphiques de l'historique d'entraînement.
    
    Args:
        history (dict): Historique contenant train_loss, val_loss, train_acc, val_acc, lr
        best_epoch (int, optional): Epoch du meilleur modèle (pour marquage)
    
    Example:
        >>> plot_training_history(history, best_epoch=12)
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('📈 Historique d\'entraînement', fontsize=16, fontweight='bold')
    
    # 1. Loss (Train vs Val)
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
    if best_epoch:
        ax1.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Evolution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy (Train vs Val)
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Acc', linewidth=2, markersize=4)
    if best_epoch:
        ax2.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Evolution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Overfitting Analysis
    ax3 = axes[1, 0]
    gap_loss = np.array(history['train_loss']) - np.array(history['val_loss'])
    ax3.plot(epochs, gap_loss, 'purple', label='Loss Gap (Train - Val)', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax3.fill_between(epochs, gap_loss, 0, where=(gap_loss > 0), 
                     color='red', alpha=0.2, label='Overfitting Zone')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Gap', fontsize=12)
    ax3.set_title('Overfitting Analysis (Loss)', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Learning Rate Evolution
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['lr'], 'orange', linewidth=2, marker='o', markersize=4)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Learning Rate', fontsize=12)
    ax4.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print('\n📊 Graphiques sauvegardés: training_history.png')
    plt.show()
    
    # ==================== RÉSUMÉ STATISTIQUE ====================
    print("\n" + "="*70)
    print("📋 RÉSUMÉ STATISTIQUE")
    print("="*70)
    print(f"Train Loss → Min: {min(history['train_loss']):.4f}, Max: {max(history['train_loss']):.4f}, Final: {history['train_loss'][-1]:.4f}")
    print(f"Val Loss   → Min: {min(history['val_loss']):.4f}, Max: {max(history['val_loss']):.4f}, Final: {history['val_loss'][-1]:.4f}")
    print(f"Train Acc  → Min: {min(history['train_acc']):.2f}%, Max: {max(history['train_acc']):.2f}%, Final: {history['train_acc'][-1]:.2f}%")
    print(f"Val Acc    → Min: {min(history['val_acc']):.2f}%, Max: {max(history['val_acc']):.2f}%, Final: {history['val_acc'][-1]:.2f}%")
    print(f"Overfitting Gap → {history['train_acc'][-1] - history['val_acc'][-1]:.2f}% (Train - Val)")
    print("="*70)

