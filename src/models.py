"""
Architectures de modèles pour la classification de scènes.

Ce module contient:
- BaselineCNN (my_model_CNN): CNN simple avec 2 blocs convolutifs
- RBFLinear: Couche avec activation RBF (Radial Basis Function)
- CustomCNN: CNN profond avec 4 blocs convolutifs et MLP avec RBF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
# BASELINE MODEL
# ===================================================================

class BaselineCNN(nn.Module):
    """
    CNN simple avec 2 blocs convolutifs (baseline).
    
    Architecture:
        Conv(3→24) → ReLU → MaxPool → Conv(24→8) → ReLU → AvgPool
        → Flatten → Linear(4232→220) → GELU → Linear(220→6)
    
    Args:
        num_classes (int): Nombre de classes de sortie (défaut: 6)
    
    Notes:
        - Conçu pour des images 100x100
        - Tendance à l'overfitting (pas de Batch Normalization)
        - Utilisé comme point de comparaison
    """
    def __init__(self, num_classes=6):
        super(BaselineCNN, self).__init__()
        
        # Bloc 1
        self.conv1 = nn.Conv2d(3, 24, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Bloc 2
        self.conv2 = nn.Conv2d(24, 8, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(2, 2)
        
        # Classificateur
        self.layer1 = nn.Linear(4232, 220)
        self.act = nn.GELU()
        self.layerfn = nn.Linear(220, num_classes)
    
    def forward(self, x):
        # Blocs convolutifs
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classificateur
        x = self.layer1(x)
        x = self.act(x)
        x = self.layerfn(x)
        
        return x


# Alias pour compatibilité avec l'ancien nom
my_model_CNN = BaselineCNN


# ===================================================================
# RBF ACTIVATION FUNCTION
# ===================================================================

class RBFLinear(nn.Module):
    """
    Couche avec activation RBF (Radial Basis Function).
    
    La fonction RBF est définie comme:
        RBF(x) = exp(-γ * (x - center)²)
    
    Args:
        input_dim (int): Dimension d'entrée
        output_dim (int): Dimension de sortie (non utilisé actuellement)
        num_centers (int, optional): Nombre de centres (non utilisé)
        gamma (float, optional): Paramètre gamma fixe
        gamma_scaler (float): Scalaire pour initialiser gamma (défaut: 0.01)
    
    Attributes:
        center (Parameter): Centre de la fonction RBF
        gamma (Parameter or float): Paramètre de largeur de la fonction RBF
    
    Example:
        >>> rbf = RBFLinear(128, 128, gamma_scaler=0.01)
        >>> x = torch.randn(32, 128)
        >>> output = rbf(x)
        >>> print(output.shape)  # torch.Size([32, 128])
    """
    def __init__(self, input_dim, output_dim, num_centers=None, gamma=None, gamma_scaler=0.01):
        super(RBFLinear, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_centers = num_centers
        
        # Initialiser les centres
        self.center = nn.Parameter(torch.zeros(1, self.input_dim))
        
        # Initialiser gamma
        if gamma is not None:
            self.gamma = gamma
        elif gamma_scaler is not None:
            self.gamma = nn.Parameter(torch.empty(1, self.input_dim) * gamma_scaler)
        else:
            self.gamma = nn.Parameter(torch.ones(1, self.input_dim))
        
        # Initialisation Xavier
        nn.init.xavier_normal_(self.center)
        if isinstance(self.gamma, nn.Parameter):
            nn.init.xavier_normal_(self.gamma)
    
    def forward(self, x):
        """
        Forward pass avec fonction RBF.
        
        Args:
            x (Tensor): Input tensor de shape (batch_size, input_dim)
        
        Returns:
            Tensor: Output de shape (batch_size, input_dim)
        """
        # Fonction RBF (Radial Basis Function)
        return torch.exp(-self.gamma * (x - self.center) ** 2)
    
    def extra_repr(self):
        """Information pour print(model)"""
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}'


# ===================================================================
# DEEP CNN WITH RBF
# ===================================================================

class CustomCNN(nn.Module):
    """
    CNN profond avec 4 blocs convolutifs et MLP avec activation RBF.
    
    Architecture:
        - 4 blocs convolutifs: Conv → BatchNorm → GELU → MaxPool
        - MLP avec 2 hidden layers:
            * FC → RBF activation
            * FC → ReLU activation
            * FC → Output
    
    Args:
        num_classes (int): Nombre de classes de sortie (défaut: 6)
        img_size (int): Taille des images en entrée (défaut: 120)
    
    Features:
        - Batch Normalization dans chaque bloc convolutif
        - Activation GELU pour les convolutions
        - Activation RBF innovante dans le MLP
        - Architecture pyramidale: 3→16→24→16→8 channels
    
    Example:
        >>> model = CustomCNN(num_classes=6, img_size=120)
        >>> x = torch.randn(32, 3, 120, 120)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 6])
    """
    def __init__(self, num_classes=6, img_size=120):
        super(CustomCNN, self).__init__()
        
        # ===============================================================
        # BLOCS CONVOLUTIFS
        # ===============================================================
        
        # Bloc 1: 3 → 16 channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.GELU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 120 → 60
        
        # Bloc 2: 16 → 24 channels
        self.conv2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.act2 = nn.GELU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 60 → 30
        
        # Bloc 3: 24 → 16 channels
        self.conv3 = nn.Conv2d(24, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.act3 = nn.GELU()
        self.pool3 = nn.MaxPool2d(2, 2)  # 30 → 15
        
        # Bloc 4: 16 → 8 channels
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(8)
        self.act4 = nn.GELU()
        self.pool4 = nn.MaxPool2d(2, 2)  # 15 → 7 (avec flooring)
        
        # ===============================================================
        # MLP CLASSIFIER
        # ===============================================================
        
        # Calcul de la taille après flatten
        # Pour img_size=120: 120 / (2^4) = 120 / 16 = 7.5 → 7 (floor)
        flatten_size = 8 * (img_size // 16) * (img_size // 16)  # 8 * 7 * 7 = 392
        
        # Hidden Layer 1: Linear → RBF
        self.fc1 = nn.Linear(flatten_size, 128)
        self.rbf1 = RBFLinear(128, 128, gamma_scaler=0.01)
        
        # Hidden Layer 2: Linear → ReLU
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        
        # Output Layer
        self.fc_out = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        Forward pass du réseau.
        
        Args:
            x (Tensor): Input images de shape (batch_size, 3, img_size, img_size)
        
        Returns:
            Tensor: Logits de shape (batch_size, num_classes)
        """
        # ===============================================================
        # FEATURE EXTRACTION (Blocs Convolutifs)
        # ===============================================================
        
        # Bloc 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        # Bloc 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        # Bloc 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)
        
        # Bloc 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.pool4(x)
        
        # ===============================================================
        # CLASSIFICATION (MLP)
        # ===============================================================
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Hidden Layer 1 avec RBF
        x = self.fc1(x)
        x = self.rbf1(x)
        
        # Hidden Layer 2 avec ReLU
        x = self.fc2(x)
        x = self.relu2(x)
        
        # Output Layer
        x = self.fc_out(x)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Retourne les feature maps de chaque bloc (pour visualisation).
        
        Args:
            x (Tensor): Input images
        
        Returns:
            dict: Dictionnaire contenant les feature maps de chaque bloc
        """
        features = {}
        
        # Bloc 1
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        features['bloc1'] = x
        
        # Bloc 2
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        features['bloc2'] = x
        
        # Bloc 3
        x = self.pool3(self.act3(self.bn3(self.conv3(x))))
        features['bloc3'] = x
        
        # Bloc 4
        x = self.pool4(self.act4(self.bn4(self.conv4(x))))
        features['bloc4'] = x
        
        return features


# ===================================================================
# UTILITY FUNCTIONS
# ===================================================================

def count_parameters(model):
    """
    Compte le nombre de paramètres entraînables dans un modèle.
    
    Args:
        model (nn.Module): Modèle PyTorch
    
    Returns:
        int: Nombre total de paramètres entraînables
    
    Example:
        >>> model = CustomCNN(num_classes=6)
        >>> num_params = count_parameters(model)
        >>> print(f"Parameters: {num_params:,}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model):
    """
    Retourne des informations sur le modèle.
    
    Args:
        model (nn.Module): Modèle PyTorch
    
    Returns:
        dict: Dictionnaire avec les informations du modèle
    """
    num_params = count_parameters(model)
    model_size_mb = num_params * 4 / (1024 ** 2)  # Approximation en float32
    
    return {
        'name': model.__class__.__name__,
        'parameters': num_params,
        'size_mb': model_size_mb
    }


if __name__ == '__main__':
    """Tests rapides des modèles"""
    
    print("="*70)
    print("Test des modèles")
    print("="*70)
    
    # Test BaselineCNN
    print("\n1. BaselineCNN:")
    model1 = BaselineCNN(num_classes=6)
    x1 = torch.randn(2, 3, 100, 100)
    y1 = model1(x1)
    print(f"   Input: {x1.shape} → Output: {y1.shape}")
    print(f"   Parameters: {count_parameters(model1):,}")
    
    # Test RBFLinear
    print("\n2. RBFLinear:")
    rbf = RBFLinear(128, 128, gamma_scaler=0.01)
    x2 = torch.randn(4, 128)
    y2 = rbf(x2)
    print(f"   Input: {x2.shape} → Output: {y2.shape}")
    
    # Test CustomCNN
    print("\n3. CustomCNN:")
    model3 = CustomCNN(num_classes=6, img_size=120)
    x3 = torch.randn(2, 3, 120, 120)
    y3 = model3(x3)
    print(f"   Input: {x3.shape} → Output: {y3.shape}")
    print(f"   Parameters: {count_parameters(model3):,}")
    
    print("\n" + "="*70)
    print("✅ Tous les tests réussis!")

