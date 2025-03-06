# config.py
import torch

# ======================
# Device Configuration
# ======================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ======================
# Model and Dataset Options
# ======================
MODEL_TYPE = "vit"  # Options: "resnet18", "resnet50", "vit"
DATASET = "CIFAR10"  # Options: "CIFAR10", "CIFAR100"
NUM_CHANNELS = 3
NUM_CLASSES = 10  # Adjust for CIFAR-10 (or change for CIFAR-100)

# ViT-specific parameters (if MODEL_TYPE=="vit")
VIT_PATCH_SIZE = 32  # For 32x32 images, smaller patch size may be needed
VIT_EMBED_DIM = 32
VIT_DEPTH = 4
VIT_NUM_HEADS = 2


# ======================
# Federated Learning Parameters
# ======================
NUM_CLIENTS = 10
NUM_ROUNDS = 1
CLIENTS_PER_ROUND = 5
LOCAL_EPOCHS = 1

# ======================
# Batch Size and Data Distribution
# ======================
LOCAL_BATCH_SIZE = 64
MIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 128
MIN_SAMPLES_PER_CLIENT = 200  # Minimum samples each client should have

# ======================
# Optimization Hyperparameters
# ======================
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# ======================
# Paths
# ======================
DATA_PATH = "./data"
MODEL_PATH = "./models"

# ======================
# Logging Configuration
# ======================
LOG_INTERVAL = 10
WANDB_PROJECT = "federated-vision-attack-defence"
ENABLE_WANDB = False

# ======================
# Existing Attack Configurations
# ======================
LABEL_FLIP_CONFIG = {
    'num_malicious': 7,
    'flip_probability': 1.0,
    'source_label': 0,
    'target_label': 2,
    'malicious_client_ids': [1, 2, 3, 4, 5, 6, 7],
    'attack_type': 'label_flip'
}

BACKDOOR_CONFIG = {
    'num_malicious': 7,
    'target_label': 0,
    'poison_ratio': 0.8,
    'trigger_size': 5,
    'trigger_intensity': 2.0,
    'malicious_client_ids': [1, 2, 3, 4, 5, 6, 7],
    'attack_type': 'backdoor'
}

MODEL_REPLACEMENT_CONFIG = {
    'num_malicious': 7,
    'boost_factor': 20.0,
    'noise_range': 5.0,
    'target_layers': ['layer3', 'layer4'],
    'scale_weights': True,
    'malicious_client_ids': [1, 2, 3, 4, 5, 6, 7],
    'attack_type': 'model_replacement'
}

CASCADE_ATTACK_CONFIG = {
    'num_malicious': 7,
    'scale_factor': 5.0,
    'initial_poison_ratio': 0.3,
    'malicious_client_ids': [1, 2, 3, 4, 5, 6, 7],
    'attack_type': 'cascade'
}

DELTA_ATTACK_CONFIG = {
    'num_malicious': 7,
    'base_scale': 10.0,
    'malicious_client_ids': [1, 2, 3, 4, 5, 6, 7],
    'attack_type': 'delta',
    'momentum_factor': 0.4,
    'noise_scale': 0.2
}

# ======================
# Novel Attack: Constrained Optimization Attack
# ======================
NOVEL_ATTACK_CONFIG = {
    'enabled': True,
    'norm_bound': 0.1,             # Maximum L2 norm of perturbation
    'malicious_weight': 1.0,       # Trade-off factor for the malicious objective
    'inner_lr': 0.01,              # Learning rate for the inner optimization loop
    'num_inner_steps': 5,          # Number of optimization steps to craft the update
    'malicious_client_ids': [1, 2, 3, 4, 5, 6, 7],
    'attack_type': 'novel'
}

# ======================
# Defense Configuration
# ======================
# Defense type options: "fedavg", "krum", "median", "norm_clipping"
DEFENSE_TYPE = "fedavg"
KRUM_NEIGHBORS = 3           # Used if DEFENSE_TYPE=="krum"
TRIM_PERCENTAGE = 0.2        # Used if DEFENSE_TYPE=="median" (trimmed mean: percentage to remove)
CLIP_THRESHOLD = 5.0         # Used if DEFENSE_TYPE=="norm_clipping"

# ======================
# General Experiment Options
# ======================
EXPERIMENT_ID = "exp_001"
SEED = 42