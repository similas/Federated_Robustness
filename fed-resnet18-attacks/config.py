import torch

# Device Configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Network Configuration
NUM_CHANNELS = 3
NUM_CLASSES = 10

# Federated Learning Parameters
NUM_CLIENTS = 10
NUM_ROUNDS = 10
CLIENTS_PER_ROUND = 5
LOCAL_EPOCHS = 10

# Batch Size Parameters
LOCAL_BATCH_SIZE = 64
MIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 128

# Data Distribution Parameters
MIN_SAMPLES_PER_CLIENT = 200  # Minimum samples each client should have
SAMPLES_PER_CLASS = 20        # Minimum samples per class for each client

# Optimization Parameters
LEARNING_RATE = 0.001  # Suitable for MPS
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Paths
DATA_PATH = "./data"
MODEL_PATH = "./models"

# Logging Configuration
LOG_INTERVAL = 10
WANDB_PROJECT = "federated-resnet18-cifar10-mps"
ENABLE_WANDB = False

# Attack Configuration
LABEL_FLIP_CONFIG = {
    'num_malicious': 7,              # Increased to 70% of clients
    'flip_probability': 1.0,         # Always flip labels
    'source_label': 0,              # Target specific class (airplane)
    'target_label': 2,              # Convert to bird - similar enough to be confusing
    'malicious_client_ids': [1, 2, 3, 4, 5, 6, 7],  # Majority of clients are malicious
    'attack_type': 'label_flip'
}

BACKDOOR_CONFIG = {
    'num_malicious': 7,
    'target_label': 0,
    'poison_ratio': 0.8,           # Poison 80% of local data
    'trigger_size': 5,             # Larger trigger pattern
    'trigger_intensity': 2.0,      # More visible trigger
    'malicious_client_ids': [1, 2, 3, 4, 5, 6, 7],
    'attack_type': 'backdoor'
}

MODEL_REPLACEMENT_CONFIG = {
    'num_malicious': 7,
    'boost_factor': 20.0,          # Double the amplification
    'noise_range': 5.0,            # More aggressive noise injection
    'target_layers': ['layer3', 'layer4'],  # Attack multiple crucial layers
    'scale_weights': True,         # Scale legitimate weights down
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

ATTACK_PARAMS = {
    'num_malicious': 7,
    'malicious_client_ids': [1, 2, 3, 4, 5, 6, 7],
    'target_layers': ['layer3', 'layer4'],  # Most critical layers
    'scale_weights': True,
    'noise_range': 5.0,
    'base_scale': 10.0,
    'boost_factor': 20.0,
    'scale_factor': 15.0,
    'poison_ratio': 0.8,
    'flip_probability': 1.0,
    'source_label': 0,
    'target_label': 2,
    'trigger_size': 5,
    'trigger_intensity': 2.0,
    'momentum_factor': 0.9,
    'noise_scale': 0.2,
    'initial_poison_ratio': 0.6,
    'attack_type':""
}

# print(f"Using device: {DEVICE}")
# print(f"Attack configuration enabled: {ATTACK_PARAMS['num_malicious']} malicious clients")