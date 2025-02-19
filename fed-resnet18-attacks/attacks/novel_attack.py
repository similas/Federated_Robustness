# attacks/novel_attack.py
import torch
from collections import OrderedDict
import torch.nn.functional as F
from client import FederatedClient
import config

class NovelAttackClient(FederatedClient):
    """
    Implements a novel constrained optimization attack.
    The attacker computes its honest update first and then applies a small, 
    optimized perturbation that maximizes a malicious objective (e.g., increasing loss on a target class)
    while staying within a norm-bound of the honest update.
    """
    def __init__(self, client_id, data_indices, attack_config):
        self.attack_config = attack_config
        super().__init__(client_id, data_indices)
        print(f"Initialized novel attack client {client_id} with config: {attack_config}")
    
    def train_local_model(self):
        # Step 1: Perform normal local training to get the honest update
        honest_state, loss, samples = super().train_local_model()
        
        # Step 2: Create a perturbable copy of the honest state
        perturbed_state = OrderedDict()
        norm_bound = self.attack_config.get('norm_bound', 0.1)
        malicious_weight = self.attack_config.get('malicious_weight', 1.0)
        inner_lr = self.attack_config.get('inner_lr', 0.01)
        num_inner_steps = self.attack_config.get('num_inner_steps', 5)
        
        for key, param in honest_state.items():
            perturbed_state[key] = param.clone().detach().float().requires_grad_(True)
        
        # Adjust dummy input size based on model type
        img_size = 224 if config.MODEL_TYPE.lower() == "vit" else 32
        dummy_input = torch.randn(1, config.NUM_CHANNELS, img_size, img_size).to(config.DEVICE)
        target_label = torch.tensor([config.NUM_CLASSES - 1]).to(config.DEVICE)  # Example target
        
        # Optimize the perturbation with gradient ascent
        for step in range(num_inner_steps):
            temp_model = self._build_temp_model(perturbed_state)
            temp_model.train()
            output = temp_model(dummy_input)
            malicious_loss = torch.nn.functional.cross_entropy(output, target_label)
            # Updated code in novel_attack.py (inside train_local_model)
            grads = torch.autograd.grad(
                malicious_loss, list(perturbed_state.values()),
                retain_graph=False, allow_unused=True
            )

            for idx, key in enumerate(perturbed_state.keys()):
                grad = grads[idx]
                if grad is None:
                    grad = torch.zeros_like(perturbed_state[key])
                # Gradient ascent update
                perturbed_state[key] = perturbed_state[key] + inner_lr * malicious_weight * grad
                # Project perturbation back into the allowed norm ball
                diff = perturbed_state[key] - honest_state[key].float()
                norm = diff.norm()
                if norm > norm_bound:
                    diff = diff * (norm_bound / norm)
                perturbed_state[key] = honest_state[key].float() + diff

        fake_loss = loss * 0.8  # Adjusted reported loss
        inflated_samples = samples  # Or adjust if desired
        
        return perturbed_state, fake_loss, inflated_samples
    
    def _build_temp_model(self, state_dict):
        """
        Build a temporary model using the current global model architecture and the given state dict.
        This is used to evaluate the malicious objective.
        """
        from torchvision.models import resnet18, resnet50
        if config.MODEL_TYPE == "resnet18":
            model = resnet18(num_classes=config.NUM_CLASSES)
        elif config.MODEL_TYPE == "resnet50":
            model = resnet50(num_classes=config.NUM_CLASSES)
        elif config.MODEL_TYPE == "vit":
            # For simplicity, we can use torchvision's ViT model (assuming available) or a custom ViT.
            # Here we assume a ViT is available; adjust parameters as needed.
            from torchvision.models import vit_b_16, vit_b_32
            model = vit_b_32(num_classes=config.NUM_CLASSES)
        else:
            model = resnet18(num_classes=config.NUM_CLASSES)
        
        model.load_state_dict(state_dict, strict=False)
        model.to(config.DEVICE)
        return model