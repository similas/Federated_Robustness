# text_model.py
import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification
from collections import OrderedDict

class TextClassifier(nn.Module):
    """
    Text classifier based on DistilBERT for federated learning.
    Provides compatibility with the existing FL framework.
    """
    def __init__(self, num_classes=4, pretrained_model="distilbert-base-uncased"):
        super(TextClassifier, self).__init__()
        
        # Load pre-trained DistilBERT model with a classification head
        self.model = DistilBertForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            return_dict=True
        )
        
        # Freeze earlier layers to reduce computational cost and improve convergence
        # Only train the last transformer layer and the classification head
        modules = [self.model.distilbert.embeddings, *self.model.distilbert.transformer.layer[:-1]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token ids (batch_size, sequence_length)
            attention_mask: Attention mask for padding (batch_size, sequence_length)
            
        Returns:
            Model output with logits
        """
        # Process inputs to include required tensors
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        }
        
        # Forward pass through DistilBERT
        outputs = self.model(**inputs)
        
        # Return logits
        return outputs.logits
    
    def get_layer_keys(self):
        """Get all parameter keys grouped by layer for feature extraction."""
        layer_keys = {
            'embeddings': [],
            'attention': [],
            'transformer': [],
            'classifier': []
        }
        
        for name, _ in self.named_parameters():
            if 'embeddings' in name:
                layer_keys['embeddings'].append(name)
            elif 'attention' in name:
                layer_keys['attention'].append(name)
            elif 'transformer' in name:
                layer_keys['transformer'].append(name)
            elif 'classifier' in name or 'pre_classifier' in name:
                layer_keys['classifier'].append(name)
        
        return layer_keys
    
    def get_activation_statistics(self, input_ids, attention_mask=None):
        """
        Get activation statistics for feature extraction (for AAF).
        This is a helper method to extract model-specific features for anomaly detection.
        
        Returns:
            Dictionary of statistics for various model components
        """
        # Store original training mode and switch to eval
        training_mode = self.training
        self.eval()
        
        stats = {}
        with torch.no_grad():
            # Process inputs
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask if attention_mask is not None else torch.ones_like(input_ids)
            }
            
            # Get hidden states from model
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Extract statistics from hidden states
            hidden_states = outputs.hidden_states
            
            # Last hidden state statistics
            last_hidden = hidden_states[-1]
            stats['hidden_mean'] = last_hidden.mean().item()
            stats['hidden_std'] = last_hidden.std().item()
            stats['hidden_norm'] = torch.norm(last_hidden).item()
            
            # Logits statistics
            logits = outputs.logits
            stats['logits_mean'] = logits.mean().item()
            stats['logits_std'] = logits.std().item()
            stats['logits_norm'] = torch.norm(logits).item()
            
            # Attention statistics (if available)
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attentions = outputs.attentions
                stats['attention_mean'] = torch.stack(attentions).mean().item()
                stats['attention_std'] = torch.stack(attentions).std().item()
        
        # Restore original training mode
        self.train(training_mode)
        
        return stats
    
    def extract_update_features(self, old_state_dict, new_state_dict):
        """
        Extract features from model updates for AAF.
        These features help detect anomalous updates from malicious clients.
        
        Args:
            old_state_dict: Previous model state
            new_state_dict: Updated model state
            
        Returns:
            Dictionary of update features
        """
        features = {}
        
        # Calculate global update norm
        squared_sum = 0
        for key in new_state_dict:
            if key in old_state_dict:
                param_diff = new_state_dict[key] - old_state_dict[key]
                squared_sum += torch.sum((param_diff) ** 2).item()
        
        features['global_update_norm'] = torch.sqrt(torch.tensor(squared_sum)).item()
        
        # Layer group statistics
        layer_keys = self.get_layer_keys()
        
        for group_name, keys in layer_keys.items():
            group_squared_sum = 0
            group_diff_values = []
            
            for key in keys:
                if key in new_state_dict and key in old_state_dict:
                    param_diff = new_state_dict[key] - old_state_dict[key]
                    group_squared_sum += torch.sum((param_diff) ** 2).item()
                    group_diff_values.extend(param_diff.flatten().tolist())
            
            # Calculate layer-specific features
            if group_diff_values:
                features[f'{group_name}_update_norm'] = torch.sqrt(torch.tensor(group_squared_sum)).item()
                features[f'{group_name}_update_mean'] = np.mean(np.abs(group_diff_values))
                features[f'{group_name}_update_std'] = np.std(group_diff_values)
                features[f'{group_name}_update_max'] = np.max(np.abs(group_diff_values))
        
        return features