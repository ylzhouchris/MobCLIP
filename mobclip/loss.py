import torch
import torch.nn.functional as F
import torch.nn as nn


class Loss(nn.Module):

    def __init__(self,):
        super().__init__()

    
    
    def forward(self,logits):
        """
        Compute the average cross-entropy loss across multiple logits.

        Args:
            logits (dict): 
                A dictionary where keys are identifiers

        Returns:
            torch.Tensor: 
                The averaged cross-entropy loss across all valid logits. 

        """
        
        for key, value in logits.items():
            if value is not None: 
                device = value.device  
                shape = value.shape  
                labels = torch.arange(shape[0], device=device, dtype=torch.long)
                break

        clip_loss = 0
        valid_count = 0

        for key, value in logits.items():
            if value is not None:
                clip_loss += F.cross_entropy(value, labels)
                valid_count += 1

        if valid_count > 0:
            clip_loss /= valid_count
        else:
            clip_loss = torch.tensor(0.0, device=labels.device)
            
            
        return clip_loss 
    
    
