import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



class LightGCN(nn.Module):
    def __init__(self,all_mob_features, num_layers):
        super(LightGCN, self).__init__()
        
        self.num_layers = num_layers
        self.ebds = nn.Parameter(all_mob_features)   
        
        
    def forward(self, adj):
        embeds = self.ebds
        embeds_list = [embeds]
        for layer in range(self.num_layers):
            embeddings = torch.spmm(adj, embeds_list[-1])
            embeds_list.append(embeddings)

        # Aggregate embeddings from all layers
        all_embeddings = torch.stack(embeds_list, dim=0)
        all_embeddings = torch.sum(all_embeddings, dim=0)
        self.final_embeds = all_embeddings

        return all_embeddings

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    
    def forward(self, x):

        x = self.fc1(x)
        x = self.activation(x) 
        x = self.fc2(x)        
        return x



class LinearHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearHead, self).__init__()
        self.head = nn.Linear(input_dim, output_dim)  
   

    def forward(self, x):
        x = self.head(x)
        return x
    

class MobCLIP(nn.Module):
    def __init__(self, poi_dim, demo_dim, image_dim, demo_hidden_dim, embedding_dim, mob_features, gnn_layers, poi_scale, demo_scale, image_scale  
            ):
        super(MobCLIP, self).__init__()
    
        self.poi_dim = poi_dim
        self.demo_dim = demo_dim
        self.image_dim = image_dim
        self.demo_hidden_dim = demo_hidden_dim
        self.embedding_dim = embedding_dim
        self.mob_features = mob_features
        self.gnn_layers = gnn_layers
        self.poi_scale = poi_scale
        self.demo_scale = demo_scale
        self.image_scale = image_scale

        self.mob_lightgcn = LightGCN(all_mob_features = self.mob_features, num_layers = self.gnn_layers)
        self.poihead = LinearHead(self.poi_dim,self.embedding_dim)
        self.demo_encoder = MLPEncoder(self.demo_dim, self.demo_hidden_dim, self.embedding_dim)
        self.imagehead = LinearHead(self.image_dim,self.embedding_dim)
    
        self.poi_logit_scale =  torch.tensor(np.log(1 / self.poi_scale), dtype=torch.float32)
        self.demo_logit_scale =  torch.tensor(np.log(1 / self.demo_scale), dtype=torch.float32)
        self.image_logit_scale =  torch.tensor(np.log(1 / self.image_scale), dtype=torch.float32)
        
        
        self.apply(self.init_weights)
 

        
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):

            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
            


    
    def forward(self, batch, mob_adj, global_indices = None
               ):
        
        logits = {}
        clip_embeddings = {}


        global_mob_ebd = self.mob_lightgcn(mob_adj)
        mob_ebd = global_mob_ebd[global_indices]
        mob_embeddings = mob_ebd / mob_ebd.norm(dim=1, keepdim=True) 
        clip_embeddings["mob"] = mob_embeddings
        
        
        poi_features = batch["poi"]
        poi_ebd = self.poihead(poi_features)  
        poi_embeddings = poi_ebd / poi_ebd.norm(dim=1, keepdim=True)
        clip_embeddings["poi"] = poi_embeddings
      
        demo_features = batch["demo"]
        demo_ebd = self.demo_encoder(demo_features)      
        demo_embeddings = demo_ebd / demo_ebd.norm(dim=1, keepdim=True)
        clip_embeddings["demo"] = demo_embeddings
        
        
        image_features = batch["image"]
        image_ebd = self.imagehead(image_features)   
        image_embeddings = image_ebd / image_ebd.norm(dim=1, keepdim=True)
        clip_embeddings["image"] = image_embeddings
    

        poi_logit_scale = self.poi_logit_scale.exp()
        demo_logit_scale = self.demo_logit_scale.exp()
        image_logit_scale = self.image_logit_scale.exp()
   
        

        logits["logits_per_mob_poi"] = poi_logit_scale * clip_embeddings["mob"] @ clip_embeddings["poi"].t()
        logits["logits_per_poi_mob"] = logits["logits_per_mob_poi"].t()

        logits["logits_per_mob_demo"] = demo_logit_scale * clip_embeddings["mob"] @ clip_embeddings["demo"].t()
        logits["logits_per_demo_mob"] = logits["logits_per_mob_demo"].t()

        logits["logits_per_mob_image"] = image_logit_scale * clip_embeddings["mob"] @ clip_embeddings["image"].t()
        logits["logits_per_image_mob"] = logits["logits_per_mob_image"].t()
        


        return logits, mob_ebd
    
   

    
        
       
    
        