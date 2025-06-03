import torch
from torch.utils.data import Dataset,DataLoader
from typing import Dict
import lightning.pytorch as pl
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp




class FeatureDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        mob_path, 
        poi_path, 
        demo_path, 
        image_path,
        mob_graph_path,
        batch_size: int=64,
        num_workers=4, 
        val_random_split_fraction = 0.1,
        ):

        super().__init__()
        self.mob_path = mob_path
        self.poi_path = poi_path
        self.demo_path = demo_path
        self.image_path = image_path
        self.mob_graph_path = mob_graph_path

        
        self.batch_size= batch_size
        self.num_workers = num_workers
        self.val_random_split_fraction = val_random_split_fraction
        
        self.save_hyperparameters()

    def setup(self, stage= None):
        
        self.dataset = CustomDataset(self.mob_path, self.poi_path, self.demo_path, self.image_path, self.mob_graph_path)
        N_val = int(len(self.dataset) * self.val_random_split_fraction)
        N_train = len(self.dataset) - N_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [N_train, N_val])

        self.mob_adj = self.dataset.mob_adj  
        self.mob_features = self.dataset.mob_features

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    

class CustomDataset(Dataset):
    def __init__(self, mob_path, poi_path, demo_path, image_path, mob_graph_path):

        self.mob_features = self._load_npy(mob_path) 
        self.poi_features = self._load_npy(poi_path)
        self.demo_features = self._load_npy(demo_path) 
        self.image_features = self._load_npy(image_path)


        assert len(self.mob_features) == len(self.poi_features) == len(self.demo_features)  == len(self.image_features), \
            "Feature lengths do not match! Ensure all directories have the same number of samples."
        
        
        self.mob_adj = self._load_mob_adj(mob_graph_path) 
        self.num_nodes = self.mob_adj.size(0)
       
    def _load_npy(self, path):

        return np.load(path)    

      
    
    def _normalize_adj(self, mat):
        """Laplacian normalization for mat in coo_matrix

        Args:
            mat (scipy.sparse.coo_matrix): the un-normalized adjacent matrix

        Returns:
            scipy.sparse.coo_matrix: normalized adjacent matrix
        """
        degree = np.array(mat.sum(axis=-1)) + 1e-10
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)

        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()
    
    def _load_mob_adj(self, mob_graph_path):
        """
        Load a mobility graph adjacency matrix from a file, normalize it, 
        and convert it to a PyTorch sparse tensor.

        Args:
            mob_graph_path (str): 
                The file path to the mobility graph data stored in `.npz` format. 

        Returns:
            torch.sparse.Tensor: 
                A normalized adjacency matrix in PyTorch sparse tensor format. 
        """
        
        loaded = np.load(mob_graph_path)
        mob_adj_coo_mat = coo_matrix((loaded["data"], (loaded["row"], loaded["col"])), shape=loaded["shape"])
        print(f"Total edges : {mob_adj_coo_mat.nnz}")
        normalized_adj_mat = self._normalize_adj(mob_adj_coo_mat)
        
        idxs = torch.from_numpy(np.vstack([normalized_adj_mat.row, normalized_adj_mat.col]).astype(np.int64))
        vals = torch.from_numpy(normalized_adj_mat.data.astype(np.float32))
        shape = torch.Size(normalized_adj_mat.shape)

        return torch.sparse_coo_tensor(idxs, vals, size=shape)
   
    def __len__(self):
        return len(self.mob_features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mob = self.mob_features[idx]  
        poi = self.poi_features[idx]  
        demo = self.demo_features[idx]  
        image = self.image_features[idx]

        mob = torch.tensor(mob, dtype=torch.float)
        poi = torch.tensor(poi, dtype=torch.float)
        demo = torch.tensor(demo, dtype=torch.float)
        image = torch.tensor(image, dtype = torch.float)

        return {
                "mob": mob,
                "poi": poi,
                "demo": demo,
                "image": image,
                "index": idx,        
            }
            

    def get_mob_graph(self):
        return self.mob_adj


