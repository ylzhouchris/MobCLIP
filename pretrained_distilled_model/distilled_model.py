import torch
import torch.nn as nn
from spherical_harmonics_ylm import SH as SH_analytic


"""
Spherical Harmonics locaiton encoder
"""
class SphericalHarmonics(nn.Module):
    def __init__(self, legendre_polys):
        """
        legendre_polys: determines the number of legendre polynomials.
                        more polynomials lead more fine-grained resolutions

        """
        super(SphericalHarmonics, self).__init__()
        self.L, self.M = int(legendre_polys), int(legendre_polys)
        self.embedding_dim = self.L * self.M
        self.SH = SH_analytic

    def forward(self, lonlat):
        lon, lat = lonlat[:, 0], lonlat[:, 1]

        # convert degree to rad
        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)

        Y = []
        for l in range(self.L):
            for m in range(-l, l + 1):
                y = self.SH(m, l, phi, theta)
                if isinstance(y, float):
                    y = y * torch.ones_like(phi)
                Y.append(y)

        return torch.stack(Y,dim=-1)

    

class MLP(nn.Module):
    def __init__(self, layer_dims):
        """
        Args:
            layer_dims (list of int): A list containing the dimensions of each layer, 
                                      including the input layer, hidden layers, and output layer.
    
        """
        super(MLP, self).__init__()
        layers = []

       
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1])) 
            if i < len(layer_dims) - 2: 
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    

class DistilledMobCLIP(nn.Module):
    def __init__(self, mlp_layer_dims, legendre_polys: int = 32):
        
        
        super(DistilledMobCLIP, self).__init__()
        self.posenc = SphericalHarmonics(legendre_polys=legendre_polys).double()
        self.net = MLP(mlp_layer_dims).double()
        
        
    def forward(self,lonlat):
        x = self.posenc(lonlat)
        return self.net(x)



def load(ckpt_path, device):
    ckpt = torch.load(ckpt_path,map_location=device, weights_only=True)
    model = DistilledMobCLIP([1024] + [512] * 8 + [128]).to(device)
    model.net.load_state_dict(ckpt['model_state_dict'])
    
    model.eval()

    return model