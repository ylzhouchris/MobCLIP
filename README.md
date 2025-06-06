# MobCLIP - Learning General-purpose Geospatial Representation at Scale

![mobclip](/figures/mobclip.png)

*The methodological framework of MobCLIP.*

## 

MobCLIP is the first nationwide general-purpose location encoder that integrates an unprecedented diversity of data modalities (i.e., human mobility as a graph, Points-of-Interest (POIs) as textual data, satellite imagery as visual input, and demographic distributions represented as tabular categorical histograms) through a scalable multimodal fusion framework. It innovatively leverages mobility data as the backbone, fusing information from other modalities into it using a CLIP-based approach.  


## Environment Settings
Install required packages

```python
pip install -r requirements.txt
```


## Training

To train MobCLIP, prepare the required dataset, configure the paths appropriately, adjust the training parameters in `mobclip/configs/default_ChinaFullSet.yaml`, and start training by executing:
```bash
cd mobclip
python main.py
```


## Inference 

We distill the MobCLIP embeddings for privacy concerns. The distilled model consists of two components, i.e., a position encoder that maps countinuous geographic coordinates into 1024-d embeddings and an MLP trained to fit the pretrained region embeddings from MobCLIP. The model is trained under the supervision of MobCLIP’s grid-level embeddings, allowing users to retrieve embeddings for any arbitrary coordinate.

Usage of pretrained surrogate model is simple.

```python
from distilled_model import *
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

c = torch.randn(32, 2)  # Represents a batch of 32 locations (lon/lat)

model = load(path, device=device) # Load the distilled model as a surrogate for MobCLIP.

model.eval()
with torch.no_grad():
    emb = model(c.double().to(device)).detach().cpu()
```

For detailed usage demonstration, please find the tutorial notebook: [MobCLIP usage tutorial](tutorial.ipynb) 

## Downstream evaluation

We construct a benchmark dataset of 11 downstream prediction tasks spanning social, economic, and natural domains at various spatial scales to evaluate the performance of the embeddings. MobCLIP achieves significantly superior performance—improving by over 35\% on average—across all general-purpose tasks. See [Downstream tasks evaluation](evaluation/downstream_evaluation.ipynb)  for a detailed demonstration of downstream usage and evaluation.


## Accreditation

Dataset and codebase were created by Wen Ya, Jixuan Cai, Qiyao Ma, and Yulun Zhou. Please cite the following article for all use of the open-sourced dataset and codebase.

```
@article{wen2025mobclip,
  title={MobCLIP: Learning General-purpose Geospatial Representation at Scale},
  author={Wen, Ya and Cai, Jixuan and Ma, Qiyao and Li, Linyan and Chen, Xinhua and Webster, Chris and Zhou, Yulun},
  journal={arXiv preprint arXiv:2506.01297},
  year={2025}
}
```

Ya Wen and Jixuan Cai and Qiyao Ma and Linyan Li and Xinhua Chen and Chris Webster and Yulun Zhou (2025). MobCLIP: Learning General-purpose Geospatial Representation at Scale. Arxiv. https://arxiv.org/abs/2506.01297.
     
     

![tasks](/figures/tasks.png)



