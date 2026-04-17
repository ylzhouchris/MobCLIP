# MoRA - Mobility as the Backbone for Geospatial Representation Learning at Scale

![MoRA](/figures/MoRA.png)

*The methodological framework of MoRA.*

##

MoRA is a human-centric geospatial location encoder that integrates an unprecedented diversity of data modalities (i.e., human mobility as a graph, Points-of-Interest (POIs) as textual data, satellite imagery as visual input, and demographic distributions represented as tabular categorical histograms) through a scalable multimodal fusion framework. It leverages mobility data as the backbone, fusing information from other modalities into it using a CLIP-based approach.

## Environment Settings
Install required packages

```python
pip install -r requirements.txt
```

## Training

To train MoRA, prepare the required dataset, configure the paths appropriately, adjust the training parameters in `MoRA/configs/default_ChinaFullSet.yaml`, and start training by executing:

```bash
cd MoRA
python main.py
```

## Inference

We distill the MoRA embeddings for privacy concerns. The distilled model consists of two components, i.e., a position encoder that maps countinuous geographic coordinates into 1024-d embeddings and an MLP trained to fit the pretrained region embeddings from MoRA. The model is trained under the supervision of MoRA's grid-level embeddings, allowing users to retrieve embeddings for any arbitrary coordinate.

Usage of pretrained surrogate model is simple.

```python
import sys
import torch

# Run this example from the repository root.
sys.path.append("pretrained_distilled_model")

from distilled_model import load

path = "pretrained_distilled_model/distilled_MoRA.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

c = torch.randn(32, 2)  # Represents a batch of 32 locations (lon/lat).

model = load(path, device=device)  # Load the distilled model as a surrogate for MoRA.

model.eval()
with torch.no_grad():
    emb = model(c.double().to(device)).detach().cpu()
```

For detailed usage demonstration, please find the tutorial notebook: [MoRA usage tutorial](tutorial.ipynb)

## Downstream evaluation

We construct a benchmark dataset of 9 downstream prediction tasks spanning social and economic domains at various spatial scales to evaluate the performance of the embeddings. MoRA achieves significantly superior performance, improving by 12.9% on average, across all tasks. See [Downstream tasks evaluation](evaluation/downstream_evaluation.ipynb) for a detailed demonstration of downstream usage and evaluation.

![tasks](/figures/tasks.png)

## Accreditation

Ya Wen and Jixuan Cai and Qiyao Ma and Linyan Li and Xinhua Chen and Chris Webster and Yulun Zhou* (2025). MoRA: Mobility as the Backbone for Geospatial Representation Learning at Scale. ICLR 2026. https://arxiv.org/abs/2506.01297.
