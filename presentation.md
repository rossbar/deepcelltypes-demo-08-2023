---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Deepcelltypes Demo

```{code-cell} ipython3
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tff

from celltype_processing import (
    preprocess, convert_to_model_input, run_model, visualize_predictions
)
```

Load image and prepare for segmentation

```{code-cell} ipython3
# Load image
img = tff.imread("data/DCIS_Risom_breast_example_MIBI.tif")
# Load channel metadata
channels = np.loadtxt("data/channelnames.txt", dtype=str, delimiter="\t")
# Mapping channel names to img indices
ch_to_idx = dict(zip(channels, range(channels.shape[0])))
```

```{code-cell} ipython3
# Rm cruft from previous runs
if os.path.exists("data/mask.tif"):
    os.remove("data/mask.tif")
```

Segment

```{code-cell} ipython3
!docker run -it -v \
    $PWD/data:/data \
    vanvalenlab/deepcell-applications:latest mesmer \
    --nuclear-image /data/DCIS_Risom_breast_example_MIBI.tif \
    --nuclear-channel 27 \
    --membrane-image /data/DCIS_Risom_breast_example_MIBI.tif \
    --membrane-channel 44 \
    --image-mpp 0.49 \
    --output-name /data/mask.tif \
    --compartment whole-cell
```

```{code-cell} ipython3
# Load segmentation mask resulting from mesmer
mask = tff.imread("data/mask.tif").squeeze()
```

Visualize

```{code-cell} ipython3
# Visualize nuclear, membrane, and resulting masks
nuc_ch = img[..., ch_to_idx["HH3"]]
mem_ch = img[..., ch_to_idx["sum_PanCK_CD45"]]
nuc_ch = np.clip(nuc_ch, a_min=0, a_max=np.quantile(nuc_ch, 0.999))
mem_ch = np.clip(mem_ch, a_min=0, a_max=np.quantile(mem_ch, 0.999))
```

```{code-cell} ipython3
fig, ax = plt.subplots(1, 3, figsize=(9, 12))
for a, im, ttl in zip(
    ax,
    (nuc_ch, mem_ch, mask.squeeze()),
    ("HH3 (nuclear)", "PanCK+CD45 (membrane)", "Segmentation"),
):
    a.imshow(im.T)
    a.set_title(ttl)
```

Prepare image data for prediction

```{code-cell} ipython3
model_input = convert_to_model_input(*preprocess(img, mask, channels))
```

Run prediction

```{code-cell} ipython3
model_dir = Path("model")
celltypes = run_model(model_input, model_dir)
```

Predicted distribution of cell types

```{code-cell} ipython3
categories, counts = np.unique(list(celltypes.values()), return_counts=True)
dict(zip(categories, counts))
```

Visualize

```{code-cell} ipython3
visualize_predictions(mask, celltypes)
```
