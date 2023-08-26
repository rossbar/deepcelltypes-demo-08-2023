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

# Deepcelltypes Demo - Cell type prediction

+++ {"slideshow": {"slide_type": "subslide"}}

This notebook provides and example of the deepcell ecosystem applied to
multiplexed spatial proteomic data for cell-type prediction.

The dataset used in this demo consists of a MIBI image of breast tissue drawn
from:

> Risom T, Glass DR, Averbukh I, Liu CC, Baranski A, Kagel A, McCaffrey
> EF, Greenwald NF, Rivero-Gutierrez B, Strand SH, Varma S, Kong A, Keren L,
> Srivastava S, Zhu C, Khair Z, Veis DJ, Deschryver K, Vennam S, Maley C, Hwang
> ES, Marks JR, Bendall SC, Colditz GA, West RB, Angelo M. Transition to invasive
> breast cancer is associated with progressive changes in the structure and
> composition of tumor stroma. Cell. 2022 Jan 20;185(2):299-310.e18.

```{code-cell} ipython3
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tff
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from celltype_processing import (
    preprocess, convert_to_model_input, run_model, visualize_predictions
)
```

We begin by loading the multiplexed image:

```{code-cell} ipython3
# Load image
img = tff.imread("data/DCIS_Risom_breast_example_MIBI.tif")
# Load channel metadata
channels = np.loadtxt("data/channelnames.txt", dtype=str, delimiter="\t")
# Mapping channel names to img indices
ch_to_idx = dict(zip(channels, range(channels.shape[0])))
channels  # The image marker panel
```

```{code-cell} ipython3
# A bit of clean up to prevent collision with previous runs
if os.path.exists("data/mask.tif"):
    os.remove("data/mask.tif")
```

Next, the image is segmented using [Mesmer](https://pubmed.ncbi.nlm.nih.gov/34795433/).
The mesmer model expects two inputs for whole-cell segmentation: a nuclear
marker and a pan-membrane marker.

In this case, we can use Histone H3 and pan-cytokeratin + CD45, respectively:

```{code-cell} ipython3
print(f"Nuclear channel index: {ch_to_idx['HH3']}")
print(f"Whole-cell index: {ch_to_idx['sum_PanCK_CD45']}")
```

At this point we have all the necessary inputs to run Mesmer. For fine-grained
control of the computational environment (including hardware acceleration),
we'll use a containerized version of the model:

```{code-cell}
:tags: [remove-output]
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

The model generates a segmentation mask, which is saved to disk.

```{code-cell} ipython3
# Load segmentation mask resulting from mesmer
mask = tff.imread("data/mask.tif").squeeze()
```

```{code-cell} ipython3
# Visualize nuclear, membrane, and resulting masks
nuc_ch = img[..., ch_to_idx["HH3"]]
mem_ch = img[..., ch_to_idx["sum_PanCK_CD45"]]
nuc_ch = np.clip(nuc_ch, a_min=0, a_max=np.quantile(nuc_ch, 0.999))
mem_ch = np.clip(mem_ch, a_min=0, a_max=np.quantile(mem_ch, 0.999))
```

```{code-cell} ipython3
fig, ax = plt.subplots(1, 3, figsize=(10, 8))
for a, im, ttl in zip(
    ax,
    (nuc_ch, mem_ch, mask.squeeze()),
    ("HH3 (nuclear)", "PanCK+CD45 (membrane)", "Segmentation"),
):
    a.imshow(im.T)
    a.set_title(ttl)
```

The primary inputs to the deepcelltypes prediction model are the multiplexed
image, the segmentation, and the marker panel which describes the mapping between
the multiplexed channel and the protein marker.

The current version of the model has been trained on datasets across imaging
platforms comprising a wide variety of protein markers, and incorporates over
50 markers for prediction.

The number of protein markers utilized by the model continually increases with
each iteration as more data is ingested.

```{code-cell} ipython3
model_input = convert_to_model_input(*preprocess(img, mask, channels))
```

The deepcelltypes model classifies each segmented cell. This version of the
model comprises 17 cell-type classes:

```{code-cell} ipython3
:tags: [remove-output]
model_dir = Path("model")
celltypes = run_model(model_input, model_dir)
```

```{code-cell} ipython3
categories, counts = np.unique(list(celltypes.values()), return_counts=True)
dict(zip(categories, counts))
```

The predictions map to the cell masks from the segmentation:

```{code-cell} ipython3
visualize_predictions(mask, celltypes)
```
