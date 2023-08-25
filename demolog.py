import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tff

from celltype_processing import preprocess, convert_to_model_input, run_model

# Load image
img = tff.imread("data/DCIS_Risom_breast_example_MIBI.tif")
# Load channel metadata
channels = np.loadtxt("data/channelnames.txt", dtype=str, delimiter="\t")

# Rm cruft from previous runs
if os.path.exists("data/mask.tif"):
    os.remove("data/mask.tif")

# Run mesmer in container
get_ipython().system('sudo docker run -it --gpus all -v $PWD/data:/data vanvalenlab/deepcell-applications:latest-gpu mesmer --nuclear-image /data/DCIS_Risom_breast_example_MIBI.tif --nuclear-channel 27 --membrane-image /data/DCIS_Risom_breast_example_MIBI.tif --membrane-channel 44 --image-mpp 0.49 --output-name /data/mask.tif --compartment whole-cell')

# Load segmentation mask resulting from mesmer
mask = tff.imread("data/mask.tif").squeeze()

# Mapping channel names to img indices
ch_to_idx = dict(zip(channels, range(channels.shape[0])))

# Visualize nuclear, membrane, and resulting masks
nuc_ch = img[..., ch_to_idx["HH3"]]
mem_ch = img[..., ch_to_idx["sum_PanCK_CD45"]]
nuc_ch = np.clip(nuc_ch, a_min=0, a_max=np.quantile(nuc_ch, 0.999))
mem_ch = np.clip(mem_ch, a_min=0, a_max=np.quantile(mem_ch, 0.999))
fig, ax = plt.subplots(1, 3)
for a, im, ttl in zip(ax, (nuc_ch, mem_ch, mask.squeeze()), ("HH3 (nuclear)", "PanCK+CD45 (membrane)", "Segmentation")):
    a.imshow(im.T)
    a.set_title(ttl)

# Preprocessing (normalization, etc.)
# X, y = preprocess(img, mask, channels)

# Reformat data for model
model_input = convert_to_model_input(*preprocess(img, mask, channels))

# Run prediction
model_dir = Path("model")
celltypes = run_model(model_input, model_dir)

# Get total number of each cell type in prediction
ctidx = np.array(list(celltypes.keys())) + 1
categories, counts = np.unique(list(celltypes.values()), return_counts=True)

# Map categories to integers for viz
cat_to_int = {c: i for c, i in zip(categories, range(1, len(categories) + 1))}
ct = list(celltypes.values())
ctidx_to_ct = dict(zip(ctidx, ct))

# Map predictions to segmentation mask
lbl_img = np.zeros_like(mask, dtype=np.uint8)
cell_pixels = mask != 0
lbls = np.array([cat_to_int[ctidx_to_ct[val]] for val in tqdm(mask[cell_pixels])])
lbl_img[cell_pixels] = lbls

# Show segmentation mask
plt.figure()
plt.imshow(lbl_img.T, cmap="PuBuGn")
# Update colorbar with celltype prediction labels
cbar = plt.colorbar()
ticklabels = [                                                              
    f"{num}: {lbl} ({cts})" for num, (lbl, cts) in                          
    enumerate(zip([""] + list(cat_to_int), ["bgnd"] + counts.tolist()))     
]                                                                           
ticks = range(len(ticklabels))                                              
cbar.set_ticks(ticks, labels=ticklabels)

# Add cell mask outline to image
outline = np.zeros_like(mask, dtype=np.uint8)
outline[:, 1:][mask[:, :-1] != mask[:, 1:]] = 1
outline[1:, :][mask[:-1, :] != mask[1:, :]] = 1
plt.imshow(outline.T, cmap=plt.cm.gray, alpha=outline.T.astype(float))

plt.show()
