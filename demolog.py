# IPython log file

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tff
img = tff.imread("data/DCIS_Risom_breast_example_MIBI.tif")
channels = np.loadtxt("data/channelnames.txt", dtype=str, delimiter="\t")
channels
get_ipython().system('sudo docker run -it --gpus all -v $PWD/data:/data vanvalenlab/deepcell-applications:latest-gpu mesmer --nuclear-image /data/DCIS_Risom_breast_example_MIBI.tif --nuclear-channel 27 --membrane-image /data/DCIS_Risom_breast_example_MIBI.tif --membrane-channel 44 --image-mpp 0.49 --output-name /data/mask.tif --compartment whole-cell')
get_ipython().run_line_magic('ls', 'data')
mask = tff.imread("data/mask.tif")
ch_to_idx = zip(channels, range(channels.shape[0]))
ch_to_idx
ch_to_idx = dict(zip(channels, range(channels.shape[0])))
ch_to_idx
nuc_ch = img[ch_to_idx["HH3"]]
mem_ch = img[ch_to_idx["sum_PanCK_CD45"]]
nuc_ch = np.clip(nuc_ch, a_min=0, a_max=np.quantile(nuc_ch, 0.999))
mem_ch = np.clip(mem_ch, a_min=0, a_max=np.quantile(mem_ch, 0.999))
fig, ax = plt.subplots(1, 3)
for a, im, ttl in zip(ax, (nuc_ch, mem_ch, mask.squeeze()), ("HH3 (nuclear)", "PanCK+CD45 (membrane)", "Segmentation")):
    a.imshow(im.T)
    a.set_title(ttl)
    
nuc_ch = img[..., ch_to_idx["HH3"]]
mem_ch = img[..., ch_to_idx["sum_PanCK_CD45"]]
nuc_ch = np.clip(nuc_ch, a_min=0, a_max=np.quantile(nuc_ch, 0.999))
mem_ch = np.clip(mem_ch, a_min=0, a_max=np.quantile(mem_ch, 0.999))
fig, ax = plt.subplots(1, 3)
for a, im, ttl in zip(ax, (nuc_ch, mem_ch, mask.squeeze()), ("HH3 (nuclear)", "PanCK+CD45 (membrane)", "Segmentation")):
    a.imshow(im.T)
    a.set_title(ttl)
    
img
orig_img = img
orig_img.shape
model_dir = Path("model/saved_model")
from pathlib import Path
model_dir = Path("model/saved_model")
config_path = Path("model/config.yaml")
with open(config_path, "r") as fh:                                          
    model_config = yaml.load(fh, yaml.Loader)                               
master_channel_lst = model_config["channels"]                               
master_cell_types = np.asarray(model_config["cell_types"])
import yaml
with open(config_path, "r") as fh:                                          
    model_config = yaml.load(fh, yaml.Loader)                               
master_channel_lst = model_config["channels"]                               
master_cell_types = np.asarray(model_config["cell_types"])
master_channels = {ch.upper(): ch for ch in master_channel_lst}
master_channels
channel_lst, img = [], []                                                   
for idx, ch in enumerate(ch_names):                                         
    key = ch.upper()                                                        
    if key in master_channels:                                              
        channel_lst.append(master_channels[key])                            
        img.append(orig_img[..., idx].T)                                    
multiplex_img = np.asarray(img)
ch_names
channels
ch_names = channels
channel_lst, img = [], []                                                   
for idx, ch in enumerate(ch_names):                                         
    key = ch.upper()                                                        
    if key in master_channels:                                              
        channel_lst.append(master_channels[key])                            
        img.append(orig_img[..., idx].T)                                    
multiplex_img = np.asarray(img)
set(master_channels) & {ch.upper() for ch in ch_names}
multiplex_img.shape
assert len(marker_intersection) == multiplex_img.shape[0]
marker_intersection = set(master_channels) & {ch.upper() for ch in ch_names}
assert len(marker_intersection) == multiplex_img.shape[0]
class_X = multiplex_img.T.astype(np.float32)
class_X.shape
assert not set(channel_lst) - set(master_channel_lst)
assert len(channel_lst) == class_X.shape[-1]
def histogram_normalization(image, kernel_size=None):                           
    """                                                                         
    Pre-process images using Contrast Limited Adaptive                          
    Histogram Equalization (CLAHE).                                             
    If one of the inputs is a constant-value array, it will                     
    be normalized as an array of all zeros of the same shape.                   
    Args:                                                                       
        image (numpy.array): numpy array of phase image data with shape         
            (H, W, C). Note there is no batch index here.                       
        kernel_size (integer): Size of kernel for CLAHE,                        
            defaults to 1/8 of image size.                                      
    Returns:                                                                    
        numpy.array: Pre-processed image data with dtype float32.               
    """                                                                         
    image = image.astype("float32")                                             
    assert len(image.shape) == 3                                                
                                                                                
    for channel in range(image.shape[-1]):                                      
        X = image[..., channel]                                                 
        sample_value = X[(0,) * X.ndim]                                         
        if (X == sample_value).all():                                           
            image[..., channel] = np.zeros_like(X)                              
            continue                                                            
                                                                                
        X = rescale_intensity(X, out_range=(0.0, 1.0))                          
        X = equalize_adapthist(X, kernel_size=kernel_size)                      
        image[..., channel] = X                                                 
    return image
    
X = histogram_normalization(class_X, kernel_size=kernel_size)
kernel_size = 128
X = histogram_normalization(class_X, kernel_size=kernel_size)
def pad_cell(X: np.ndarray, y: np.ndarray,  crop_size: int):                    
    delta = crop_size // 2                                                      
    X = np.pad(X, ((delta, delta), (delta, delta), (0,0)))                      
    y = np.pad(y, ((delta, delta), (delta, delta)))                             
    return X, y
    
from skimage.exposure import rescale_intensity
def pad_cell(X: np.ndarray, y: np.ndarray,  crop_size: int):                    
    delta = crop_size // 2                                                      
    X = np.pad(X, ((delta, delta), (delta, delta), (0,0)))                      
    y = np.pad(y, ((delta, delta), (delta, delta)))                             
    return X, y
    
X = histogram_normalization(class_X, kernel_size=kernel_size)
from skimage.exposure import rescale_intensity, equalize_adapthist
X = histogram_normalization(class_X, kernel_size=kernel_size)
X.shape
def pad_cell(X: np.ndarray, y: np.ndarray,  crop_size: int):                    
    delta = crop_size // 2                                                      
    X = np.pad(X, ((delta, delta), (delta, delta), (0,0)))                      
    y = np.pad(y, ((delta, delta), (delta, delta)))                             
    return X, y
    
crop_size = 64
X, y = pad_cell(X, mask, crop_size)
mask.shape
mask = tff.imread("data/mask.tif").squeeze()
X, y = pad_cell(X, mask, crop_size)
X.shape
y.shape
get_ipython().run_line_magic('edit', '')
inp = convert_to_model_input(X, y)
from skimage.measure import regionprops
inp = convert_to_model_input(X, y)
from tqdm import tqdm
inp = convert_to_model_input(X, y)
def get_crop_box(centroid, delta):                                              
    minr = int(centroid[0]) - delta                                             
    maxr = int(centroid[0]) + delta                                             
    minc = int(centroid[1]) - delta                                             
    maxc = int(centroid[1]) + delta                                             
    return np.array([minr, minc, maxr, maxc])
    
inp = convert_to_model_input(X, y)
def get_neighbor_masks(mask, cbox, cell_idx):                                   
    """Returns binary masks of a cell and its neighbors. This function expects padding around
    the edges, and will throw an error if you hit a wrap around."""             
    minr, minc, maxr, maxc = cbox                                               
    assert np.issubdtype(mask.dtype, np.integer) and isinstance(cell_idx, int)  
                                                                                
    cell_view = mask[minr:maxr, minc:maxc]                                      
    binmask_cell = (cell_view == cell_idx).astype(np.int32)                     
                                                                                
    binmask_neighbors = (cell_view != cell_idx).astype(np.int32) * (cell_view != 0).astype(
        np.int32                                                                
    )                                                                           
    return np.stack([binmask_cell, binmask_neighbors])
    
inp = convert_to_model_input(X, y)
import tensorflow as tf
inp = convert_to_model_input(X, y)
inp = convert_to_model_input(X, y)
get_ipython().run_line_magic('edit', '_70')
inp = convert_to_model_input(X, y)
get_ipython().run_line_magic('edit', '_84')
get_ipython().run_line_magic('edit', '_70')
inp = convert_to_model_input(X, y)
get_ipython().run_line_magic('edit', '_70')
inp = convert_to_model_input(X, y)
get_ipython().run_line_magic('edit', '_70')
inp = convert_to_model_input(X, y)
def run_model(model_input, model_dir, output_categories):                       
                                                                                
    ctm = load_model(model_dir, compile=False)                                  
                                                                                
    logits = ctm.predict(model_input)                                           
                                                                                
    pred_idx = np.argmax(sp.special.softmax(logits, axis=1), axis=1)            
    cell_type_predictions = output_categories[1:][pred_idx]                     
                                                                                
    return dict(enumerate(cell_type_predictions))
    
run_model(inp, model_dir, master_cell_types)
from tensorflow.keras.models import load_model
run_model(inp, model_dir, master_cell_types)
import scipy as sp
run_model(inp, model_dir, master_cell_types)
get_ipython().run_line_magic('pdb', '')
run_model(inp, model_dir, master_cell_types)
ctm = load_model(model_dir, compile=False)
logits = ctm.predict(inp)
logits
len(logits)
def run_model(model_input, model_dir, output_categories):                       
                                                                                
    ctm = load_model(model_dir, compile=False)                                  
                                                                                
    logits = ctm.predict(model_input)[0]                                        
                                                                                
    pred_idx = np.argmax(sp.special.softmax(logits, axis=1), axis=1)            
    cell_type_predictions = output_categories[1:][pred_idx]                     
                                                                                
    return dict(enumerate(cell_type_predictions))
    
run_model(inp, model_dir, master_cell_types)
mask.shape
mask.max()
celltypes = run_model(inp, model_dir, master_cell_types)
ctidx = np.array(list(celltypes.keys()))
ctids
get_ipython().run_line_magic('pdb', '')
ctidx
ctidx = np.array(list(celltypes.keys())) + 1
categories, counts = np.unique(list(celltypes.values()), return_counts=True)
categories
counts
cat_to_int = {c: i for c, i in zip(categories, range(1, len(categories) + 1))}
lbl_img = np.zeros_like(mask, dtype=np.uint8)
cell_pixels = mask != 0
lbls = np.array([cat_to_int[ctidx_to_ct[val]] for val in tqdm(mask[cell_pixels])])
ctidx
ct = list(celltypes.values())
ct
ctidx
ctidx_to_ct = dict(zip(ctidx, ct))
lbls = np.array([cat_to_int[ctidx_to_ct[val]] for val in tqdm(mask[cell_pixels])])
lbl_img[cell_pixels] = lbls
plt.imwho(lbl_img, cmap="PuBuGn")
plt.imshow(lbl_img, cmap="PuBuGn")
cbar = plt.colorbar()
ticklabels = [                                                              
    f"{num}: {lbl} ({cts})" for num, (lbl, cts) in                          
    enumerate(zip([""] + list(cat_to_int), ["bgnd"] + counts.tolist()))     
]                                                                           
ticks = range(len(ticklabels))                                              
cbar.set_ticks(ticks, labels=ticklabels)
mask.shape
l = mask[1:] - mask[:-1]
binmask = (mask > 0).astype(np.int32)
l = mask[1:] - mask[:-1]
l = binmask[1:] - binmask[:-1]
r = binmask[:, :-1] - binmask[:, 1:]
u = binmask[:-1, :] - binmask[1:, :]
d = binmask[1:, :] - binmask[:-1, :]
l.shape
u.shape
r.shape
l = binmask[:, 1:] - binmask[:, :-1]
l.shape
r.shape
u.shape
d.shape
outline = np.zeros_like(mask, dtype=np.uint8)
outline[:, :-1][l < 0] = 1
outline[:, 1:][r < 0] = 1
outline[:-1, :][u < 0] = 1
outline[1:, :][d < 0] = 1
outline
plt.figure();
plt.imshow(outline)
plt.figure(); plt.imshow(l)
binmask.shape
get_ipython().system('sudo docker run -it --gpus all -v $PWD/data:/data vanvalenlab/deepcell-applications:latest-gpu mesmer --nuclear-image /data/DCIS_Risom_breast_example_MIBI.tif --nuclear-channel 27 --membrane-image /data/DCIS_Risom_breast_example_MIBI.tif --membrane-channel 44 --image-mpp 0.5 --output-name /data/mask.tif --compartment whole-cell')
get_ipython().run_line_magic('mv', 'data/mask.tif data/mask.tif_49')
get_ipython().system('sudo docker run -it --gpus all -v $PWD/data:/data vanvalenlab/deepcell-applications:latest-gpu mesmer --nuclear-image /data/DCIS_Risom_breast_example_MIBI.tif --nuclear-channel 27 --membrane-image /data/DCIS_Risom_breast_example_MIBI.tif --membrane-channel 44 --image-mpp 0.5 --output-name /data/mask.tif --compartment whole-cell')
mask = tff.imread("data/mask.tif").squeeze()
mask.shape
plt.figure(); plt.imshow(mask)
ol2 = np.zeros_like(mask, dtype=np.uint8)
ol2.shape
ol2[:, 1:][mask[:, :-1] != mask[:, 1:]] = 1
plt.figure(); plt.imshow(ol2)
ol2[1:, :][mask[:-1, :] != mask[1:, :]] = 1
plt.figure(); plt.imshow(ol2)
plt.figure(); plt.imshow(ol2, cmap=cm.gray)
plt.figure(); plt.imshow(ol2, cmap=cm.gray_4)
plt.figure(); plt.imshow(ol2, cmap=cm.gray_r)
plt.figure(); plt.imshow(ol2, cmap=cm.gray_r, alpha=ol2)
plt.figure();
plt.imshow(lbl_img, cmap="PuBuGn")
plt.imshow(ol2, cmap=cm.gray_r)
plt.figure();
plt.imshow(lbl_img, cmap="PuBuGn")
plt.imshow(ol2, cmap=cm.gray_r, alpha=np.abs(1-ol2))
plt.figure();
plt.imshow(lbl_img, cmap="PuBuGn")
plt.imshow(ol2, cmap=cm.gray_r, alpha=ol2)
plt.figure();
plt.imshow(lbl_img, cmap="PuBuGn")
plt.imshow(ol2, cmap=cm.gray, alpha=ol2)
plt.figure();
plt.imshow(lbl_img, cmap="PuBuGn")
plt.imshow(ol2, cmap=cm.gray)
plt.figure();
plt.imshow(lbl_img, cmap="PuBuGn")
plt.imshow(ol2, cmap=cm.gray, alpha=ol2.astype(float))
get_ipython().run_line_magic('logstart', 'demolog.py')
exit()
