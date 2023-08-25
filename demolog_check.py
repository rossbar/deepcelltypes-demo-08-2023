# IPython log file

get_ipython().run_line_magic('run', 'demolog.py')
data = np.load("holdout_Point2331_pt1105_30985.npz")
data = np.load("holdout_Point2331_pt1105_30985.npz", allow_pickle=True)
data
data.files
true_cell_types = data["cell_types"]
true_cell_types
meta_fname = "Point2331_pt1105_30985.npz.dvc"
import yaml
with open(meta_fname, "r") as fh
with open(meta_fname, "r") as fh:
    metadata = yaml.load(fh, yaml.Loader)
    
metadata
metadata["meta"]["file_contents"]["cell_types"]["mapper"]
true_ct_mapper = metadata["meta"]["file_contents"]["cell_types"]["mapper"]
true_cell_types
celltypes
predicted_celltypes = celltypes
true_cell_types
true_cell_types = data["cell_types"][0]
true_cell_types = data["cell_types"][:]
true_cell_types = data["cell_types"][]
true_cell_types = data["cell_types"][()]
true_cell_types
y = data["y"]
y.shape
y.max(axis=-1)
y.max(axis=(0, 1, 2))
y = data["y"][0, ..., 0]
y.max()
true_cell_types
true_ct_mapper
tct_lbls = {k : true_ct_mapper[v] for k, v in true_cell_types.items()}
tct_lbls
true_categories, true_counts = np.unique(list(tct_lbls.values()), return_counts=True)
true_categories
true_counts
counts
categories
tcat_to_int  = {c: i for c, i in zip(true_categories, range(1, len(true_categories) + 1))}
y
y.shape
tlbl_img = np.zeros_like(mask, dtype=np.uint8)
cell_pixels = y != 0
tlbl_img[cell_pixels] = np.array(

0

)
cat_to_int
tcat_to_int
ctidx_to_ct
tct_lbls
tlbls = np.array([tcat_to_int[tct_lbls[val]] for val in y[cell_pixels]])
tlbls.shape
tlbl_img[cell_pixels] = tlbls
plt.figure(); plt.imshow(tlbl_img.T, cmap="PuBuGn")
cbar = plt.colorbar()
ticklabels = [                                                                  
    f"{num}: {lbl} ({cts})" for num, (lbl, cts) in                              
    enumerate(zip([""] + list(tcat_to_int), ["bgnd"] + true_counts.tolist()))         
]
ticks = range(len(ticklabels))                                                  
cbar.set_ticks(ticks, labels=ticklabels)
get_ipython().run_line_magic('logstart', 'demolog_check.py')
exit()
