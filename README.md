## Dependencies
* Python 3 (In our experiments, we used python 3.8.8)
* Blender 2.82a

### others
```
albumentations==1.3.0
einops==0.6.0
huggingface-hub==0.12.1
imageio==2.22.4
kornia==0.6.9
Levenshtein==0.20.9
numpy==1.22.3
opencv-python==4.5.5.64
pandas==1.4.2
scikit-image==0.19.3
scikit-learn==1.0.2
scipy==1.8.0
timm==0.4.12
torch==1.11.0+cu113
torchaudio==0.11.0+cu113
torchvision==0.12.0+cu113
tqdm==4.64.0
transformers==4.26.1
typing_extensions==4.2.0
wandb==0.13.5
```
To ensure no omissions, `./pip_freeze.txt` shows the output when `pip freeze` is executed in our environment.


## Making SynDocDS  

### Preparation
1. [DDI-100](https://github.com/machine-intelligence-laboratory/DDI-100/tree/master/dataset): Download documents consisting of texts and figures. Then, Combine texts under the `orig_texts` folder and figures under the `orig_backgrounds` folder. Put obtained document images under `./materials/docs/`.
2. [Doc3D](https://github.com/cvlab-stonybrook/doc3D-dataset): Download paper meshes (.obj). Put these under `./materials/paper_meshes/`.
3. [SVBRDFs dataset (Deschaintre+, 2018)](https://repo-sam.inria.fr/fungraph/deep-materials/): Download the dataset (85GB zipped) and crop a subset of sand and fabric normal map images. Then, put these under `./materials/selected_normal_maps/`.
4. [ShapeNet](https://shapenet.org/): Download ShapeNetCore.v2 and unzip it. Then, put it under `./materials/objects/`.
5. [Laval Indoor HDR dataset](http://indoor.hdrdb.com/): Download panoramas and convert these `.png` files, then put these under `./materials/panoramas/indoor/`
6. [SUN360 dataset](https://vision.cs.princeton.edu/projects/2012/SUN360/data/): Contact the author, download panoramas, and convert these `.png` files. Then, put these under `./materials/panoramas/indoor/` and `./materials/panoramas/outdoor/`, respectively.
7. Execute `python3 utils/making_material_csv.py` in `src` directory.  

### NOTE
We initially put a few materials for placing checks and reproducibility checks.  
However, we will not provide these all materials, only downloaded by original links.

### Install Blender
We used blender 2.82a, and you should install some packages. An example installation script is below.  
```
wget https://download.blender.org/release/Blender2.82/blender-2.82-linux64.tar.xz
tar xvf blender-2.82-linux64.tar.xz  
export PATH=/home/ubuntu/local/blender-2.82-linux64/:$PATH  
wget https://bootstrap.pypa.io/get-pip.py  
/home/ubuntu/local/blender-2.82-linux64/2.82/python/bin/python3.7m get-pip.py  
/home/ubuntu/local/blender-2.82-linux64/2.82/python/bin/python3.7m -m pip install easydict  
/home/ubuntu/local/blender-2.82-linux64/2.82/python/bin/python3.7m -m pip install pillow  
/home/ubuntu/local/blender-2.82-linux64/2.82/python/bin/python3.7m -m pip install opencv-python  
/home/ubuntu/local/blender-2.82-linux64/2.82/python/bin/python3.7m -m pip install pandas  
/home/ubuntu/local/blender-2.82-linux64/2.82/python/bin/python3.7m -m pip install tqdm  
```


### Rendering
**!!!Every command should be executed in `src` directory!!!**
```
# rendering synthetic document images
blender -b --python utils/SynDocRenderer.py -- --rendering --save_path output
python3 utils/postprocess_SynDocDS.py
```

Then, the dataset, SynDocDS, is under the `./datasets/SynDocDS` and the corresponding csv file is under the `./datasets/csv/SynDocDS`.  

Brief descriptions are below.  
* `./src/utils/SynDocRenderer.py`: Rendering synthetic images. This mainly corresponds to `Section 3.1. Image Rendering` in our paper.  
* `./src/utils/postprocess_SynDocDS.py`: Providing post-processed shadow mattes, background color, and dataset csv. Augmenting shadow images corresponding to `Section 3.2. Enriching Shadow Images` in our paper.

***We will release this code and our SynDocDS dataset, which was used for training when our paper is public.***


## Real Dataset
1. [OSR dataset](https://github.com/BingshuCV/DocumentShadowRemoval): We used `Control_*` data, which have ground truth.
2. [Kligler's dataset](https://github.com/xuhangc/Document-Enhancement-using-Visibility-Detection)
3. [Jung's dataset](https://github.com/seungjun45/Water-Filling)  

Each image is placed in the `./datasets/[DATASETNAME]/shadowfree_image` and `./datasets/[DATASETNAME]/shadow_image` folders with the same name. In addition, the OSR dataset also contains shadow masks, which are placed in the `./datasets/[DATASETNAME]/shadow_mask` folder.  
To make shadow masks and background colors for real datasets, use `postprocess_real_dataset.ipynb`. Put target dataset name `NAME = DIR = [DATASETNAME]`.

Training/validation/test splits are specified in `./datasets/csv/[DATASETNAME]/` 

## Training  
```
python3 train.py configs/DSFN_SynDocDS/config.yaml
```
If you want to use W&B to display output images, losses, and evaluation scores, you can use `--use_wandb` option.
  
### Pre-trained model
Please download [here](https://drive.google.com/file/d/1j00v2CXvyMqWcu2GQrYSurmDX5yXm0AR/view?usp=drive_link).  

### FineTuning  
Please check an example config file for finetuning, `DSFN_SynDocDS_FT_[DATASETNAME]/config.yaml`.
```
*example
python3 train.py configs/DSFN_SynDocDS_FT_Kligler/config.yaml
```

## Evaluation
```
python3 evaluate.py configs/[CONFIG_NAME]/config.yaml --test_data [DATASET_NAME]

*example
python3 evaluate.py configs/DSFN_SynDocDS/config.yaml --test_data JungAll

python3 evaluate.py configs/DSFN_SynDocDS_FT_Jung/config.yaml --test_data Jung
```
For the test dataset name, please refer to `./src/libs/dataset_csv.py`  
The result images are under `./configs/[CONFIG_NAME]/` and evaluation score is saved as `./configs/[CONFIG_NAME]/evaluation_score.txt`  


## OCR
We used Tesseract for OCR. It should be installed following [this instruction](https://github.com/tesseract-ocr/tesseract#installing-tesseract).

We used parts of OSR dataset that have detectable texts.  
The images (the file name starts from these numbers) have detectable texts: `["003", "005", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015", "016", "024", "025", "026"]`  

The images (the file name starts from these numbers) are rotated 90 degrees to the right: `["001", "002", "017", "019", "020", "021", "023"]` (*NOTE*: These are actually not used.)

`ocr.ipynb` is a short code for calculating edit distances.

## About reproduction of SDSRD

We used the rendering code for SDSRD from [this repository](https://github.com/frank840306/BlenderRender).  
We carefully selected the WORKLOAD files by referring to commit logs, etc., to use a dataset as close as possible to those used in previous studies (Lin+, CVPR2020).
