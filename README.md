# DMR Project
Sapir Kontente and Amit Shomer


This is Implementation of the paper [Deep Mesh Reconstruction from Single RGB Images via Topology Modification Networks](https://arxiv.org/abs/1909.00321). 



# Installation and packges

After clone the repo install the environment:
```shelll
conda env create -f dmr_project.yml
```

Install the following [Chamfer distance](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) or install other implementation and change the code. 
Follow the authors instructions to install the Chmafer distance correctly. 
The git should be in the following path : ``` Project_DMR/ChamferDistancePytorch```

# Data 

Download the following data that the authours released which include the chair category only. 

* [The ShapeNet point clouds (*.mat)](https://drive.google.com/file/d/1Z0d8W4PJnWIoCqt1jM4ziSFd1tgBUHa6/view?usp=sharing) go in ``` data/customShapeNet_mat```
* [the rendered views (*.png)](https://drive.google.com/file/d/1eu2-Qm6T9AhjDkKP6IY-G__ti1N37VBr/view?usp=sharing) go in ``` data/ShapeNetRendering```

# Pre-trained Model
Here are the download [links for our model checkpoint](https://drive.google.com/drive/folders/1_Y7jKgiTt3rxpmcuBAmdq6fLEBuM4-FE?usp=share_link) 
The entire log folder should located as follow : ``` Project_DMR/log```

# Our mesh results 
Our [mesh results can be found at the following link](https://drive.google.com/drive/folders/1gOPv8FlQ6_IqXQ0pwZkdXrPecgM9uFaN?usp=sharing)

# Test the model 
```
python plot_mesh.py
```
The meshes will save at ```log/plot_mesh_folder```

# Train the model 
The following command will train all sub-models one after the other
```
bash train.sh
```


## Acknowledgment
Some of the code released by the authors in the [original repo](https://github.com/jnypan/TMNet) is been used. We thank the authors for sharing their code.