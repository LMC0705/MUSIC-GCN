# MUSIC-GCN

A novel multi-tasking pipeline for the analysis of single-cell transcriptomic data using residual graph convolution networks.
# Introduction

Single-cell transcriptomics (scRNA-seq) is a powerful approach for the characterization of gene transcription at a cellular resolution. This approach requires efficient informatic pipelines to undertake essential tasks including clustering, dimensionality reduction, imputation and denoising. Currently, most such pipelines undertake individual tasks without considering the inter-dependence between or among these tasks, which has intrinsic limitations. Here, we present an advanced pipeline, called MUSIC-GCN, which employs graph convolutional neural networking and autoencoder to perform multi-task scRNA-seq data analysis. The rationale is that multiple related tasks can be carried out simultaneously to enable enhanced learning and more effective representations through the ‘sharing of knowledge’ regarding individual tasks. ![image](https://github.com/LMC0705/MUSIC-GCN/blob/main/log_image.png)

# Requirement:
```console
scanpy 1.7.2
scikit-learn 0.24.2
torch 1.10.0
python 3.6.13
```
# Datasets
The Immune cells dataset is downloaded from: https://www.tissueimmunecellatlas.org/

The packer dataset is downloaded from: https://depts.washington.edu:/trapnell-lab/software/monocle3/celegans/data/packer_embryo_expression.rds

The raw datasets are downloaded from: https://hemberg-lab.github.io/scRNA.seq.datasets/

# Usage

```console
import scanpy as sc
import numpy as np
from MUSIC-GCN import train_musigcn 
adata=sc.read_h5ad("/home/liuyan/code/singlecell/MUSIC-GCN/data/global.h5ad")
true_label=adata.obs["Manually_curated_celltype"]
cell_selected=list(np.random.randint(0,adata.shape[0],40000))
new_adata=adata.X[cell_selected]
Y=list(true_label[cell_selected])
new_adata=sc.AnnData(new_adata)
sc.pp.normalize_per_cell(new_adata, counts_per_cell_after=1e4)  #This type of regularization can also be used in MUSIC-GCN
sc.pp.log1p(new_adata)
sc.pp.highly_variable_genes(
            new_adata,n_top_genes=2000)
highly_variable_genes = new_adata.var["highly_variable"]
X=new_adata[:,highly_variable_genes].X.toarray()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(Y)
y=le.transform(Y)
n_clusters =len(list(set(list(y))))
#n_clusters =？ # 
n_input = 2000
train_musicgcn(X,y,X)
```
# Connect

If you have any questions, please contact yanliu@njust.edu.cn
