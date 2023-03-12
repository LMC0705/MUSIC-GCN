#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 11:42:20 2023

@author: liuyan
"""
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns 
import anndata as ad
from evaluation import nmi_ari_eva
data=np.load("trj/MUSIC_imputation.npy")
import numpy 
import numpy 
adata=ad.AnnData(data)
adata.write_csvs("MUSIC_imputation.csv")







cell_metadata=pd.read_csv("trj/cell_metadata .csv")
cell_type=cell_metadata["cell.type"]
# gene_metadata=pd.read_csv("trj/gene_annotation.csv")
# genes_list=gene_metadata["gene_short_name"]
# data.columns=genes_list
adata=ad.AnnData(data)
adata.obs["cell_type"]=np.array(cell_type)
adata.obs["time"]=np.array(cell_metadata["time.point"])
# adata.write_h5ad("trj/packer_embryo.h5ad")
adata=ad.AnnData(data)
adata.obs["cell_type"]=np.array(cell_type)
adata.obs["time"]=np.array(cell_metadata["time.point"])
# adata.write_h5ad("trj/packer_embryo.h5ad")
# sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
# sc.pp.log1p(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
sc.tl.leiden(adata)
pred_scanpy=adata.obs["leiden"]
y=adata.obs["cell_type"]
nmi_ari_eva(list(y), list(pred_scanpy))

sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
sc.tl.leiden(adata)

sc.tl.draw_graph(adata)
sc.pl.draw_graph(adata, color='leiden', legend_loc='on data')

sc.tl.umap(adata)
sc.pl.umap(adata, color=['leiden'])

sc.tl.tsne(adata)
sc.pl.tsne(adata, color=['leiden'])

sc.tl.diffmap(adata)

sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_diffmap')
sc.tl.draw_graph(adata)
sc.pl.draw_graph(adata, color='cell_type', legend_loc='on data')

sc.tl.paga(adata, groups='cell_type')

sc.pl.paga(adata, threshold=0.03, show=False)

sc.tl.draw_graph(adata, init_pos='cell_type')
sc.pl.draw_graph(adata, color=['leiden','MEGF6', 'MT-CO2', 'AKR7A3'], legend_loc='on data')