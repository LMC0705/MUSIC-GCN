#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:39:50 2023

@author: liuyan
"""
import os

import matplotlib.pyplot as plt
import scanpy as sc
import scvi
from evaluation import eva,nmi_ari_eva
from utils import load_graph, load_data_origin_data
import numpy as np
dataset_name="deng.csv"
file_path = "./datasets/" + dataset_name
dataset = load_data_origin_data(file_path, "csv",scaling=True)
true_cluster=dataset.y

adata=sc.AnnData(dataset.x)
adata.layers["counts"] = adata.X.copy() 
scvi.model.LinearSCVI.setup_anndata(adata, layer="counts")

model = scvi.model.LinearSCVI(adata, n_latent=10)
model.train(max_epochs=250, plan_kwargs={"lr": 5e-3}, check_val_every_n_epoch=10)
Z_hat = model.get_latent_representation()
for i, z in enumerate(Z_hat.T):
    adata.obs[f"Z_{i}"] = z
