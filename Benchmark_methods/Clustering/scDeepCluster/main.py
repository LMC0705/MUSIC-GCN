#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 10:07:33 2022

@author: liuyan
"""
from time import time
import math, os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scDeepCluster import scDeepCluster
from single_cell_tools import *
import numpy as np
from sklearn import metrics
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize
class Args(object):
    def __init__(self):
        self.n_clusters = 0
        self.knn = 20
        self.resolution = 0.8
        self.select_genes = 0
        self.batch_size = 256
        self.data_file = 'deng.csv'
        self.maxiter = 2000
        self.pretrain_epochs = 300
        self.gamma = 1.
        self.sigma = 2.5
        self.update_interval = 1
        self.tol = 0.001
        self.ae_weights = None
        self.save_dir = 'results/scDeepCluster/'
        self.ae_weight_file = 'AE_weights.pth.tar'
        self.final_latent_file = 'final_latent_file.txt'
        self.predict_label_file = 'pred_labels.txt'
        self.device = 'cpu'
args = Args()
dataset = load_data_origin_data(self.data_file,scaling=True)

x = dataset.x
y=dataset.y
# y is the ground truth labels for evaluating clustering performance
# If not existing, we skip calculating the clustering performance metrics (e.g. NMI ARI)
if 'Y' in data_mat:
    y = np.array(data_mat['Y'])
else:
    y = None
data_mat.close()

if args.select_genes > 0:
    importantGenes = geneSelection(x, n=args.select_genes, plot=False)
    x = x[:, importantGenes]

# preprocessing scRNA-seq read counts matrix
adata = sc.AnnData(x)
if y is not None:
    adata.obs['Group'] = y

adata = read_dataset(adata,
                 transpose=False,
                 test_split=False,
                 copy=True)

adata = normalize(adata,
                  size_factors=True,
                  normalize_input=True,
                  logtrans_input=True)

input_size = adata.n_vars

print(args)

print(adata.X.shape)
if y is not None:
    print(y.shape)
    
model = scDeepCluster(input_dim=adata.n_vars, z_dim=32, 
            encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=args.sigma, gamma=args.gamma, device=args.device)

print(str(model))
t0 = time()
if args.ae_weights is None:
    model.pretrain_autoencoder(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, 
                            batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
else:
    if os.path.isfile(args.ae_weights):
        print("==> loading checkpoint '{}'".format(args.ae_weights))
        checkpoint = torch.load(args.ae_weights)
        model.load_state_dict(checkpoint['ae_state_dict'])
    else:
        print("==> no checkpoint found at '{}'".format(args.ae_weights))
        raise ValueError

print('Pretraining time: %d seconds.' % int(time() - t0))

if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

if args.n_clusters > 0:
    y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, n_clusters=args.n_clusters, init_centroid=None, 
                y_pred_init=None, y=y, batch_size=args.batch_size, num_epochs=args.maxiter, update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
else:
    ### estimate number of clusters by Louvain algorithm on the autoencoder latent representations
    pretrain_latent = model.encodeBatch(torch.tensor(adata.X)).cpu().numpy()
    adata_latent = sc.AnnData(pretrain_latent)
    sc.pp.neighbors(adata_latent, n_neighbors=args.knn, use_rep="X")
    sc.tl.louvain(adata_latent, resolution=args.resolution)
    y_pred_init = np.asarray(adata_latent.obs['louvain'],dtype=int)
    features = pd.DataFrame(adata_latent.X,index=np.arange(0,adata_latent.n_obs))
    Group = pd.Series(y_pred_init,index=np.arange(0,adata_latent.n_obs),name="Group")
    Mergefeature = pd.concat([features,Group],axis=1)
    cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
    n_clusters = cluster_centers.shape[0]
    print('Estimated number of clusters: ', n_clusters)
    y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, n_clusters=n_clusters, init_centroid=cluster_centers, 
                y_pred_init=y_pred_init, y=y, batch_size=args.batch_size, num_epochs=args.maxiter, update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)


print('Total time: %d seconds.' % int(time() - t0))

if y is not None:
    #    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print('Evaluating cells: NMI= %.4f, ARI= %.4f' % (nmi, ari))

final_latent = model.encodeBatch(torch.tensor(adata.X)).cpu().numpy()
np.savetxt(args.final_latent_file, final_latent, delimiter=",")
np.savetxt(args.predict_label_file, y_pred, delimiter=",", fmt="%i")


