
from utils import load_graph, load_data_origin_data
import scanpy as sc
import argparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluation import eva,nmi_ari_eva
from sklearn.metrics import silhouette_score
adata=sc.read_csv("datasets/Filtered_TM_data.csv")
import pandas as pd
LabelsPath="datasets/Labels.csv"
labels=pd.read_csv(LabelsPath, header=0,index_col=None, sep=',')
labels=labels.values
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(labels)
y=le.transform(labels)
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(
        adata,n_top_genes=2000)
highly_variable_genes = adata.var["highly_variable"]
adata = adata[:, highly_variable_genes]
x=adata.X
#############################################
pca = PCA(n_components=32)
new_pca = pca.fit_transform(x)
n_clusters=10 #based on you
kmeans = KMeans(n_clusters= n_clusters, n_init=24).fit(new_pca)
eva(y, kmeans.labels_)
silhouette=silhouette_score(new_pca,y)
print ("PCA_silhouette_score:",silhouette)
from sklearn.decomposition import KernelPCA
#核函数同样采用的是高斯径向基函数：
rbf_pca = KernelPCA(n_components = 32, kernel="rbf", gamma=0.004)
X_reduced = rbf_pca.fit_transform(x)
silhouette=silhouette_score(X_reduced,y)
print ("kPCA_silhouette_score:",silhouette)