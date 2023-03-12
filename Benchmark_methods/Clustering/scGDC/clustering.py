import scipy.io as scio
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from munkres import Munkres
from sklearn import cluster
from utils import load_graph, load_data_origin_data
def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate
def best_map(L1, L2):
    # L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

file_path = ""
dataset = load_data_origin_data(file_path,scaling=True)
data=dataset.x
Label=dataset.y
print(data)
print(type(data))
data[data>=0.1] = 1
data[data<0.1] = 0
spectral = cluster.SpectralClustering(n_clusters=4, eigen_solver='arpack', affinity='precomputed',
                                      assign_labels='discretize')
# 按常理来说 此处
spectral.fit(data)
# grp 为返回的聚类标签
grp = spectral.fit_predict(data)
y_x = grp
missrate_x = err_rate(Label, y_x)
# ari
ari_x = adjusted_rand_score(Label, y_x)
nmi_x = normalized_mutual_info_score(Label, y_x)
acc_x = 1 - missrate_x
print("ari_x,nmi_x,acc_x",ari_x,nmi_x,acc_x)
