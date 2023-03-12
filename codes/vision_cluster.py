#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:40:27 2021

@author: liuyan
"""


from sklearn.manifold import TSNE

import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE


import numpy as np
import pandas as pd

def draw_gcn_batch_cluster(train,y_pred,batchy,clusters_number,name,nz):
    tsne = TSNE()
    train=tsne.fit_transform(train)
    c = ['b','c','y','r','g','m','gold','tan',"deepskyblue"]
    colors = [c[i%len(c)] for i in range(clusters_number)]
    fig, ax = plt.subplots()
    
    types = []
    
    for i, color in enumerate(colors):
        need_idx = np.where(y_pred==i)[0]
        ax.scatter(train[need_idx,1],train[need_idx,0], c=color, label=i,s=1)
    fig_name="./newcluster_result/"+name+"gcn_k_"+str(nz)+".eps"
    plt.savefig(fig_name,dpi=500)
    fig1, ax1 = plt.subplots()
    batch_c = ["grey","royalblue","lime","crimson"]
    for i, color in enumerate(batch_c):
        need_idx = np.where(batchy==i)[0]
        # ax.scatter(train[need_idx,1],train[need_idx,0], c=color, label=i,s=1)
        ax1.scatter(train[need_idx,1],train[need_idx,0], c=color, label=i,s=1)
    # legend = ax.legend(loc='upper right')
    
    fig_name1="./newcluster_result/"+name+"gcn_k_batch_"+str(nz)+".eps"

    plt.savefig(fig_name1,dpi=500,transparent=True)
    
    

def draw_restruct_cluster(train,y_pred,clusters_number,name,nz):
    tsne = TSNE()
    train=tsne.fit_transform(train)
    c = ['b','c','y','r','g','m','gold','tan',"deepskyblue"]
    colors = [c[i%len(c)] for i in range(clusters_number)]
    fig, ax = plt.subplots()
    types = []
    
    for i, color in enumerate(colors):
        need_idx = np.where(y_pred==i)[0]
        ax.scatter(train[need_idx,1],train[need_idx,0], c=color, label=i,s=1)
    # legend = ax.legend(loc='upper right')
    fig_name="./newcluster_result/"+name+"restruct_k"+str(nz)+".eps"
    plt.savefig(fig_name,dpi=500,transparent=True)
    
    
def draw_gcn_cluster(train,y_pred,clusters_number,name,nz):
    tsne = TSNE()
    train=tsne.fit_transform(train)
    c = ['b','c','y','r','g','m','gold','tan',"deepskyblue"]
    colors = [c[i%len(c)] for i in range(clusters_number)]
    fig, ax = plt.subplots()
    types = []
    
    for i, color in enumerate(colors):
        need_idx = np.where(y_pred==i)[0]
        ax.scatter(train[need_idx,1],train[need_idx,0], c=color, label=i,s=1)
    # legend = ax.legend(loc='upper right')
    fig_name="./newcluster_result/"+name+"gcn_k_"+str(nz)+".eps"
    plt.savefig(fig_name,dpi=500,transparent=True)
    
def draw_latent_cluster(train,y_pred,clusters_number,name,nz):
    # tsne = TSNE()
    # train=tsne.fit_transform(train)
    c = ['b','c','y','r','g','m','gold','tan',"deepskyblue"]
    colors = [c[i%len(c)] for i in range(clusters_number)]
    fig, ax = plt.subplots()
    types = []
    
    for i, color in enumerate(colors):
        need_idx = np.where(y_pred==i)[0]
        ax.scatter(train[need_idx,1],train[need_idx,0], c=color, label=i,s=1)
    # legend = ax.legend(loc='upper right')
    fig_name="./newcluster_result/"+name+"latent_k_"+str(nz)+".eps"
    plt.savefig(fig_name,dpi=500,transparent=True)
  
def draw_desc(train,y_pred,clusters_number,name):
    tsne = TSNE()
    train=tsne.fit_transform(train)
    c = ['b','c','y','r','g','m','gold','tan',"deepskyblue"]
    colors = [c[i%len(c)] for i in range(clusters_number)]
    fig, ax = plt.subplots()
    types = []
    
    for i, color in enumerate(colors):
        need_idx = np.where(y_pred==i)[0]
        ax.scatter(train[need_idx,1],train[need_idx,0], c=color, label=i,s=1)
    # legend = ax.legend(loc='upper right')
    fig_name="./newcluster_result/"+name+"latent_k_"+str("desc")+".eps"
    plt.savefig(fig_name,dpi=500,transparent=True)
    
    
def draw_VASC(train,y_pred,clusters_number,name):
    tsne = TSNE()
    train=tsne.fit_transform(train)
    c = ['b','c','y','r','g','m','gold','tan',"deepskyblue"]
    colors = [c[i%len(c)] for i in range(clusters_number)]
    fig, ax = plt.subplots()
    types = []
    
    for i, color in enumerate(colors):
        need_idx = np.where(y_pred==i)[0]
        ax.scatter(train[need_idx,1],train[need_idx,0], c=color, label=i,s=1)
    # legend = ax.legend(loc='upper right')
    fig_name="./newcluster_result/"+name+"latent_k_"+str("VASC")+".eps"
    plt.savefig(fig_name,dpi=500,transparent=True)