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
