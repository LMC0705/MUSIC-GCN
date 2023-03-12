library(Seurat)
library(SingleCellExperiment)
library(SC3)
library(scater)
library(SeuratObject)
string1 <- "./data/"
dataset_name <- "data.csv"
data_url <- paste0(string1, dataset_name)
data<-read.csv(data_url,header = FALSE)
data<-t(data)
colnames(data) <- paste0("Gene", seq(ncol(data)))
cluster_data <- CreateSeuratObject(counts =data)

sce <- as.SingleCellExperiment(cluster_data)
library(future)
# check the current active plan
plan()
# change the current plan to access parallelization
plan("multiprocess", workers = 4)
plan()
sce <- sc3(sce, ks = 2:10, biology = TRUE) # too time

sc3_plot_expression(
  sce, k = 4, 
  show_pdata = c(
    "cell_type1", 
    "log10_total_features",
    "sc3_3_clusters", 
    "sc3_3_log2_outlier_score"
  )
)