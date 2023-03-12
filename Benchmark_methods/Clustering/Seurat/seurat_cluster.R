library(Seurat)
library(SeuratObject)
string1 <- "./real_Data/"
dataset_name <- "darmanis.csv"
data_url <- paste0(string1, dataset_name)
data<-read.csv(data_url,header = FALSE)
data<-t(data)
colnames(data) <- paste0("Gene", seq(ncol(data)))
cluster_data <- CreateSeuratObject(counts =data)
cluster_data=FindVariableFeatures(cluster_data) 
all.genes <- rownames(cluster_data)
cluster_data<- ScaleData(cluster_data,features = all.genes)
cluster_data<- RunPCA(cluster_data)
cluster_data<- FindNeighbors(cluster_data, dims = 1:10)
#聚类，包含设置下游聚类的“间隔尺度”的分辨率参数resolution ，增加值会导致更多的聚类。
cluster_data<- FindClusters(cluster_data, resolution = 0.5)
#可以使用idents函数找到聚类情况：
#查看前5个细胞的聚类id
head(Idents(cluster_data))
cluster_result=Idents(cluster_data)
write.csv(as.character(cluster_result), file = dataset_name)

