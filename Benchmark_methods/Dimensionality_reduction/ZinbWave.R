install.packages("BiocManager")
BiocManager::install("zinbwave")

library(zinbwave)
library(scRNAseq)
library(matrixStats)
library(magrittr)
library(ggplot2)
library(biomaRt)
library(SingleCellExperiment)

mydata <- read.csv("/path/to/mydata.csv", header = TRUE, row.names = 1)
sce <- SingleCellExperiment(assays = list(counts = mydata))
sce <- rowSums(assay(sce)>5)>5
assay(sce) %>% log1p %>% rowVars -> vars
names(vars) <- rownames(sce)
vars <- sort(vars, decreasing = TRUE)
zinb <- zinbFit(sce, K=32, epsilon=1000)
W <- getW(zinb)
W=as.data.frame(W)

write.csv(W,"ZinbWave_W.csv")
