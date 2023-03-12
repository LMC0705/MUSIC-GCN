library(monocle3)
expression_matrix <- readRDS(url("https://depts.washington.edu:/trapnell-lab/software/monocle3/celegans/data/packer_embryo_expression.rds"))
#saver_expression<-read.csv("/home/liuyan/code/singlecell/MUSIC-GCN/trj/packer_saver.csv")
cell_metadata <- readRDS(url("https://depts.washington.edu:/trapnell-lab/software/monocle3/celegans/data/packer_embryo_colData.rds"))
gene_annotation <- readRDS(url("https://depts.washington.edu:/trapnell-lab/software/monocle3/celegans/data/packer_embryo_rowData.rds"))

data<-read.csv("/home/liuyan/code/singlecell/MUSIC-GCN/trj/DCA_imputation.csv",header = FALSE)
data<-as.matrix.data.frame(data)
colnames(data)<-colnames(cell_metadata)

rownames(data) <- rownames(gene_annotation)
# 将数据转换为稀疏矩阵
library(Matrix)
data_sparse <- Matrix(as.matrix(data), sparse=TRUE)

# 将稀疏矩阵转换为压缩稀疏列（compressed sparse column）格式的矩阵
data_dgCMatrix <- as(data_sparse, "dgCMatrix")
colnames(data_dgCMatrix)<-colnames(expression_matrix)
rownames(data_dgCMatrix) <- rownames(expression_matrix)
cds <- new_cell_data_set(data_dgCMatrix,
                         cell_metadata = cell_metadata,
                         gene_metadata = gene_annotation)
cds <- preprocess_cds(cds, num_dim = 50)
cds <- align_cds(cds, alignment_group = "batch", residual_model_formula_str = "~ bg.300.loading + bg.400.loading + bg.500.1.loading + bg.500.2.loading + bg.r17.loading + bg.b01.loading + bg.b02.loading")
cds <- reduce_dimension(cds)
plot_cells(cds, label_groups_by_cluster=FALSE,  color_cells_by = "cell.type")
ciliated_genes <- c("che-1",
                    "hlh-17",
                    "nhr-6",
                    "dmd-6",
                    "ceh-36",
                    "ham-1")
pdf("./DCA_single_genes.pdf",width = 8.27,height = 8.27)
plot_cells(cds,
           genes=ciliated_genes,
           label_cell_groups=FALSE,
           show_trajectory_graph=FALSE)
dev.off()
cds <- cluster_cells(cds)
#plot_cells(cds, color_cells_by = "partition")


cds <- learn_graph(cds)
pdf("./DCA_celltype.pdf",width = 8.27,height = 8.27)
plot_cells(cds,
           color_cells_by = "cell.type",
           label_groups_by_cluster=FALSE,
           label_leaves=FALSE,
           label_branch_points=FALSE,
           graph_label_size=3)
dev.off()
pdf("./DCA_embryo.time.bin.pdf",width = 8.27,height = 8.27)
plot_cells(cds,
           color_cells_by = "embryo.time.bin",
           label_cell_groups=FALSE,
           label_leaves=TRUE,
           label_branch_points=TRUE,
           graph_label_size=3)
dev.off()
