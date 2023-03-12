library(SAVER)
data.path="/home/liuyan/code/singlecell/MUSIC-GCN/packer.csv"
cortex <- read.csv(data.path, header = TRUE, row.names = 1, 
                       check.names = FALSE)
cortex <- as.matrix(raw.data[, -1])

cellnames <- read.table(data.path, skip = 7, nrows = 1, row.names = 1, 
                        stringsAsFactors = FALSE)
colnames(cortex) <- cellnames[-1]

dim(cortex)
cortex.saver <- saver(cortex, ncores = 12)
write.csv(cortex.saver,"packer.csv")


packer_saver=as.data.frame(cortex.saver[1])

write.csv(packer_saver,"packer_saver.csv")