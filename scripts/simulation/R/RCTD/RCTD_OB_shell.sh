for dataset in seqFISH+_OBcortex_cellType_50.h5seurat
do
Rscript RCTD_pipeline.R Z:/jxliu/project/ID-GAT/ID/simulation/scRNA-seq-R/seqFISH+_OBcortex_single.h5Seurat Z:/jxliu/project/ID-GAT/ID/simulation/ST-R/$dataset cell_type Z:/jxliu/project/ID-GAT/ID/simulation/results
done