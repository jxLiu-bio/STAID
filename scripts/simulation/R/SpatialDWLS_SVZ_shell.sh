for dataset in seqFISH+_SScortex_cellType_200.h5seurat
do
Rscript SpatialDWLS_pipeline.R Z:/jxliu/project/ID-GAT/ID/simulation/scRNA-seq-R/seqFISH+_SScortex_single.h5Seurat Z:/jxliu/project/ID-GAT/ID/simulation/ST-R/$dataset cell_type Z:/jxliu/project/ID-GAT/ID/simulation/results
done