for dataset in seqFISH+_SScortex_cellType_50.h5seurat
do
Rscript SPOTlight_pipeline.R Z:/jxliu/project/ID-GAT/ID/simulation/scRNA-seq-R/seqFISH+_SScortex_single.h5Seurat Z:/jxliu/project/ID-GAT/ID/simulation/ST-R/$dataset cell_type Z:/jxliu/project/ID-GAT/ID/simulation/results
done