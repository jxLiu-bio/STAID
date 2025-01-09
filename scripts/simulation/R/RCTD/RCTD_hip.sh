for dataset in Puck-200115-08_55_simulated.h5seurat Puck-200115-08_100_simulated.h5seurat
do
Rscript RCTD_pipeline.R /home/jxliu/Desktop/projects/staid/data/simulation/scRNA_Hippocampus_200115-08_s1.h5seurat /home/jxliu/Desktop/projects/staid/data/simulation/$dataset cell_type /home/jxliu/Desktop/projects/staid/results/simulation/s1 s1
done

for dataset in Puck-191204-01_55_simulated.h5seurat Puck-191204-01_100_simulated.h5seurat
do
Rscript RCTD_pipeline.R /home/jxliu/Desktop/projects/staid/data/simulation/scRNA_Hippocampus_191204-01_s1.h5seurat /home/jxliu/Desktop/projects/staid/data/simulation/$dataset cell_type /home/jxliu/Desktop/projects/staid/results/simulation/s1 s1
done

for dataset  in Puck-191204-01_55_simulated.h5seurat
do
Rscript RCTD_pipeline.R /home/jxliu/Desktop/projects/staid/data/simulation/scRNA_Hippocampus_191204-01_s2_Endothelial-Fit1.h5seurat /home/jxliu/Desktop/projects/staid/data/simulation/$dataset cell_type /home/jxliu/Desktop/projects/staid/results/simulation/s2 s2_End
done

for dataset in Puck-191204-01_55_simulated.h5seurat
do
Rscript RCTD_pipeline.R /home/jxliu/Desktop/projects/staid/data/simulation/scRNA_Hippocampus_191204-01_s2_Polydendrocyte-Tnr.h5seurat /home/jxliu/Desktop/projects/staid/data/simulation/$dataset cell_type /home/jxliu/Desktop/projects/staid/results/simulation/s2 s2_Pol
done

for dataset in Puck-191204-01_55_simulated.h5seurat
do
Rscript RCTD_pipeline.R /home/jxliu/Desktop/projects/staid/data/simulation/scRNA_Hippocampus_191204-01_s3.h5seurat /home/jxliu/Desktop/projects/staid/data/simulation/$dataset cell_type /home/jxliu/Desktop/projects/staid/results/simulation/s3 s3
done
