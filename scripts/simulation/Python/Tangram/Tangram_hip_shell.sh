# s1
for dataset in Puck-191204-01_55_simulated.h5ad Puck-191204-01_100_simulated.h5ad
do
python Tangram_pipeline.py /home/jxliu/Desktop/projects/staid/data/simulation/scRNA_Hippocampus_191204-01_s1.h5ad /home/jxliu/Desktop/projects/staid/data/simulation/$dataset cell_type /home/jxliu/Desktop/projects/staid/results/simulation/s1 s1
done

for dataset in Puck-200115-08_55_simulated.h5ad Puck-200115-08_100_simulated.h5ad
do 
python Tangram_pipeline.py /home/jxliu/Desktop/projects/staid/data/simulation/scRNA_Hippocampus_200115-08_s1.h5ad /home/jxliu/Desktop/projects/staid/data/simulation/$dataset cell_type /home/jxliu/Desktop/projects/staid/results/simulation/s1 s1
done

# s2
for dataset in Puck-191204-01_55_simulated.h5ad
do
python Tangram_pipeline.py /home/jxliu/Desktop/projects/staid/data/simulation/scRNA_Hippocampus_191204-01_s2_Endothelial-Fit1.h5ad /home/jxliu/Desktop/projects/staid/data/simulation/$dataset cell_type /home/jxliu/Desktop/projects/staid/results/simulation/s2 s2_End
done

for dataset in Puck-191204-01_55_simulated.h5ad
do
python Tangram_pipeline.py /home/jxliu/Desktop/projects/staid/data/simulation/scRNA_Hippocampus_191204-01_s2_Polydendrocyte-Tnr.h5ad /home/jxliu/Desktop/projects/staid/data/simulation/$dataset cell_type /home/jxliu/Desktop/projects/staid/results/simulation/s2 s2_Pol
done

#s3
for dataset in Puck-191204-01_55_simulated.h5ad
do
python Tangram_pipeline.py /home/jxliu/Desktop/projects/staid/data/simulation/scRNA_Hippocampus_191204-01_s3.h5ad /home/jxliu/Desktop/projects/staid/data/simulation/$dataset cell_type /home/jxliu/Desktop/projects/staid/results/simulation/s3 s3
done
