for dataset in Puck-200115-08_55_simulated.h5ad Puck-200115-08_100_simulated.h5ad
do 
python cell2location_pipeline.py /home/jxliu/Desktop/projects/staid/data/simulation/$dataset /home/jxliu/Desktop/projects/staid/data/simulation/scRNA_Hippocampus_200115-08_s1.h5ad cell_type /home/jxliu/Desktop/projects/staid/results/simulation/s1 $dataset s1
done
