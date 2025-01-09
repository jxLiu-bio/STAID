for dataset in Puck-191204-01_55_simulated.h5ad
do 
python cell2location_pipeline.py /home/jxliu/Desktop/projects/staid/data/simulation/$dataset /home/jxliu/Desktop/projects/staid/data/simulation/scRNA_Hippocampus_191204-01_s3.h5ad cell_type /home/jxliu/Desktop/projects/staid/results/simulation/s3 $dataset s3
done

