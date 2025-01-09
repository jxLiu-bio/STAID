for dataset in seqFISH+_OBcortex_cellType_100.h5ad seqFISH+_OBcortex_cellType_150.h5ad seqFISH+_OBcortex_cellType_200.h5ad
do
python cell2location_pipeline.py /home/frank/Documents/GAT-ID/simulation/scRNA-seq/seqFISH+_OBcortex_single.h5ad /home/frank/Documents/GAT-ID/simulation/ST/$dataset cell_type /home/frank/Documents/GAT-ID/simulation/results
done
