for dataset in 200 300 400 500 600 700
do
python cell2location_pipeline.py Z:/jxliu/project/ID-GAT/ID/simulation/seqFISH-plus/seqFISH-plus_reference_data.h5ad Z:/jxliu/project/ID-GAT/ID/simulation/seqFISH-plus/seqFISH-plus_major_combined_simulation_$dataset.h5ad major_cell_type Z:/jxliu/project/ID-GAT/ID/results/simulation_seqFISH seqFISH-plus_major_combined_simulation_$dataset.h5ad major
done

for dataset in 200 300 400 500 600 700
do
python cell2location_pipeline.py Z:/jxliu/project/ID-GAT/ID/simulation/seqFISH-plus/seqFISH-plus_reference_data.h5ad Z:/jxliu/project/ID-GAT/ID/simulation/seqFISH-plus/seqFISH-plus_minor_combined_simulation_$dataset.h5ad minor_cell_type Z:/jxliu/project/ID-GAT/ID/results/simulation_seqFISH seqFISH-plus_minor_combined_simulation_$dataset.h5ad minor
done
