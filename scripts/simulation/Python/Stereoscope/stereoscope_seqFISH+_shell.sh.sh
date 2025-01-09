for dataset in 200 300 400 500 600 700
do
python stereoscope_pipeline.py Z:/jxliu/project/ID-GAT/ID/simulation/seqFISH-plus/seqFISH-plus_reference_data.h5ad Z:/jxliu/project/ID-GAT/ID/simulation/seqFISH-plus/seqFISH-plus_major_combined_simulation_$dataset.h5ad major_cell_type Z:/jxliu/project/ID-GAT/ID/results/simulation_seqFISH major
done

for dataset in 200 300 400 500 600 700
do
python stereoscope_pipeline.py Z:/jxliu/project/ID-GAT/ID/simulation/seqFISH-plus/seqFISH-plus_reference_data.h5ad Z:/jxliu/project/ID-GAT/ID/simulation/seqFISH-plus/seqFISH-plus_minor_combined_simulation_$dataset.h5ad minor_cell_type Z:/jxliu/project/ID-GAT/ID/results/simulation_seqFISH minor
done
