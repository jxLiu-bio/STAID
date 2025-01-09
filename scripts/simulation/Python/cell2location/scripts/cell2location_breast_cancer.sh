for dataset in CID4290 CID4465 CID4535 CID44971
do
python cell2location_pipeline.py /home/frank/Documents/GAT-ID/simulation/breast_cancer_sc/${dataset}_scRNA_seq_with_annotations.h5ad /home/frank/Documents/GAT-ID/simulation/breast_cancer_ST/${dataset}_visium_breast_cancer.h5ad celltype_major /home/frank/Documents/GAT-ID/simulation/results_breast_cancer
done
