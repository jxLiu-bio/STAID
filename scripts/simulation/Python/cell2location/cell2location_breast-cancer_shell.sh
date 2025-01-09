for dataset in CID4290 CID4465 CID4535 CID44971
do
python cell2location_pipeline.py /data/bqliu_data/jxliu_data/projects/staid/data/Breast_cancer/merged_datasets/scRNA-seq/${dataset}_scRNA_seq_with_annotations.h5ad /data/bqliu_data/jxliu_data/projects/staid/data/Breast_cancer/merged_datasets/spatial/${dataset}_visium_breast_cancer.h5ad celltype_major /data/bqliu_data/jxliu_data/projects/staid/results/breast_cancer/others breast_cancer
done
