sc_list=(CK159 CK161 CK160 CK164 CK165 CK163 CK162 CK158) 
sp_list=(10X0027 10X0017 10X0026 10X0020 10X0018 10X0025 10X009 10X001)
for i in {0..7}
do
python DestVI_pipeline.py /data/bqliu_data/jxliu_data/projects/staid/data/human_myocardial_infraction/processed/scRNA-seq/${sc_list[$i]}.h5ad /data/bqliu_data/jxliu_data/projects/staid/data/human_myocardial_infraction/processed/SRT/10X_Visium_${sp_list[$i]}.h5ad cell_type /data/bqliu_data/jxliu_data/projects/staid/results/myocardial_infaration/others myocardial-infarction
done


sc_list=(CK374 CK364 CK365 CK366 CK367 CK368)
sp_list=(ACH002 ACH0024  ACH0010 ACH005 ACH006 ACH008)
for i in {0..5}
do
python DestVI_pipeline.py /data/bqliu_data/jxliu_data/projects/staid/data/human_myocardial_infraction/processed/scRNA-seq/${sc_list[$i]}.h5ad /data/bqliu_data/jxliu_data/projects/staid/data/human_myocardial_infraction/processed/SRT/10X_Visium_${sp_list[$i]}.h5ad cell_type /data/bqliu_data/jxliu_data/projects/staid/results/myocardial_infaration/others myocardial-infarction
done

sc_list=(CK369 CK370 CK356 CK371 CK372 CK356 CK373)
sp_list=(ACH0016 ACH007 ACH0022 ACH0011 ACH0013 ACH0021 ACH0015)
for i in {0..6}
do
python DestVI_pipeline.py /data/bqliu_data/jxliu_data/projects/staid/data/human_myocardial_infraction/processed/scRNA-seq/${sc_list[$i]}.h5ad /data/bqliu_data/jxliu_data/projects/staid/data/human_myocardial_infraction/processed/SRT/10X_Visium_${sp_list[$i]}.h5ad cell_type /data/bqliu_data/jxliu_data/projects/staid/results/myocardial_infaration/others myocardial-infarction
done

sc_list=(CK357 CK358 CK359 CK360 CK361 CK362 CK363)
sp_list=(ACH003 ACH004 ACH0028 ACH0023 ACH0014 ACH0019 ACH0012)
for i in {0..6}
do
python DestVI_pipeline.py /data/bqliu_data/jxliu_data/projects/staid/data/human_myocardial_infraction/processed/scRNA-seq/${sc_list[$i]}.h5ad /data/bqliu_data/jxliu_data/projects/staid/data/human_myocardial_infraction/processed/SRT/10X_Visium_${sp_list[$i]}.h5ad cell_type /data/bqliu_data/jxliu_data/projects/staid/results/myocardial_infaration/others myocardial-infarction
done
