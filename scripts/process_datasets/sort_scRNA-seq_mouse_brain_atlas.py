from pathlib import Path

import pandas as pd

data_dir = Path("Z:/jxliu/datasets/scRNA-seq/mouse_brain_atlas/raw_data")
exp_path = data_dir / "exp" / "GSE116470_F_GRCm38.81.P60Cerebellum_ALT.raw.dge.txt.gz"
tmp = pd.read_csv(exp_path, compression="gzip", index_col=0, sep='\t')
