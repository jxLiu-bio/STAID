## Data Preparation

###  Transcriptomics data
STAID requires the following input data:

- **scRNA-seq data (raw counts)**
  - Must be provided as an `AnnData` object before deconvolution.
  - The data should contain **raw count values** (not normalized or log-transformed).
  - Cell type annotations must be provided in `adata.obs.keys()`, e.g.:
    ```python
    sc_adata.obs['celltype']
    ```

- **Spatial transcriptomics data (raw counts)**
  - Must be provided as an `AnnData` object before deconvolution.
  - The expression matrix should contain **raw count values**.
  - Spatial coordinates (e.g., `spatial`) should be included in `adata.obsm`.

Both datasets should share a common set of genes (overlapping gene symbols), which STAID uses to perform deconvolution.


---

### Example Datasets
The demo spatial transcriptomics data (human breast cancer Visium) are available at https://doi.org/10.5281/zenodo.4739739 
and match human breast cancer scRNA-seq reference datasets are available through the Gene Expression Omnibus under accession number GSE176078. 

For convenience, we also provide a sorted version on Google Drive: [Download from Google Drive](https://drive.google.com/drive/folders/1-GhHslCBIYvNFb1Zs3DmLKVg9JZx1QSP?usp=sharing).



