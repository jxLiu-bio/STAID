##  About

**STAID** is a deep learning framework for **iterative deconvolution of spatial transcriptomics data**.  
It combines **graph signal processing** with **pseudo-spot refinement** to align single-cell references with spatial data, enabling accurate inference of cell-type compositions.  

---


### Key Features of STAID

- **Refined Pseudo-spot Generation**  
  Estimates likely cell types in each spot to create pseudo spots that better approximate real biological mixtures.  

- **Graph Signal Processing with Fourier Transform**  
  Encodes expression profiles as graph signals, applies graph Fourier transform, and uses low-pass filtering to denoise, extracting robust features for accurate inference.  

- **Iterative Pseudo-spot Refinement**  
  Generates new pseudo spots based on previous deconvolution results, progressively improving accuracy and robustness.  

- **Reliable Cell-type Composition Estimation**  
  Provides accurate and stable deconvolution across diverse spatial transcriptomics datasets, enabling downstream studies of tissue architecture and cellular heterogeneity.  

---

###  Applications

STAID provides a versatile framework for mapping cell-type distributions and gaining insights into tissue architecture and heterogeneity.

---

