# SlideLab: A Modular Framework for Standardized Whole Slide Image Preprocessing and Zero-Disk Streaming

**Lorenzo Olmo Marchal**¹\*  

¹ Department of Bioengineering, University of California, San Diego & ColabBio Project (`https://colabbio.org`)  
\* Correspondence: `lolmomarchal@colabbio.org` / `aolmomarchal@ucsd.edu`  

*Based on Aim 1 of the M.S. Thesis Defense in Bioengineering at UC San Diego (September 22, 2025; Committee: Prof. Ludmil B. Alexandrov [Chair], Prof. Benjamin Smarr, Prof. Vineet Bafna).*

**bioRxiv Category:** Bioinformatics / Computational Pathology  

---

## Abstract

**Motivation:** Computational pathology relies on gigapixel Whole Slide Images (WSIs) exceeding $100{,}000 \times 100{,}000$ pixels. However, reproducibility and scalability are limited by custom, unshared preprocessing scripts, rigid end-to-end toolkits that fail at scale or lack comprehensive quality control (QC), and heavy disk I/O caused by materializing tens of thousands of intermediate tile images per slide.

**Results:** We present **SlideLab**, an open-source, modular WSI preprocessing pipeline engineered for both traditional disk-based HPC workflows and zero-disk on-the-fly streaming. On a standardized benchmark of **1,000 TCGA-COAD WSIs** (20×, 256 px tiles) executed prior to the in-memory streaming integration, SlideLab achieved the highest tissue segmentation Dice score (**0.741 mean / 0.815 median**, Wilcoxon $p = 1.39 \times 10^{-6}$ vs. STAMP; $p < 10^{-5}$ vs. all others) and **88.25% artifact-free tissue tiles**, while running **7.7× faster in CPU time** (**6.1× user time**) than CLAM and **2.2× faster CPU time** (**1.65× user time**) than SlideFlow in full QC + Macenko stain normalization. Holding a downstream Attention-Based Multiple Instance Learning (ABMIL) classifier constant for Microsatellite Instability (MSI vs. MSS) prediction, SlideLab preprocessing yielded top performance on both **FFPE** ($\text{AUC} = 0.761 \pm 0.108$, $\text{F1} = 0.497 \pm 0.136$) and **Flash-Frozen** cohorts ($\text{AUC} = 0.710 \pm 0.008$, $\text{F1} = 0.460 \pm 0.030$). Furthermore, integrating SlideLab (`v2.1.0`) with **MedVFS** (`cb-vfs`) and **Nextflow** (`cb-slidelab-abmil`) introduces zero-disk in-memory tile streaming and database-first OMERO write-back, eliminating 5,000–20,000 intermediate tile files per WSI (**99.6% disk-write reduction**) and delivering a **3×–4× end-to-end pipeline speedup**.

**Availability:** Source code is available under the Apache 2.0 License at `https://github.com/colabbio/SlideLab` and `https://github.com/colabbio/cb-slidelab-abmil`.

---

## 1. Introduction

Digitized hematoxylin and eosin (H&E) Whole Slide Images (WSIs) enable quantitative biomarker discovery and deep learning–based clinical prediction. Yet most computational pathology studies rely on bespoke preprocessing scripts, hindering cross-cohort reproducibility. Existing frameworks such as CLAM, STAMP, SlideFlow, TiaToolBox, and PathML tightly couple tiling, quality control, stain normalization, and feature encoding to fixed workflows, and frequently exhaust memory or storage beyond 100–300 WSIs.

**SlideLab** addresses this gap by decoupling WSI preprocessing into a modular, resumable producer–consumer pipeline with comprehensive artifact filtering, per-slide summary/error reporting, and native support for both standalone HPC execution and zero-disk cloud/Kubernetes streaming.

---

## 2. Methods

### 2.1 Modular Preprocessing Pipeline

SlideLab (`SlideLab/SlidePreprocessing.py`) implements composable stages:

1. **Slide-Level Tissue Segmentation (`TissueMask`):** Adapts the H&E Otsu thresholding method of Schreiber et al. (2023) to isolate foreground tissue from background glass prior to patching.
2. **Producer–Consumer Tile Processing (`SlidePreprocessing`):** Separates coordinate indexing (`Index Queue`) from parallel tile extraction, QC, normalization, and saving (`Save Queue`) across configurable CPU and GPU workers.
3. **Quality Control & Artifact Filtering:** Rejects blurry tiles via Laplacian variance (`--blur_threshold`), filters red and blue pen markings via RGB channel thresholding (`--red_pen_check`, `--blue_pen_check`), and enforces minimum tissue coverage (`--tissue_threshold`).
4. **Stain Normalization & Feature Encoding:** Applies Macenko stain normalization (`torchstain`) and modular feature extraction (e.g., ResNet50, UNI) to HDF5 (`features.h5`).

### 2.2 Zero-Disk In-Memory Streaming (`v2.1.0`) and OMERO Write-Back

To remove the I/O bottleneck of writing thousands of PNG/JPEG tiles per slide, SlideLab `v2.1.0` adds `StreamingTileDataset` (`SlideLab/tiling/no_saving/cpu.py`) and `omero_writer.py`:

- **MedVFS POSIX Streaming:** Reads WSI byte ranges lazily over a **MedVFS** (`cb-vfs`) FUSE mount (4 MB LRU block cache + TIFF IFD prefetch) directly into RAM/VRAM, writing only compact `features.h5` embeddings (~10–20 MB/slide) to disk.
- **Database-First OMERO Write-Back:** Persists slide-level metrics, classification probabilities, and **Polarized Attentional Certainty (PAC)** scores back to OMERO via persistent image IDs as `MapAnnotation` key-value pairs, `ROI` overlays, and heatmap attachments, with passwordless ephemeral `OMERO_SESSION` support.

---

## 3. Results

### 3.1 Computational Performance (Disk-Based Benchmark, 1,000 TCGA-COAD WSIs)

> *Note: The 1,000-WSI multi-tool benchmark below was conducted using SlideLab's disk-based pipeline prior to the `v2.1.0` in-memory streaming integration.*

All tools were benchmarked under identical parameters (20× magnification, 256 px tiles) up to 1,000 WSIs from TCGA-COAD. **TiaToolBox** and **PathML** failed beyond 100–300 WSIs due to memory and runtime constraints.

- **Baseline Tiling Only (1,000 WSIs):** SlideLab completed in **177.54 min user time** (1.37× faster user time than STAMP, which extracted coordinates only). CLAM was **40.6× slower** in CPU time (**17.5× slower** in user time), and SlideFlow was **11.6× slower** in CPU time (**4.7× slower** in user time).
- **Full Pipeline (Tiling + QC + Macenko Stain Normalization, 1,000 WSIs):**
  - **7.7× faster** in CPU time and **6.1× faster** in user time than **CLAM**.
  - **2.2× faster** in CPU time and **1.65× faster** in user time than **SlideFlow**.
  - Comparable user time to **STAMP** (1.1×), while executing full QC and stain normalization rather than coordinate extraction alone.

### 3.2 Quality Control & Tissue Segmentation

Evaluated against GrandQC ($\text{Dice} = 0.938$ tissue segmentation reference):
- **Slide-Level Tissue Dice:** SlideLab achieved the highest **mean Dice (0.741)** and **median Dice (0.815)**, significantly outperforming STAMP (Wilcoxon signed-rank $p = 1.39 \times 10^{-6}$) and all other tools ($p < 10^{-5}$).
- **Tile-Level Cleanliness:** SlideLab retained the highest percentage of **artifact-free tissue tiles (88.25%)** and lowest contamination across artifact categories (except air bubbles).

### 3.3 Downstream ABMIL Classification (MSI vs. MSS)

Holding the ABMIL architecture constant across all preprocessing pipelines (10-fold cross-validation internally, evaluated on an unseen TCGA-COAD test cohort):

| Cohort | Preprocessing Pipeline | AUC | F1 Score | Notes |
|---|---|---|---|---|
| **FFPE** | **SlideLab (Full: QC + Stain Norm)** | **$0.761 \pm 0.108$** | **$0.497 \pm 0.136$** | Highest recall ($p < 0.01$ vs. all others) |
| **FFPE** | **SlideLab (QC Only)** | $0.733 \pm 0.023$ | $0.482 \pm 0.054$ | Lowest variance; F1 $>$ SlideFlow ($p=0.002$), STAMP ($p=0.0039$) |
| **FFPE** | SlideFlow | $0.721$ | $0.386$ | Lower F1 vs. SlideLab Full ($p=0.0273$) |
| **FFPE** | STAMP | $0.714$ | $0.327$ | Lower AUC ($p=0.0469$) & F1 ($p=0.0039$) vs. SlideLab QC |
| **FFPE** | CLAM | $0.644$ | $0.294$ | Lowest across all metrics |
| **Flash-Frozen** | **SlideLab (Full: QC + Stain Norm)** | **$0.710 \pm 0.008$** | **$0.460 \pm 0.030$** | Statistically superior to CLAM, SlideFlow, STAMP ($p < 0.05$) |
| **Flash-Frozen** | **SlideLab (QC Only)** | $0.688 \pm 0.016$ | $0.281 \pm 0.073$ | Significant AUC vs. all other tools ($p < 0.006$) |
| **Flash-Frozen** | SlideFlow | $0.660$ | $0.336$ | Lower AUC vs. SlideLab QC ($p=0.0059$) |

### 3.4 Impact of Zero-Disk In-Memory Streaming (`v2.1.0` + MedVFS)

Following the 1,000-WSI disk-based evaluation, SlideLab `v2.1.0` was integrated with **MedVFS** (`cb-vfs`) and **Nextflow 25.04.6 DSL2** (`cb-slidelab-abmil`) to stream WSI tiles directly in memory:

| Pipeline Metric | Disk-Based SlideLab (Pre-Streaming) | Zero-Disk SlideLab `v2.1.0` + MedVFS | Observed Improvement |
|---|---|---|---|
| **Intermediate tile files / WSI** | 5,000–20,000 PNG/JPEG files | **0 files** (in-memory tensors) | **100% eliminated** |
| **Disk volume written / WSI** | ~2.5–5.0 GB | **~10–20 MB** (`features.h5`) | **99.6% reduction** |
| **End-to-end I/O & throughput** | Write-then-read PVC bottleneck | Single-pass memory streaming | **3×–4× speedup** |
| **Kubernetes Pod cold-start** | Monolithic GPU image (~18 GB) | Lightweight `slidelab:cpu` (~1.2 GB) | **< 3 seconds** |

---

## 4. Conclusion

Clean, standardized preprocessing has a greater impact on downstream MIL accuracy than raw tile count or model complexity. **SlideLab** combines best-in-class tissue QC and Macenko normalization with zero-disk MedVFS streaming and OMERO write-back, providing an Apache 2.0 foundation for reproducible computational pathology. For the interpretability methodology developed alongside SlideLab, see [**Polarized Attentional Certainty (PAC)**](pac_article.md).

---

## References

1. Schreiber BA, et al. Bang and the artefacts are gone! Rapid artefact removal and tissue segmentation in haematoxylin and eosin stained biopsies. *arXiv:2308.13304*. 2023.
2. Macenko M, et al. A method for normalizing histology slides for quantitative analysis. *Proc IEEE ISBI*. 2009:1107–1110.
3. Ilse M, Tomczak J, Welling M. Attention-based Deep Multiple Instance Learning. *Proc ICML*. 2018;80:2127–2136.
4. Chen RJ, et al. Towards a general-purpose foundation model for computational pathology. *Nat Med*. 2024.
