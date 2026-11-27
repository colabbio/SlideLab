# Polarized Attentional Certainty (PAC): Resolving Directionality and Confidence in Weakly-Supervised Multiple Instance Learning

**Lorenzo Olmo Marchal**¹\*  

¹ Department of Bioengineering, University of California, San Diego & ColabBio Project (`https://colabbio.org`)  
\* Correspondence: `lolmomarchal@colabbio.org` / `aolmomarchal@ucsd.edu`  

*Based on Aim 2 of the M.S. Thesis Defense in Bioengineering at UC San Diego (September 22, 2025; Committee: Prof. Ludmil B. Alexandrov [Chair], Prof. Benjamin Smarr, Prof. Vineet Bafna).*

**bioRxiv Category:** Bioinformatics / Machine Learning Interpretability  

---

## Abstract

**Motivation:** Weakly-supervised Attention-Based Multiple Instance Learning (ABMIL) is widely used in computational pathology and single-cell diagnostics, where a single bag-level label supervises thousands of image patches or cells. Standard interpretability relies on scalar attention weights ($\alpha_i$) or instance-level probabilities, yet neither suffices: attention quantifies *importance* without indicating whether an instance supports the positive or negative class, whereas probability ignores how strongly an instance influences the bag-level decision.

**Results:** We introduce **Polarized Attentional Certainty (PAC)**, a bounded $[-1, 1]$ per-instance interpretability score that combines attention magnitude with class-logit certainty (the difference in class logits). Across two distinct biomedical domains, PAC recovers biologically coherent phenotypes that attention or probability alone fail to separate cleanly: (1) in **TCGA-COAD colorectal cancer Whole Slide Images** (Microsatellite Instability, MSI-H vs. MSS), 99th-percentile PAC patches capture immune-hot regions enriched in inflammatory tissue (**3.09% vs. 0.79%**) and lymphocytes (**41.66% vs. 23.61%**, PanNuke HoVer-Net) with smaller, rounder nuclei, whereas 1st-percentile PAC patches capture stromal-dominant regions (**60.71% vs. 18.27%**) with elongated nuclei; (2) in **single-cell B-cell Acute Lymphoblastic Leukemia (B-ALL) flow cytometry** ($\text{AUC} = 1.0$), PAC cleanly isolates $\text{CD10}^+/\text{CD34}^+/\text{CD19}^+$ B-ALL blasts (99th percentile) from $\text{CD45}^+$ non-B-ALL T/NK lymphocytes (1st percentile), outperforming both probability- and attention-only ranking.

**Availability:** Implemented within the ColabBio SlideLab + ABMIL ecosystem under the Apache 2.0 License at `https://github.com/colabbio/SlideLab` and `https://github.com/colabbio/cb-slidelab-abmil`.

---

## 1. Introduction

Clinical adoption of artificial intelligence in pathology and hematology is constrained by the "black box" nature of deep neural networks. Clinicians require transparent explanations to trust model outputs and mitigate liability, regulatory agencies (including the FDA) increasingly emphasize explainability for medical devices, and researchers need interpretable attributions to discover genuine biological features rather than spurious dataset artifacts.

In Multiple Instance Learning (MIL), each sample (a Whole Slide Image or a flow cytometry tube) is represented as an unordered bag of $K$ instances (tiles or single cells) with a single bag-level label. Post-hoc attribution methods such as SHAP or LIME do not adapt naturally to variable-length bags of $10^3$–$10^5$ instances. While ABMIL models produce per-instance attention weights $\alpha_i$, two fundamental ambiguities remain:

1. **Attention lacks polarity:** A high attention weight $\alpha_i$ indicates that instance $i$ contributed to the aggregation, but does not reveal whether it drove the prediction toward the positive class ($+1$) or the negative class ($-1$).
2. **Probability lacks bag-level influence:** Per-instance class probabilities ignore whether the model actually weighted that instance heavily when forming the slide-level decision.

---

## 2. Polarized Attentional Certainty (PAC)

**Polarized Attentional Certainty (PAC)** resolves both ambiguities by combining the instance attention weight with the magnitude and sign of the model's certainty (the difference between class logits for each instance). The resulting score is bounded in $[-1, 1]$:

- **$\text{PAC}_i \to +1$:** High attention and high certainty toward the **positive class** (e.g., MSI-H tumor-immune microenvironment or B-ALL leukemic blasts).
- **$\text{PAC}_i \to -1$:** High attention and high certainty toward the **negative class** (e.g., MSS stroma or mature non-B-ALL lymphocytes), or high attention with low positive-class certainty.

Ranking instances by PAC enables direct comparison of the biological extremes (e.g., 99th percentile vs. 1st percentile PAC) using downstream segmentation, morphometry, or marker-expression profiling.

---

## 3. Results

### 3.1 Case Study 1: Colorectal Cancer MSI-H vs. MSS (Whole Slide Images)

We applied PAC to an ABMIL classifier trained on **SlideLab**-preprocessed TCGA-COAD WSIs to distinguish Microsatellite Instable (MSI-H, immune-active) from Microsatellite Stable (MSS) tumors. Top (99th percentile) and bottom (1st percentile) PAC patches were analyzed using U-Net tissue segmentation and HoVer-Net nuclear segmentation (pretrained on PanNuke and MoNuSAC).

#### Tissue Distribution (U-Net Segmentation)

| Tissue Class | 99th Percentile PAC (% / Area $\text{px}^2$) | 1st Percentile PAC (% / Area $\text{px}^2$) | Difference ($\Delta$ Area $\text{px}^2$) |
|---|---|---|---|
| **Tumor** | 20.79% (179,437) | 18.02% (120,568) | $+58{,}869$ |
| **Stroma** | 18.27% (133,339) | **60.71% (670,684)** | $-537{,}345$ |
| **Inflammatory** | **3.09% (9,868)** | 0.79% (237) | $+9{,}631$ |
| **Necrosis** | **24.56% (178,733)** | 4.46% (23,273) | $+155{,}460$ |
| **Others** | 33.28% | 16.01% | — |

#### Nuclei Classification & Morphology (HoVer-Net)

- **PanNuke-Pretrained HoVer-Net:** 99th percentile PAC patches were strongly enriched in **lymphocytes (41.66% vs. 23.61%)**, **macrophages (6.75% vs. 2.23%)**, and **neutrophils (4.03% vs. 1.44%)**, whereas 1st percentile PAC patches were dominated by **epithelial cells (64.93% vs. 37.95%)**.
- **MoNuSAC-Pretrained HoVer-Net:** 99th percentile PAC patches showed higher **inflammatory (17.18% vs. 6.01%)** and **dead (11.61% vs. 3.12%)** nuclei, whereas 1st percentile PAC patches showed higher **connective/stromal nuclei (48.99% vs. 27.18%)**.
- **Nuclear Morphometry:** 99th percentile PAC nuclei were **smaller and rounder** (mean area $336.89\text{ px}^2$, median $288.00\text{ px}^2$; median circularity $0.831$; median aspect ratio $1.555$), consistent with dense lymphocyte infiltration. In contrast, 1st percentile PAC nuclei were **larger and more elongated/asymmetric** (mean area $366.26\text{ px}^2$, median $319.00\text{ px}^2$; median circularity $0.772$; median aspect ratio $1.737$), consistent with stromal/fibroblast morphology and chromosomal instability.

### 3.2 Case Study 2: B-Cell Acute Lymphoblastic Leukemia (Single-Cell Flow Cytometry)

To evaluate PAC beyond histopathology on near-single-cell data, we trained an ABMIL model on **B-ALL flow cytometry** panels (evaluating fold #4, $\text{AUC} = 1.0$) measuring surface and internal markers (`FSC`, `SSC`, `CD66b`, `CD22`, `CD19`, `CD24`, `CD10`, `CD34`, `CD38`, `CD20`, `CD45`).

| Selection Method | 1st Percentile Population | 99th Percentile Population | Clinical Concordance |
|---|---|---|---|
| **Probability-Based** | High FSC/SSC & CD66b (granulocytes / non-B-ALL) | Closer to B-ALL phenotype, but excessively high FSC/SSC | Partial |
| **Attention-Based** | High FSC/SSC & CD66b (granulocytes / non-B-ALL) | Smaller cells, but ambiguous immunophenotype | Ambiguous |
| **PAC-Based (Ours)** | **Mature T/NK lymphocytes** ($\text{CD45}^+$, low CD19/CD10/CD34) | **Classical B-ALL blasts** (high $\text{CD10}^+/\text{CD34}^+/\text{CD19}^+/\text{CD24}^+/\text{CD22}^+$, low-moderate FSC/SSC) | **Clean biological separation** |

---

## 4. Discussion and Future Work

**Polarized Attentional Certainty (PAC)** bridges attention-based MIL and biological verification across both gigapixel histopathology and single-cell cytometry. By unifying attention weight and class-logit certainty into a single $[-1, 1]$ score, PAC surfaces clinically meaningful cellular phenotypes while exposing potential model biases (such as stromal or necrosis reliance). Future directions include multi-class PAC extensions, survival/prognostic modeling, and human-in-the-loop pathologist feedback workflows via OMERO ROI overlays.

---

## References

1. Ilse M, Tomczak J, Welling M. Attention-based Deep Multiple Instance Learning. *Proc ICML*. 2018;80:2127–2136.
2. Graham S, et al. Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images. *Med Image Anal*. 2019;58:101563.
3. Gamper J, et al. PanNuke: An open pan-cancer histology dataset for nuclei instance segmentation and classification. *ECDP*. 2019.
4. Verma R, et al. MoNuSAC2020: A multi-organ nuclei segmentation and classification challenge. *IEEE Trans Med Imaging*. 2021.
