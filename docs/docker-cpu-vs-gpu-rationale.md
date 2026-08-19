# Architectural Rationale: Docker Container Strategy (CPU vs GPU) in SlideLab

**Author:** ColabBio Architecture & Engineering  
**Date:** August 19, 2026  
**Branch:** `feat/on-the-fly-omero-streaming`  
**Status:** Approved / Technical Reference  

---

## 1. Design Question

> **Why build separate Docker images (`slidelab:cpu` and `slidelab:gpu`) instead of a single universal image where CPU or GPU execution is simply toggled via runtime arguments (e.g. `--device cpu|cuda`)?**

---

## 2. Technical Rationale & Comparison

Technically, **a GPU PyTorch image (`slidelab:gpu`) can run seamlessly on CPU-only nodes** by falling back to host CPU instructions when no NVIDIA GPU is present or when `--device cpu` is passed. However, in production cloud infrastructure, Kubernetes clusters, and large-scale MLOps, there are critical performance, cost, and latency reasons to maintain a multi-stage build separation:

| Dimension / Metric | `slidelab:cpu` (CPU Target) | `slidelab:gpu` (GPU Target) |
| :--- | :--- | :--- |
| **Image Size** | **~1.2 – 1.5 GB** | **~8 – 18 GB** |
| **Included Libraries** | Python, OpenSlide, AVX2/AVX-512, Intel MKL / OpenBLAS | Python, OpenSlide, CUDA Runtime (`libcudart`), cuDNN, cuBLAS, NCCL, TensorRT |
| **Download / Pull Time** | **~3 – 8 seconds** | **~2 – 5 minutes** (per Pod/Node) |
| **Autoscaling Latency (*Cold Start*)** | Near-instantaneous | High; saturates cluster network bandwidth |
| **Registry & Network Bandwidth** | Minimal footprint | Heavy network I/O on Artifact Registry / Harbor |
| **Target Use Cases** | CI/CD pipelines, local dev on laptops, lightweight QC, distributed tiling | Heavy MIL inference, high-dimensional feature encoding (UNI, ResNet-50) |

---

## 3. Critical Factors in Kubernetes & Digital Pathology Pipelines

### 3.1. Mitigating Cold-Start Latency in Elastic Auto-scaling
When Nextflow or Kubernetes orchestrates massive WSI batch processing (e.g., 50–100 parallel Pods spawned on standard compute nodes for initial tissue segmentation or quality control):
* If every Pod pulls an **18 GB** image concurrently, network bandwidth on the node pool and container registry becomes a major bottleneck, driving startup latency from seconds to several minutes.
* With the **1.5 GB** CPU image, Kubernetes scheduling and execution start within seconds.

### 3.2. Local Development & Lightweight CI/CD
* In continuous integration environments (GitHub Actions, GitLab CI) and local laptops without dedicated NVIDIA hardware, pulling or building 18 GB per commit consumes disk quotas and slows down development velocity.
* `slidelab:cpu` allows developers to run unit tests, linting, and dataset streaming logic with zero overhead.

---

## 4. Pipeline Integration in Nextflow DSL2

Using Nextflow DSL2 conditional container directives (as implemented in [`nextflow/main.nf`](../nextflow/main.nf)), the optimal container image is resolved dynamically based on requested compute hardware:

```groovy
process PROCESS_AND_ENCODE_WSI {
    tag "${patient_id}"
    
    // Dynamic container resolution based on target device
    container params.device == 'cuda' ? 'slidelab:gpu' : 'slidelab:cpu'
    accelerator params.device == 'cuda' ? 1 : 0, type: 'nvidia.com/gpu'
    
    cpus params.device == 'cpu' ? 8 : 4
    memory params.device == 'cpu' ? '16 GB' : '8 GB'
    
    script:
    """
    python /app/SlideLab/SlidePreprocessing.py \
        -i ${slide_file} \
        -o . \
        --stream_features \
        --device ${params.device} \
        --batch_size ${params.batch_size}
    """
}
```

---

## 5. Deployment Recommendations

1. **Multi-Stage Strategy (Recommended & Current):**
   * Maintain the [`Dockerfile`](../Dockerfile) with multi-stage targets (`--target cpu` and `--target gpu`).
   * Deploy `slidelab:cpu` to CPU worker pools, CI/CD runners, and local workstations.
   * Deploy `slidelab:gpu` exclusively to GPU-accelerated node pools (e.g. NVIDIA T4, A100, L4).
2. **Unified Image Strategy (Optional for specific cluster requirements):**
   * If a target deployment requires a single canonical OCI image in the catalog, publish `slidelab:gpu` as `latest` and specify `--device cpu` when running on non-accelerated nodes.
