#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
 * ═══════════════════════════════════════════════════════════════════
 *  SlideLab → MedVFS  Nextflow Pipeline
 * ═══════════════════════════════════════════════════════════════════
 *
 *  Full WSI preprocessing pipeline:
 *    1. Discover slides (from local dir, GCS, S3, or DICOM)
 *    2. Preprocess each slide (tissue mask → tile → normalize → QC)
 *    3. Encode features (optional: ResNet50, UNI)
 *    4. Aggregate reports + merge with clinical data
 *
 *  Usage:
 *    nextflow run main.nf --input_dir /data/slides --output_dir /data/results
 *    nextflow run main.nf -profile k8s --input_dir gs://bucket/slides
 *    nextflow run main.nf -profile gpu --input_dir /data/slides
 *
 *  Author: ColabBio / Lorenzo Olmo Marchal
 * ═══════════════════════════════════════════════════════════════════
 */

// ─── Parameters ──────────────────────────────────────────────────
params.input_dir        = null           // Required: path to WSI slides
params.output_dir       = './results'    // Output directory
params.tile_size        = 256            // Tile size in pixels
params.magnification    = 20             // Target magnification
params.normalize        = 'macenko'      // Stain normalization: macenko, reinhard, stainnet, null
params.device           = 'cpu'          // cpu or cuda
params.batch_size       = 32             // Batch size for GPU processing
params.encode           = true           // Enable feature encoding
params.encoder          = 'resnet50'     // Feature extractor model: resnet50, mahmood-uni, resnet50-truncated
params.stream_features  = true           // Zero-disk on-the-fly streaming directly to features.h5
params.omero_write_back = false          // Write PAC and QC findings back to OMERO Server
params.clinical_csv     = null           // Optional: clinical data CSV for join
params.tissue_threshold = 0.8
params.blur_threshold   = 0.05
params.remove_blurry    = true
params.remove_folds     = true
params.mask_scale       = 64

// Supported WSI extensions
def wsi_extensions = ['svs', 'tif', 'tiff', 'dcm', 'ndpi', 'mrxs', 'scn', 'bif']

// ─── Validation ──────────────────────────────────────────────────
if (!params.input_dir) {
    error "ERROR: --input_dir is required. Provide a path to WSI slide(s)."
}

// ─── Processes ───────────────────────────────────────────────────

/*
 * STEP 1: Discover WSI slides from input directory
 */
process DISCOVER_SLIDES {
    tag "discovery"
    
    output:
    path "slide_manifest.csv", emit: manifest
    
    script:
    def exts = wsi_extensions.collect { "*.${it}" }.join(' ')
    """
    echo "slide_path,patient_id" > slide_manifest.csv
    for ext in ${exts}; do
        find ${params.input_dir} -name "\${ext}" -type f 2>/dev/null | while read f; do
            patient_id=\$(basename "\$f" | sed 's/\\.[^.]*\$//')
            echo "\$f,\$patient_id" >> slide_manifest.csv
        done
    done
    echo "[SlideLab] Found \$(tail -n +2 slide_manifest.csv | wc -l) slides"
    """
}

/*
 * STEP 2: Preprocess individual slide (tissue mask → tile → normalize)
 */
process PREPROCESS_SLIDE {
    tag "${patient_id}"
    
    container params.device == 'cuda' ? 'slidelab:gpu' : 'slidelab:cpu'
    
    // GPU support
    accelerator params.device == 'cuda' ? 1 : 0, type: 'nvidia.com/gpu'
    
    cpus params.device == 'cpu' ? 4 : 2
    memory params.device == 'cpu' ? '8 GB' : '4 GB'
    time '2h'
    
    // Retry on OOM
    errorStrategy { task.exitStatus in [137, 143] ? 'retry' : 'terminate' }
    maxRetries 2
    
    input:
    tuple val(patient_id), path(slide_file)
    
    output:
    tuple val(patient_id), path("${patient_id}/"), emit: tiles
    path "${patient_id}/${patient_id}.csv", emit: tile_csv
    path "${patient_id}/mask.png", emit: mask, optional: true
    
    script:
    def norm_flag = params.normalize ? "--normalize_staining ${params.normalize}" : ""
    def blur_flag = params.remove_blurry ? "--remove_blurry_tiles" : ""
    def folds_flag = params.remove_folds ? "--remove_folds" : ""
    """
    python /app/SlideLab/SlidePreprocessing.py \
        -i ${slide_file} \
        -o . \
        -s ${params.tile_size} \
        -m ${params.magnification} \
        ${norm_flag} \
        ${blur_flag} \
        ${folds_flag} \
        --tissue_threshold ${params.tissue_threshold} \
        --blur_threshold ${params.blur_threshold} \
        --mask_scale ${params.mask_scale} \
        --device ${params.device} \
        --batch_size ${params.batch_size} \
        --reconstruct_slide
    """
}

/*
 * STEP 3: Feature encoding (decoupled / legacy disk mode)
 */
process ENCODE_FEATURES {
    tag "${patient_id}"
    
    container params.device == 'cuda' ? 'slidelab:gpu' : 'slidelab:cpu'
    accelerator params.device == 'cuda' ? 1 : 0, type: 'nvidia.com/gpu'
    
    cpus 4
    memory '16 GB'
    time '4h'
    
    when:
    params.encode && !params.stream_features
    
    input:
    tuple val(patient_id), path(tile_dir)
    
    output:
    tuple val(patient_id), path("${patient_id}_features.h5"), emit: features
    
    script:
    """
    python /app/SlideLab/SlidePreprocessing.py \
        -i ${tile_dir} \
        -o . \
        --output_format null \
        --encode \
        --feature_extractor ${params.encoder} \
        --device ${params.device} \
        --batch_size ${params.batch_size}
    
    # Rename output for clarity
    mv ${params.encoder}_features/${patient_id}.h5 ${patient_id}_features.h5 || \
    mv ${params.encoder}/${patient_id}.h5 ${patient_id}_features.h5 || true
    """
}

/*
 * UNIFIED STEP: Zero-Disk On-The-Fly Preprocessing & Encoding (CLAM Pattern + SlideLab QC)
 */
process PROCESS_AND_ENCODE_WSI {
    tag "${patient_id}"
    publishDir "${params.output_dir}/features", mode: 'copy', pattern: '*.h5'
    publishDir "${params.output_dir}/metrics", mode: 'copy', pattern: '*_metrics.json'
    
    container params.device == 'cuda' ? 'slidelab:gpu' : 'slidelab:cpu'
    accelerator params.device == 'cuda' ? 1 : 0, type: 'nvidia.com/gpu'
    
    cpus params.device == 'cpu' ? 8 : 4
    memory params.device == 'cpu' ? '16 GB' : '8 GB'
    time '2h'
    
    errorStrategy { task.exitStatus in [137, 143] ? 'retry' : 'terminate' }
    maxRetries 2
    
    input:
    tuple val(patient_id), path(slide_file)
    
    output:
    tuple val(patient_id), path("${patient_id}_features.h5"), emit: features
    tuple val(patient_id), path("${patient_id}_metrics.json"), emit: metrics
    
    script:
    def norm_flag = params.normalize ? "--normalize_staining ${params.normalize}" : ""
    def blur_flag = params.remove_blurry ? "--remove_blurry_tiles" : ""
    def folds_flag = params.remove_folds ? "--remove_folds" : ""
    """
    python /app/SlideLab/SlidePreprocessing.py \
        -i ${slide_file} \
        -o . \
        -s ${params.tile_size} \
        -m ${params.magnification} \
        ${norm_flag} \
        ${blur_flag} \
        ${folds_flag} \
        --tissue_threshold ${params.tissue_threshold} \
        --blur_threshold ${params.blur_threshold} \
        --mask_scale ${params.mask_scale} \
        --device ${params.device} \
        --batch_size ${params.batch_size} \
        --feature_extractor ${params.encoder} \
        --stream_features \
        --export_json_metrics
    
    # Normalize output names
    mv ${params.encoder}_features/${patient_id}.h5 ${patient_id}_features.h5
    mv ${params.encoder}_features/${patient_id}_metrics.json ${patient_id}_metrics.json
    """
}

/*
 * OMERO WRITE-BACK: Persist PAC Score, QC metrics, and diagnosis to OMERO Server
 */
process WRITE_OMERO_ANNOTATIONS {
    tag "${patient_id}"
    
    container 'slidelab:cpu'
    
    when:
    params.omero_write_back
    
    input:
    tuple val(patient_id), path(metrics_json)
    
    script:
    """
    python /app/SlideLab/integrations/omero_writer.py \
        --image_id \${OMERO_IMAGE_ID:-1} \
        --metrics_json ${metrics_json}
    """
}

/*
 * STEP 4: Aggregate reports and merge with clinical data
 */
process AGGREGATE_RESULTS {
    tag "aggregate"
    publishDir "${params.output_dir}", mode: 'copy'
    
    input:
    path tile_csvs
    path clinical_csv, stageAs: 'clinical.csv'
    
    output:
    path "aggregated_tiles.csv", emit: all_tiles
    path "pipeline_summary.csv", emit: summary
    path "training_manifest.csv", emit: manifest, optional: true
    
    script:
    def merge_clinical = params.clinical_csv ? "true" : "false"
    """
    #!/usr/bin/env python3
    import pandas as pd
    import glob
    import os
    
    # Aggregate all tile CSVs
    csv_files = glob.glob("*.csv")
    csv_files = [f for f in csv_files if f not in ['clinical.csv', 'aggregated_tiles.csv', 'pipeline_summary.csv', 'training_manifest.csv']]
    
    if csv_files:
        dfs = [pd.read_csv(f) for f in csv_files]
        all_tiles = pd.concat(dfs, ignore_index=True)
        all_tiles.to_csv("aggregated_tiles.csv", index=False)
        
        # Summary statistics
        summary = all_tiles.groupby("Patient_ID").agg(
            total_tiles=("x", "count"),
            unique_x=("x", "nunique"),
            unique_y=("y", "nunique")
        ).reset_index()
        summary.to_csv("pipeline_summary.csv", index=False)
        
        # Merge with clinical data if provided
        if ${merge_clinical} and os.path.exists("clinical.csv"):
            clinical = pd.read_csv("clinical.csv")
            # Try common join keys
            for join_key in ["Patient_ID", "patient_id", "subject_id", "person_id"]:
                if join_key in clinical.columns and join_key in all_tiles.columns:
                    manifest = all_tiles.merge(clinical, on=join_key, how="inner")
                    manifest.to_csv("training_manifest.csv", index=False)
                    print(f"Merged {len(manifest)} rows on {join_key}")
                    break
    else:
        pd.DataFrame().to_csv("aggregated_tiles.csv", index=False)
        pd.DataFrame().to_csv("pipeline_summary.csv", index=False)
    """
}

// ─── Workflow ────────────────────────────────────────────────────

workflow {
    // Step 1: Discover slides
    slide_ch = Channel
        .fromPath("${params.input_dir}/*.{${wsi_extensions.join(',')}}")
        .map { file ->
            def patient_id = file.baseName.replaceAll(/\.[^.]+$/, '')
            tuple(patient_id, file)
        }
    
    if (params.stream_features) {
        // High-performance Zero-Disk Streaming pipeline
        PROCESS_AND_ENCODE_WSI(slide_ch)
        
        if (params.omero_write_back) {
            WRITE_OMERO_ANNOTATIONS(PROCESS_AND_ENCODE_WSI.out.metrics)
        }
    } else {
        // Classical decoupled pipeline with intermediate disk tiles
        PREPROCESS_SLIDE(slide_ch)
        
        if (params.encode) {
            ENCODE_FEATURES(PREPROCESS_SLIDE.out.tiles)
        }
        
        clinical_ch = params.clinical_csv 
            ? Channel.fromPath(params.clinical_csv) 
            : Channel.of(file('NO_CLINICAL'))
        
        AGGREGATE_RESULTS(
            PREPROCESS_SLIDE.out.tile_csv.collect(),
            clinical_ch
        )
    }
}

// ─── Completion handler ──────────────────────────────────────────
workflow.onComplete {
    log.info """
    ═══════════════════════════════════════════════
     SlideLab Pipeline Complete
    ═══════════════════════════════════════════════
     Status    : ${workflow.success ? '✅ SUCCESS' : '❌ FAILED'}
     Duration  : ${workflow.duration}
     Output    : ${params.output_dir}
     Slides    : ${workflow.stats.succeededCount} processed
     Mode      : ${params.stream_features ? 'Zero-Disk Streaming (On-The-Fly)' : 'Decoupled Disk Tiling'}
    ═══════════════════════════════════════════════
    """.stripIndent()
}
