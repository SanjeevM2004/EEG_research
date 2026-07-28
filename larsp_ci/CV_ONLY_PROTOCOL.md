# LaRSP-CV: strict image-only protocol

This branch removes every molecular feature from the model path.

## Inputs permitted

- H&E RGB pixels from the public DLPFC low-resolution histology image.
- Visium spot coordinates, used only to crop patches and construct the within-section spatial graph.
- Layer labels from the two training donors, used for supervised contrastive fine-tuning and aligned training prototypes.

## Inputs forbidden

- `adata.X` or any spot-by-gene matrix.
- Highly variable genes, gene PCA, expression embeddings, marker genes, or expression-derived adjacency.
- Test-section layer labels during clustering, Riemannian alignment, prototype assignment, or prediction.

The H5AD container is opened in backed mode only to read spot barcodes and `obsm['spatial']`; the expression matrix is not materialized or accessed.

## Model path

1. Crop an H&E patch around every spot coordinate.
2. Train visual BYOL on patches from the two training donors.
3. Fine-tune the visual encoder with supervised contrastive learning on training-layer labels and freeze it.
4. Build visual-spatial affinities and infer 5--7 candidate laminar blocks with spectral clustering.
5. Compute visual block covariances.
6. Estimate one unlabeled Riemannian reference per donor and align as `C_RA = R_d^{-1/2} C R_d^{-1/2}`.
7. Compute aligned L1--L6/WM Karcher prototypes from training sections.
8. Assign held-out visual blocks by AIRM/MDM, propagate the soft evidence on the visual-spatial graph, and apply fixed local majority refinement.

## Evaluation

- Three strict leave-one-donor-out folds.
- Seeds 11, 42, and 73.
- Accuracy, macro-F1, ARI, NMI, boundary-F1, and ordinal MAE.
- Saved H&E overlays show the input image, evaluation labels, inferred spectral blocks, and final predicted layers.

## Controlled image-only baselines

- HOG + colour statistics + linear classifier.
- Frozen ImageNet ResNet-18 + linear classifier.
- Visual BYOL embedding + linear classifier.
- Spatial graph-smoothed variants of each classifier.
- RA-MDM and ordered RA-MDM ablations.

No previously reported expression-based result is reused as an image-only result.
