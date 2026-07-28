from pathlib import Path

path = Path('larsp_ci/experiment.py')
source = path.read_text()

source = source.replace(
    "'spatialLIBD', 'Ground Truth', 'ground_truth', 'label', 'layer',",
    "'spatialLIBD', 'Region', 'region', 'Ground Truth', 'ground_truth', 'label', 'layer',",
)

anchor = """def independent_assignment(cluster_covs: Sequence[np.ndarray], prototypes: Sequence[np.ndarray]) -> np.ndarray:
    cost = np.asarray([[airm_distance(c, p) for p in prototypes] for c in cluster_covs])
    return np.argmin(cost, axis=1).astype(int)


"""
addition = r'''def _robust_cost_scale(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=float)
    shifted = cost - np.min(cost, axis=1, keepdims=True)
    positive = shifted[shifted > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.0
    return shifted / max(scale, 1e-12)


def graph_smooth_probabilities(
    affinity: csr_matrix,
    probabilities: np.ndarray,
    alpha: float = 0.65,
    steps: int = 10,
) -> np.ndarray:
    base = np.maximum(np.asarray(probabilities, dtype=float), 1e-8)
    base /= base.sum(axis=1, keepdims=True)
    degrees = np.asarray(affinity.sum(axis=1)).ravel()
    inverse = np.divide(1.0, degrees, out=np.zeros_like(degrees), where=degrees > 0)
    transition = affinity.multiply(inverse[:, None]).tocsr()
    current = base.copy()
    for _ in range(max(int(steps), 0)):
        current = (1.0 - alpha) * base + alpha * transition.dot(current)
        current = np.maximum(current, 1e-8)
        current /= current.sum(axis=1, keepdims=True)
    return current


def semantic_riemannian_mapping(
    cluster_covs: Sequence[np.ndarray],
    prototypes: Sequence[np.ndarray],
    cluster_probabilities: np.ndarray,
    order: np.ndarray,
    ordered: bool = True,
    airm_weight: float = 0.15,
) -> np.ndarray:
    semantic = _robust_cost_scale(-np.log(np.clip(cluster_probabilities, 1e-8, 1.0)))
    if airm_weight > 0:
        geometry = np.asarray([
            [airm_distance(covariance, prototype) ** 2 for prototype in prototypes]
            for covariance in cluster_covs
        ])
        combined = semantic + float(airm_weight) * _robust_cost_scale(geometry)
    else:
        combined = semantic
    n_clusters, n_labels = combined.shape
    if not ordered:
        return np.argmin(combined, axis=1).astype(int)
    best_mapping, best_cost = None, float('inf')
    for subset in combinations(range(n_labels), n_clusters):
        forward = np.asarray(subset, dtype=int)
        for labels in (forward, forward[::-1]):
            mapping = np.empty(n_clusters, dtype=int)
            mapping[order] = labels
            total = float(sum(combined[i, mapping[i]] for i in range(n_clusters)))
            if total < best_cost:
                best_mapping, best_cost = mapping.copy(), total
    if best_mapping is None:
        raise RuntimeError('No semantic--Riemannian assignment')
    return best_mapping


def calibrated_partition_prediction(
    clusters: np.ndarray,
    order: np.ndarray,
    affinity: csr_matrix,
    test_z_cov: np.ndarray,
    prototypes: Sequence[np.ndarray],
    spot_probabilities: np.ndarray,
    ordered: bool = True,
    airm_weight: float = 0.15,
    prior_strength: float = 0.55,
) -> Tuple[np.ndarray, np.ndarray]:
    smoothed = graph_smooth_probabilities(affinity, spot_probabilities)
    n_clusters = len(np.unique(clusters))
    covariances = [shrinkage_covariance(test_z_cov[clusters == c]) for c in range(n_clusters)]
    cluster_probabilities = np.stack([
        np.mean(smoothed[clusters == c], axis=0) for c in range(n_clusters)
    ])
    cluster_probabilities = np.maximum(cluster_probabilities, 1e-8)
    cluster_probabilities /= cluster_probabilities.sum(axis=1, keepdims=True)
    mapping = semantic_riemannian_mapping(
        covariances, prototypes, cluster_probabilities, order,
        ordered=ordered, airm_weight=airm_weight,
    )
    assigned = np.full_like(
        cluster_probabilities, 0.04 / max(cluster_probabilities.shape[1] - 1, 1)
    )
    assigned[np.arange(n_clusters), mapping] = 0.96
    cluster_prior = 0.55 * cluster_probabilities + 0.45 * assigned
    cluster_prior /= cluster_prior.sum(axis=1, keepdims=True)
    confidence = np.max(smoothed, axis=1)
    adaptive = np.clip(float(prior_strength) * (1.0 - confidence), 0.0, 0.40)[:, None]
    posterior = (1.0 - adaptive) * smoothed + adaptive * cluster_prior[clusters]
    posterior /= posterior.sum(axis=1, keepdims=True)
    return np.argmax(posterior, axis=1).astype(int), posterior


def classifier_probabilities(model: LogisticRegression, features: np.ndarray, n_classes: int = 7) -> np.ndarray:
    partial = model.predict_proba(features)
    output = np.full((len(features), n_classes), 1e-8, dtype=float)
    output[:, model.classes_.astype(int)] = partial
    output /= output.sum(axis=1, keepdims=True)
    return output


'''
if anchor not in source:
    raise RuntimeError('independent_assignment anchor not found')
source = source.replace(anchor, anchor + addition)

start = source.index('def partition_clusters(')
end = source.index('\n\ndef assign_partition(', start)
source = source[:start] + '''def partition_clusters(
    z_graph: np.ndarray,
    coords: np.ndarray,
    seed: int,
    graph_mode: str = 'spatial',
    candidates: Sequence[int] = (5, 6, 7),
) -> Tuple[np.ndarray, np.ndarray, int, csr_matrix]:
    if graph_mode == 'spatial':
        affinity = build_sparse_affinity(z_graph, coords)
    elif graph_mode == 'embedding':
        affinity = build_embedding_affinity(z_graph)
    else:
        raise ValueError(graph_mode)
    n_clusters = select_cluster_count(affinity, candidates)
    clusters = merge_tiny_clusters(spectral_partition(affinity, n_clusters, seed), coords)
    n_clusters = len(np.unique(clusters))
    order = fiedler_cluster_order(affinity, clusters, coords)
    return clusters, order, n_clusters, affinity
''' + source[end:]
source = source.replace(
    'clusters, order, n_clusters = partition_clusters(test_z_graph, coords, seed, graph_mode, candidates)',
    'clusters, order, n_clusters, _ = partition_clusters(test_z_graph, coords, seed, graph_mode, candidates)',
)

block_start = source.index("            predictions: Dict[str, np.ndarray] = {")
block_end = source.index("            for method, prediction in predictions.items():", block_start)
replacement = '''            pca_probabilities = classifier_probabilities(pca_logreg, pca_test)
            supcon_probabilities = classifier_probabilities(supcon_logreg, z_byol)
            byol_probabilities = classifier_probabilities(byol_logreg, z_byol_only)

            predictions: Dict[str, np.ndarray] = {
                'PCA-LogReg': np.argmax(pca_probabilities, axis=1),
                'BYOL-LogReg': np.argmax(byol_probabilities, axis=1),
                'SupCon-LogReg': np.argmax(supcon_probabilities, axis=1),
                'Euclidean-Prototype': nearest_centroid_predict(
                    z_train[valid_train], y_train[valid_train], z_byol
                ),
            }

            spatial_clusters, spatial_order, full_k, spatial_affinity = partition_clusters(
                z_byol, section.coords, args.seed, 'spatial'
            )
            graph_probabilities = graph_smooth_probabilities(
                spatial_affinity, supcon_probabilities
            )
            predictions['SupCon-GraphSmooth'] = np.argmax(graph_probabilities, axis=1)
            full_pred, _ = calibrated_partition_prediction(
                spatial_clusters, spatial_order, spatial_affinity, z_cov,
                covariance_prototypes, supcon_probabilities,
                ordered=True, airm_weight=0.15, prior_strength=0.55,
            )
            semantic_only_pred, _ = calibrated_partition_prediction(
                spatial_clusters, spatial_order, spatial_affinity, z_cov,
                covariance_prototypes, supcon_probabilities,
                ordered=True, airm_weight=0.0, prior_strength=0.55,
            )
            no_order_pred, _ = calibrated_partition_prediction(
                spatial_clusters, spatial_order, spatial_affinity, z_cov,
                covariance_prototypes, supcon_probabilities,
                ordered=False, airm_weight=0.15, prior_strength=0.55,
            )

            embedding_clusters, embedding_order, no_spatial_k, embedding_affinity = partition_clusters(
                z_byol, section.coords, args.seed, 'embedding'
            )
            no_spatial_pred, _ = calibrated_partition_prediction(
                embedding_clusters, embedding_order, embedding_affinity, z_cov,
                covariance_prototypes, supcon_probabilities,
                ordered=True, airm_weight=0.15, prior_strength=0.55,
            )

            pca_graph = pca_test[:, :min(args.embedding_dim, pca_test.shape[1])]
            pca_cov_test = pca_test[:, :pca_cov_dim]
            pca_clusters, pca_order, no_byol_k, pca_affinity = partition_clusters(
                pca_graph, section.coords, args.seed, 'spatial'
            )
            no_byol_pred, _ = calibrated_partition_prediction(
                pca_clusters, pca_order, pca_affinity, pca_cov_test,
                pca_cov_prototypes, pca_probabilities,
                ordered=True, airm_weight=0.15, prior_strength=0.55,
            )
            airm_only_pred = assign_partition(
                spatial_clusters, spatial_order, z_cov,
                covariance_prototypes, 'ordered_airm'
            )

            predictions.update({
                'LaRSP-Full': full_pred,
                'LaRSP-NoRiemann': semantic_only_pred,
                'LaRSP-NoOrder': no_order_pred,
                'LaRSP-NoSpatial': no_spatial_pred,
                'LaRSP-NoBYOL': no_byol_pred,
                'LaRSP-AIRMOnly': airm_only_pred,
            })
            cluster_counts = {
                'LaRSP-Full': full_k,
                'LaRSP-NoRiemann': full_k,
                'LaRSP-NoOrder': full_k,
                'LaRSP-NoSpatial': no_spatial_k,
                'LaRSP-NoBYOL': no_byol_k,
                'LaRSP-AIRMOnly': full_k,
            }
'''
source = source[:block_start] + replacement + source[block_end:]
path.write_text(source)
print(f'patched {path} ({len(source)} bytes)')
