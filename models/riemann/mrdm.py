import numpy as np
from sklearn.model_selection import KFold
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from ._utils import get_metric_for_cov

class RiemannMRDM:
    """
    MRDM - Multi Riemannian Distance to Mean.
    Uses multiple local class means for finer geometry-aware classification.
    """

    def __init__(self, cov_type: str = "RA", n_submeans=3, n_iter=5, use_kfold=True):
        self.cov_type = cov_type
        self.metric = get_metric_for_cov(cov_type)
        self.n_submeans = n_submeans
        self.n_iter = n_iter
        self.use_kfold = use_kfold
        self.submeans_ = {}
        self.classes_ = None

    def _to_numpy(self, X):
        if isinstance(X, list):
            X = np.stack([c.detach().cpu().numpy() if hasattr(c, "detach") else np.array(c) for c in X])
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        return X

    def _generate_submeans(self, Xc):
        Xc = self._to_numpy(Xc)
        if len(Xc) <= self.n_submeans:
            return np.stack([mean_riemann(Xc)])
        if self.use_kfold:
            kf = KFold(n_splits=self.n_submeans, shuffle=True, random_state=42)
            submeans = [mean_riemann(Xc[idx]) for _, idx in kf.split(Xc)]
            return np.stack(submeans)
        # fallback: Riemannian k-means refinement
        n_clusters = min(self.n_submeans, len(Xc))
        idx = np.random.choice(len(Xc), n_clusters, replace=False)
        centers = [Xc[i] for i in idx]
        for _ in range(self.n_iter):
            clusters = [[] for _ in range(n_clusters)]
            for C in Xc:
                dists = [distance_riemann(C, G) for G in centers]
                clusters[np.argmin(dists)].append(C)
            centers = [
                mean_riemann(self._to_numpy(c)) if len(c) > 0 else Xc[np.random.randint(len(Xc))]
                for c in clusters
            ]
        return np.stack(centers)

    def fit(self, X, y, domains=None):
        X = self._to_numpy(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.submeans_ = {c: self._generate_submeans(X[y == c]) for c in self.classes_}
        return self

    def predict(self, X, domains=None):
        X = self._to_numpy(X)
        preds = []
        for C in X:
            best_c, best_d = None, np.inf
            for c in self.classes_:
                d_min = np.min([distance_riemann(C, G) for G in self.submeans_[c]])
                if d_min < best_d:
                    best_d, best_c = d_min, c
            preds.append(best_c)
        return np.array(preds)

    def score(self, X, y, domains=None):
        return np.mean(self.predict(X) == np.array(y))
