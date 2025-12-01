import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace
from ._utils import get_metric_for_cov
from .wrpa import WRPA
from .prpa import PolarRPA
from .rpa import RPA

# ============================================================
# Full pipeline: RPA_WU + TangentSpace + LogisticRegression
# ============================================================

class RiemannRPA_WU_TSLR:
    def __init__(self, cov_type="RA", C=1.0,
                 rpa_max_iter=250, rpa_lr=5e-2, seed=0,
                 lr_class_weight=None):
        self.cov_type = cov_type
        self.metric = cov_type if cov_type != "RA" else "riemann"
        self.ts = TangentSpace(metric=self.metric)
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(max_iter=2000, n_jobs=-1, C=C,
                                     class_weight=lr_class_weight)
        self.wrpa = WRPA(max_iter=rpa_max_iter, lr=rpa_lr, seed=seed)
        self.prpa = PolarRPA(seed=seed)
        self.rpa = RPA(max_iter=rpa_max_iter, lr=rpa_lr, seed=seed)

    def fit(self, X, y, domains=None):
        X_align = self.prpa.fit_transform(np.asarray(X), y=None, domains=domains)
        Z = self.ts.fit_transform(X_align)
        Z = self.scaler.fit_transform(Z)
        #Z = X_align.reshape(X_align.shape[0], -1)
        self.lr.fit(Z, y)
        return self

    def predict(self, X, domains=None):
        X_align = self.prpa.transform(np.asarray(X), domains=domains)
        Z = self.ts.transform(X_align)
        Z = self.scaler.transform(Z)
        #Z = X_align.reshape(X_align.shape[0], -1)
        return self.lr.predict(Z)

    def score(self, X, y, domains=None):
        yhat = self.predict(X, domains=domains)
        return (yhat == y).mean()