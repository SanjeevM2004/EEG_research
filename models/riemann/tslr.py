from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace
from ._utils import get_metric_for_cov
from .urpa import URPA   # use your latest dispersion+canonicalization URPA_C
from .dcra import DCRPreAligner

class RiemannTSLR:
    """
    SPD → Tangent Space → Standardize → LogisticRegression.

    Optionally applies URPA_C preprocessing before tangent projection.
    """

    def __init__(self, cov_type: str = "RA", C: float = 1.0):
        """
        Args:
            cov_type:  Type of covariance preprocessing (e.g., "RA", "EA", etc.)
            C:         Regularization constant for logistic regression
            use_urpa:  If True, perform URPA_C alignment before TSLR
        """
        self.cov_type = cov_type
        self.metric = get_metric_for_cov(cov_type)
        self.ts = TangentSpace(metric=self.metric)
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(max_iter=2000, C=C)

        # optional URPA preprocessing
        #self.dcr = DCRPreAligner(steps=1000, lr=1e-3)
        #self.urpa = URPA(split_ratio=0.8, seed=42) if use_urpa else None

    # --------------------------------------------------------
    def fit(self, X, y, domains=None):
        """
        Fit TSLR classifier (with optional URPA_C preprocessing).
        """
        X_proc = X
        #X_proc = self.urpa.fit_transform(X, y=y, domains=domains)
        Z = self.ts.fit_transform(X_proc)
        #Z = self.scaler.fit_transform(Z)
        self.lr.fit(Z, y)
        return self

    # --------------------------------------------------------
    def predict(self, X, domains=None):
        """
        Predict labels for new samples (with optional URPA_C preprocessing).
        """
        X_proc = X
        #    X_proc = self.urpa.transform(X, domains=domains)
        Z = self.ts.transform(X_proc)
        #Z = self.scaler.transform(Z)
        return self.lr.predict(Z)

    # --------------------------------------------------------
    def score(self, X, y, domains=None):
        y_pred = self.predict(X, domains=domains)
        return (y_pred == y).mean()

    def __repr__(self):
        return f"RiemannTSLR(cov_type='{self.cov_type}', C={self.lr.C}) + DCR"
