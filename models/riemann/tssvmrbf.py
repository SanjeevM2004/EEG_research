from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace
from ._utils import get_metric_for_cov
#from .urpa import URPA
#from .dcra import DCRPreAligner


class RiemannTS_SVM_RBF:
    """
    Tangent-Space → SVM with RBF kernel.
    Equivalent to TSLR, but RBF SVM replaces Logistic Regression.
    """

    def __init__(self,
                 cov_type: str = "RA",
                 C: float = 1.0,
                 gamma: str = "scale"):
        """
        Args:
            cov_type: Metric for Riemannian alignment (RA, EA, LEA, etc.)
            C:       SVM penalty parameter
            gamma:   "scale", "auto", or float
        """
        self.cov_type = cov_type
        self.metric = get_metric_for_cov(cov_type)
        self.ts = TangentSpace(metric=self.metric)

        # Standardizer recommended before SVM-RBF
        self.scaler = StandardScaler()

        # RBF SVM classifier
        self.svm = SVC(kernel="rbf", C=C, gamma=gamma)

        # Optional URPA or DCR (commented)
        #self.dcr = DCRPreAligner(steps=1000, lr=1e-3)
        #self.urpa = URPA(split_ratio=0.8, seed=42)


    # --------------------------------------------------------
    def fit(self, X, y, domains=None):
        """
        Fit Tangent-Space SVM (with optional URPA_C preprocessing).
        """
        X_proc = X
        
        # Optional:
        #X_proc = self.urpa.fit_transform(X_proc, y=y, domains=domains)

        # Tangent space projection
        Z = self.ts.fit_transform(X_proc)

        # Scaling improves SVM RBF greatly
        Zs = self.scaler.fit_transform(Z)

        # Train SVM
        self.svm.fit(Zs, y)
        return self


    # --------------------------------------------------------
    def predict(self, X, domains=None):
        """
        Predict labels for new covariance matrices.
        """
        X_proc = X
        
        # Optional:
        #X_proc = self.urpa.transform(X_proc, domains=domains)

        Z = self.ts.transform(X_proc)
        Zs = self.scaler.transform(Z)
        return self.svm.predict(Zs)


    # --------------------------------------------------------
    def score(self, X, y, domains=None):
        y_pred = self.predict(X, domains=domains)
        return (y_pred == y).mean()


    def __repr__(self):
        return (f"RiemannTS_SVM_RBF(cov_type='{self.cov_type}', "
                f"C={self.svm.C}, gamma={self.svm.gamma})")
