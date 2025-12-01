from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from .dcra import DCRPreAligner
class CSP_LDA:
    """
    Common Spatial Patterns + LDA.
    Works on raw epochs (N, C, T).
    """
    def __init__(self, n_components: int = 6, reg="oas", log=True):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.csp = CSP(n_components=n_components, reg=self.reg, log=self.log)
        self.clf = LDA()

    def fit(self, X, y, domains=None):
        if X.ndim != 3 or X.shape[1] >= X.shape[2]:
            raise ValueError("CSP_LDA expects raw epochs (N,C,T). You passed covariances.")
        Z = self.csp.fit_transform(X, y)
        self.clf.fit(Z, y)
        return self

    def predict(self, X, domains=None):
        Z = self.csp.transform(X)
        return self.clf.predict(Z)

    def score(self, X, y, domains=None):
        return (self.predict(X) == y).mean()
