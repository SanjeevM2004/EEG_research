from pyriemann.classification import MDM
from ._utils import get_metric_for_cov
from .dcra import DCRPreAligner

class RiemannMDM:
    """Classic MDM classifier on SPD covariance matrices."""

    def __init__(self, cov_type: str = "RA"):
        self.cov_type = cov_type
        self.metric = get_metric_for_cov(cov_type)
        self.clf = MDM(metric=self.metric)

    def fit(self, X, y, domains=None):
        #X = self.dcr.fit_transform(X, y=y, verbose=True)
        self.clf.fit(X, y)
        return self

    def predict(self, X, domains=None):
        #X = self.dcr.transform(X)
        return self.clf.predict(X)

    def score(self, X, y, domains=None):
        return (self.predict(X) == y).mean()
