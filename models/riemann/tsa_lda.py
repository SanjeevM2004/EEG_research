from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace
from ._utils import get_metric_for_cov
from .dcra import DCRPreAligner

class TSALDA:
    """Tangent Space Adaptation (TSA) + LDA."""

    def __init__(self, cov_type: str = "RA"):
        self.cov_type = cov_type
        self.metric = get_metric_for_cov(cov_type)
        self.ts = TangentSpace(metric=self.metric)
        self.scaler = StandardScaler()
        self.clf = LDA(solver='lsqr', shrinkage='auto')

    def fit(self, X_train, y_train, X_calib=None):
        Z = self.ts.fit_transform(X_train)
        #Z = self.scaler.fit_transform(Z)
        self.clf.fit(Z, y_train)
        return self

    def predict(self, X_test, X_calib=None):
        Z = self.ts.transform(X_test)
        #Z = self.scaler.transform(Z)
        return self.clf.predict(Z)

    def score(self, X_test, y_test, X_calib=None):
        return (self.predict(X_test) == y_test).mean()
