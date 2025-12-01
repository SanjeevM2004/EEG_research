def get_metric_for_cov(cov_type: str):
    """Return appropriate Riemannian metric based on covariance alignment."""
    cov_type = cov_type.upper()
    if cov_type == "LEA":
        return "logeuclid"
    else:
        return "riemann"
