from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

# annoying lint bug...
from .infer import (
    compute_elbo as compute_elbo,
    compute_pip as compute_pip,
    compute_pve as compute_pve,
    infer as infer,
)

from .sim import (generate_sim as generate_sim)
from .utils import (compute_lfsr as compute_lfsr)

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
