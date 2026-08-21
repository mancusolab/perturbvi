from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

from .analysis import analyze as analyze
from .infer import (
    compute_elbo as compute_elbo,
    compute_pip as compute_pip,
    compute_pve as compute_pve,
    infer as infer,
)
from .io import load_results as load_results, save_results as save_results
from .loaders import load_screen as load_screen
from .preprocess import residualize_screen as residualize_screen
from .screen import fit_screen as fit_screen, ScreenData as ScreenData
from .sim import generate_sim as generate_sim


__all__ = (
    "ScreenData",
    "__version__",
    "analyze",
    "compute_elbo",
    "compute_pip",
    "compute_pve",
    "fit_screen",
    "generate_sim",
    "infer",
    "load_results",
    "load_screen",
    "residualize_screen",
    "save_results",
)


try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError, dist_name
