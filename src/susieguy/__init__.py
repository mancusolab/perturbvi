from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

from susieguy import infer_design_matrix, infer_spike_slab, common, sparse


__all__ = ["common", "infer_design_matrix", "infer_spike_slab", "sparse"]

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
