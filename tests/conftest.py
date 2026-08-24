import os

import anndata


os.environ.setdefault("MPLBACKEND", "Agg")
anndata.settings.allow_write_nullable_strings = True
