import warnings
from spinemira.io.mids import *  # noqa: F403

warnings.warn(
    "'bids' is deprecated; import 'mids' instead.",
    DeprecationWarning,
    stacklevel=2,
)
