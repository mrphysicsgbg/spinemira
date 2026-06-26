from pathlib import Path
from unittest import TestCase

from spinemira.io.mids import Layout


class TestMidsLayout(TestCase):
    """Test Layout using example dataset."""

    def setUp(self):
        """Configure Dataset using example dataset."""
        self.layout = Layout(
            root=Path(__file__).parent / "../examples/spider_dataset",
            include_derivatives=True,
        )
        self.layout.index(load_sidecars=True)

    def test_len_raw_images(self):
        """Test number of resolved raw images."""
        self.assertEqual(
            len(
                self.layout.query("`source` == 'raw' and `file_extension` == '.nii.gz'")
            ),
            6,
        )

    def test_find_raw(self):
        """Test resolving raw entries from derivatives."""
        df_derivative = self.layout.query(
            "`source` == 'derivative' and `file_extension` == '.nii.gz'"
        )
        df_raw = self.layout.find_raw(df_derivative)
        self.assertEqual(len(df_derivative), len(df_raw))
