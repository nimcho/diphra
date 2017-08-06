
import diphra
try:
    import h5py
except ImportError:
    h5py = None

from os import remove, path
from tempfile import mkstemp
from diphra.pocs import VerticalPOcs, HDF5POcs

diphra_path = path.dirname(diphra.__file__)
test_data_path = path.join(diphra_path, "../tests/test_data/")


def test_hdf5_pocs():

    assert h5py is not None

    vertical_pocs = VerticalPOcs(path.join(test_data_path, "10k_sorted_pocs.vert.txt"))

    _, fn_temp = mkstemp()
    try:
        HDF5POcs.build(vertical_pocs, output_name=fn_temp)

        hdf5_pocs = HDF5POcs(fn_temp)
        for poc1, poc2 in zip(vertical_pocs, hdf5_pocs):
            assert poc1 == poc2
    finally:
        remove(fn_temp)
