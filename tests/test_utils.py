
import diphra

from os import remove, path
from tempfile import mkstemp
from diphra.pocs import VerticalPOcs

diphra_path = path.dirname(diphra.__file__)
test_data_path = path.join(diphra_path, "../tests/test_data/")


def test_pocs_sort():

    nb_pocs = 10000
    line_chunk_size = 1000
    file_chunk_size = 3

    fn_unsorted = path.join(test_data_path, "10k_unsorted_pocs.vert")
    fn_sorted = path.join(test_data_path, "10k_sorted_pocs.vert")
    _, fn_temp = mkstemp()

    VerticalPOcs.pocs_sort(
        inp_fn=fn_unsorted,
        out_fn=fn_temp,
        line_chunk_size=int(line_chunk_size),
        file_chunk_size=file_chunk_size
    )
    pocs_sorted = list(VerticalPOcs(fn_sorted))
    pocs_temp = list(VerticalPOcs(fn_temp))

    assert len(pocs_sorted) == nb_pocs
    assert len(pocs_sorted) == len(pocs_temp)

    for i in range(nb_pocs):
        # Do not compare phrases as we do not need stable sort.
        assert pocs_sorted[i][1] == pocs_temp[i][1]
        assert pocs_sorted[i][2] == pocs_temp[i][2]

    remove(fn_temp)
