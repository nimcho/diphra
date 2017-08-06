#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Lukas Banic <lukas.banic@protonmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
import sys

from collections import defaultdict
from os.path import exists, getsize
from numpy import zeros, uint32, iinfo, full, array

from .utils import collate, external_sort
from .reader import our_open

MASK_VALUE = iinfo(uint32).max

try:
    import h5py
except ImportError:
    h5py = None


def line2poc(line):
    parts = line.split("\t")
    return (
        parts[0],                   # phrase
        int(parts[1]),              # sentence ID
        tuple(map(int, parts[2:]))  # in-sentence positions
    )


def poc2line(poc):
    phrase = (
        poc[0].encode("utf-8")
        if type(poc[0]) == unicode
        else str(poc[0])
    )
    return "{phrase}\t{sentence_id}\t{positions}".format(
        phrase=phrase,
        sentence_id=str(poc[1]),
        positions="\t".join(map(str, poc[2]))
    )


def line2poc_(line):
    # sentence ID is merged with token positions
    parts = line.split("\t")
    return (
        parts[0],
        tuple(map(int, parts[1:]))
    )


def poc2line_(poc):
    # sentence ID is merged with token positions
    phrase = (
        poc[0].encode("utf-8")
        if type(poc[0]) == unicode
        else str(poc[0])
    )
    return "{phrase}\t{positions}".format(
        phrase=phrase,
        positions="\t".join(map(str, poc[1]))
    )


def pocs2vocab(pocs):
    """
    # Returns
        {phrase: i} dictionary
        array of frequencies
        max. number of positions
    """
    freq_dict = defaultdict(lambda: 0)
    max_len = 1
    for phrase, _, positions in pocs:
        freq_dict[phrase] += 1
        if len(positions) > max_len:
            max_len = len(positions)

    sorted_vocab = sorted(
        freq_dict.items(),
        key=lambda (p, f): f,
        reverse=True
    )
    phrase2i = {
        phrase: i for i, (phrase, freq)
        in enumerate(sorted_vocab)
    }
    freqs = array([freq for phrase, freq in sorted_vocab], dtype=uint32)

    return phrase2i, freqs, max_len


class POcs(object):

    @staticmethod
    def read_from_vertical(filename):
        with our_open(filename, buffering=int(16 * 1024**2)) as pocs_file:
            for line in pocs_file:
                yield line2poc(line.strip())

    def __init__(self, func):
        self.func = func

    def __iter__(self):
        return self.func()

    def save_to_vertical(self, filename):
        with our_open(filename, "w", buffering=int(16 * 1024**2)) as pocs_file:
            for poc in self:
                pocs_file.write(poc2line(poc) + "\n")


class VerticalPOcs(POcs):

    @staticmethod
    def pocs_sort(inp_fn, out_fn,
                  line_chunk_size=int(20e6),
                  file_chunk_size=120):
        """
        External sort for vertical pocs.
        """
        external_sort(
            inp_fn=inp_fn, out_fn=out_fn,
            line2item=line2poc_, item2line=poc2line_,
            key=lambda poc: poc[1],
            line_chunk_size=line_chunk_size,
            file_chunk_size=file_chunk_size
        )

    def __init__(self, filename):
        super(VerticalPOcs, self).__init__(
            func=lambda: POcs.read_from_vertical(filename)
        )


class MergedPOcs(POcs):

    def __init__(self, *pocs_objects):
        super(MergedPOcs, self).__init__(
            func=lambda: collate(
                *pocs_objects,
                key=(
                    lambda (phrase, sentence_id, positions):
                    (sentence_id, positions)
                )
            )
        )


class IDPOcs(POcs):

    def __init__(self, pocs):
        self.pocs = pocs
        self.phrase2i = None
        super(IDPOcs, self).__init__(self.iter_and_count)

    def iter_and_count(self):
        phrase2i = dict() if self.phrase2i is None else self.phrase2i
        for phrase, sentence_i, positions in self.pocs:
            i = phrase2i.get(phrase, None)
            if i is None:
                i = len(phrase2i)
                phrase2i[phrase] = i
            yield i, sentence_i, positions
        self.phrase2i = phrase2i


class HDF5POcs(POcs):

    @staticmethod
    def build(pocs, output_name, max_len=None, use_ids=False):

        if h5py is None:
            raise RuntimeError("`h5py` not installed.")

        if exists(output_name) and getsize(output_name) > 0:
            raise ValueError(
                "File `%s` already exists" % output_name
            )

        if max_len is None:
            sys.stdout.write(
                "Max. length not specified, iterating over "
                "pocs to find it out.\n"
            )
            max_len = 1
            for poc in pocs:
                if len(poc[2]) > max_len:
                    max_len = len(poc[2])

        if not use_ids:
            pocs = IDPOcs(pocs)

        nb_cols = 2 + max_len  # phrase ID, sentence ID, positions
        batch_size = 2 ** 20
        batch = full(
            fill_value=MASK_VALUE,
            shape=(batch_size, nb_cols),
            dtype=uint32
        )
        h5_file = h5py.File(output_name, "w")
        h5_file.create_dataset(
            name="pocs",
            shape=(0, nb_cols),
            maxshape=(None, nb_cols),
            dtype=uint32
        )
        h5_file.create_dataset(
            name="boundaries",
            shape=(0, ),
            maxshape=(None, ),
            dtype=uint32
        )

        def append_batch():
            beg = h5_file["pocs"].shape[0]
            end = beg + bp
            h5_file["pocs"].resize((end, nb_cols))
            h5_file["pocs"][beg:end] = batch[:bp]

        def append_boundaries():
            beg = h5_file["boundaries"].shape[0]
            end = beg + len(boundaries)
            h5_file["boundaries"].resize((end, ))
            h5_file["boundaries"][beg:end] = boundaries

        current_sentence = None
        boundaries = list()
        max_phrase_i = -1
        freqs = zeros(1024, dtype=uint32)  # will expand as needed
        bp = 0  # batch position
        i = 0  # valid poc number

        for poc in pocs:

            phrase_i = int(poc[0])
            if len(poc[2]) > max_len:
                continue
            batch[bp][0] = phrase_i
            batch[bp][1] = poc[1]
            batch[bp][2:2 + len(poc[2])] = poc[2]

            if phrase_i > max_phrase_i:
                max_phrase_i = phrase_i
                if max_phrase_i >= len(freqs):
                    # rather more, so that we quickly
                    # get a stable shape:
                    freqs.resize((max_phrase_i * 2, ))

            freqs[phrase_i] += 1

            if current_sentence != poc[1]:
                if i != 0:
                    boundaries.append(i)
                    if len(boundaries) >= batch_size:
                        append_boundaries()
                        boundaries = list()  # reset
                current_sentence = poc[1]

            bp += 1
            if bp >= batch_size:
                sys.stdout.write(
                    "Writing pocs to disk, currently at {:,}\n".format(i)
                )
                append_batch()
                batch.fill(MASK_VALUE)  # reset
                bp = 0

            i += 1

        # --- end of pocs iteration --- #

        nb_pocs = i
        boundaries.append(nb_pocs)
        append_boundaries()

        if bp > 0:  # last batch
            append_batch()

        if not use_ids:
            phrases = [
                p for p, pi
                in sorted(
                    pocs.phrase2i.items(),
                    key=lambda (_p, _pi): _pi
                )
            ]
            h5_file["phrases"] = phrases

        h5_file["freqs"] = freqs[:max_phrase_i + 1]
        h5_file.close()

    def __init__(self, filename):
        if h5py is None:
            raise RuntimeError("`h5py` not installed.")
        self.filename = filename
        self.file = h5py.File(filename)
        nb_pocs = len(self.file["pocs"])
        super(HDF5POcs, self).__init__(lambda: self.read_pocs(0, nb_pocs))
        self.i2phrase = None
        if "phrases" in self.file:
            self.i2phrase = dict()
            for i, phrase in enumerate(self.file["phrases"].value):
                self.i2phrase[i] = phrase

    def _read_rows(self, beg, end, cache_size=int(1e6), add=0):
        for xbeg in range(beg, end, cache_size):
            xend = min(end, xbeg + cache_size)
            data = self.file["pocs"][xbeg:xend]
            if add != 0:
                data += add
            for i in range(len(data)):
                yield data[i]
            del data

    def read_pocs(self, beg=None, end=None, cache_size=int(1e6)):
        for row in self._read_rows(beg, end, cache_size):
            phrase_i = row[0]
            sentence_i = row[1]
            positions = row[2:][row[2:] != MASK_VALUE]
            phrase = (
                self.i2phrase[phrase_i]
                if self.i2phrase is not None
                else phrase_i
            )
            yield phrase, sentence_i, tuple(positions)

    def __del__(self):
        self.file.close()

