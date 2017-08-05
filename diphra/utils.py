#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Lukas Banic <lukas.banic@protonmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import sys

from collections import defaultdict
from os.path import splitext, exists
from os import rename, remove

from more_itertools import peekable

from .reader import our_open


class SelfClosingFile(object):
    """
    Closes itself when deleted, even if an exception occurs.

    Use whenever you need to read/write simultaneously
    from/to multiple files without using try/catch block.
    """

    def __init__(self, filename, *args, **kwargs):
        self.file = our_open(filename, *args, **kwargs)

    def __del__(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def __iter__(self):
        return peekable(line for line in self.file)

    def write(self, s):
        self.file.write(s)

    def close(self):
        self.file.close()


def collate(*iterables, **kwargs):
    """
    A slightly faster version of more_itertools.collate
    """
    key = kwargs.get("key", lambda x: x)

    peekables = [peekable(it) for it in iterables]
    peekables = [pee for pee in peekables if pee]  # remove empties
    vals = [key(pee.peek()) for pee in peekables]

    while len(peekables) > 0:

        min_i = 0
        min_val = vals[0]
        for i, val in enumerate(vals):
            if val < min_val:
                min_i = i
                min_val = val

        yield peekables[min_i].next()

        if not peekables[min_i]:
            peekables = [pee for pee in peekables if pee]  # remove empties
            vals = [key(pee.peek()) for pee in peekables]
        else:
            vals[min_i] = key(peekables[min_i].peek())


def chunked(it, n):
    """
    Like more_itertools.chunked, only without memory leaks (??).
    """
    l = list()
    for x in it:
        l.append(x)
        if len(l) >= n:
            yield l
            l = list()
    if len(l) > 0:
        yield l


def external_sort(
        inp_fn, out_fn, key,
        line2item, item2line,
        line_chunk_size=int(20e6),
        file_chunk_size=120
):
    """
    Memory-friendly line-based sort for huge files.
    """
    assert file_chunk_size >= 2  # otherwise infinite looping
    assert line_chunk_size >= int(1e3)

    name_root, name_ext = splitext(out_fn)

    def get_fn(label):
        # Names of temporary files
        return "{name_root}-{label}{name_ext}".format(
            name_root=name_root,
            label=label,
            name_ext=name_ext
        )

    def read_items(_fn):
        with our_open(_fn) as f:
            for line in f:
                yield line2item(line.strip())

    def write_items(_items, _fn):
        with our_open(_fn, "w") as f:
            for item in _items:
                f.write("%s\n" % item2line(item))

    # PHASE 1 / Divide the file into small chunks
    # and sort each one of them separately.

    chunk_i = 0
    items = read_items(inp_fn)
    chunks = chunked(items, line_chunk_size)
    for chunk in chunks:
        beg = chunk_i * line_chunk_size
        end = beg + line_chunk_size
        sys.stdout.write("Sorting lines {:,} - {:,}\n".format(beg, end))
        chunk.sort(key=key)
        write_items(chunk, get_fn(chunk_i))
        chunk_i += 1

    nb_chunks = chunk_i  # will be modified
    total_nb_chunks = chunk_i  # archived for later

    # PHASE 2 / Merge sorted chunks.
    #
    # There may be a large number of chunks and we could easily
    # exceed the max. allowed number of opened files (ulimit -n).
    #
    # So here we merge sorted chunks iteratively,
    # i.e. we merge at most `file_chunk_size`
    # chunks in parallel, yielding a new set of chunks.
    #
    # As soon as only one chunk remains, we are done.
    #
    while nb_chunks > 1:

        sys.stdout.write(
            "Number of chunks: %i\n" % nb_chunks
        )
        chunk_ranges = [
            (beg_chunk, min(nb_chunks, beg_chunk + file_chunk_size))
            for beg_chunk in range(0, nb_chunks, file_chunk_size)
        ]
        for chunk_i, (beg_chunk, end_chunk) in enumerate(chunk_ranges):
            sys.stdout.write(
                "Merging chunks %i - %i\n" % (beg_chunk, end_chunk)
            )
            chunks = [
                read_items(get_fn(i))
                for i in range(beg_chunk, end_chunk)
            ]
            items = collate(*chunks, key=key)
            safe_fn = get_fn("safe-%i" % chunk_i)
            then_fn = get_fn(chunk_i)
            write_items(items, safe_fn)
            rename(safe_fn, then_fn)

        nb_chunks = len(chunk_ranges)

    # Assign the final file with desired name
    rename(get_fn(0), out_fn)

    # Remove temporary files
    for i in range(total_nb_chunks):
        fn = get_fn(i)
        if exists(fn):
            remove(fn)


def bounded_count(keys, filename, cache_size=int(1e7)):
    """
    Counts `keys` in a memory friendly way, always keeping
    at most `cache_size` items in the frequency dictionary.

    `keys` must be a stream of strings.

    The output will be saved to `filename` tab-separated vertical,
    sorted by keys, freqency after the last tab character.
    """

    def iter_chunks():
        counter = defaultdict(lambda: 0)
        for k in keys:
            counter[k] += 1
            if len(counter) > cache_size:
                yield sorted(counter.items(), key=lambda item: item[0])
                counter = defaultdict(lambda: 0)
        if len(counter) > 0:
            yield sorted(counter.items(), key=lambda item: item[0])

    def iter_from_disk(fn):
        with our_open(fn) as f:
            for line in f:
                _key, _freq = line.strip().rsplit("\t", 1)
                yield _key, int(_freq)

    def merge_sum(stream1, stream2):
        collated = collate(stream1, stream2, key=lambda x: x[0])
        cur_key = None
        cur_freq = None
        for _key, _freq in collated:
            if cur_key == _key:
                cur_freq += 1
            else:
                if cur_key is not None:
                    yield cur_key, cur_freq
                cur_key = _key
                cur_freq = _freq
        if cur_key is not None:
            yield cur_key, cur_freq

    fn_temp1 = filename + ".temp1.gz"
    fn_temp2 = filename + ".temp2.gz"

    # Make file `fn_temp1` exist and empty.
    with open(fn_temp1, "w"):
        pass

    for chunk in iter_chunks():
        rename(fn_temp1, fn_temp2)
        merged = merge_sum(iter_from_disk(fn_temp2), chunk)
        with our_open(fn_temp1, "w") as new_file:
            for key, freq in merged:
                new_file.write("%s\t%s\n" % (key, freq))
        del chunk

    rename(fn_temp1, filename)

    for fn_temp in [fn_temp1, fn_temp2]:
        if exists(fn_temp):
            remove(fn_temp)
