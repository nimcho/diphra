#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Lukas Banic <lukas.banic@protonmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
"""
POcs2Vec: Distributed Representations of Phrases & POcs Data Structure
======================================================================

A modified implementation of Skip-Gram with Negative Sampling method,
first introduced by Mikolov et al. [1]_ [2]_ as a part of Word2Vec system.

A great inspiration came from Levy and Goldberg's work [3]_,
who pointed out that the SG+NS can be reinterpreted and trained
on an arbitrary stream of (target, context) pairs -- an idea
that was implemented as Word2Vecf [4]_.

POcs2Vec consumes pocs (phrase occurrences), which can be nested,
overlapping and discontinuous.

On the technical side, POcs2Vec is built on ideas
from the efficient implementation of Word2Vec
in Gensim package written by Radim Rehurek [5]_, [6]_.

## References

.. [1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean.
       Efficient Estimation of Word Representations in Vector Space.
       In Proceedings of Workshop at ICLR, 2013.
.. [2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean.
       Distributed Representations of Words and Phrases and their Compositionality.
       In Proceedings of NIPS, 2013.
.. [3] Omer Levy and Yoav Goldberg. 2014a. Dependency-based,word embeddings.
       In ACL, pages 302–308
.. [4] Yoav Goldberg.  2015.  word2vecf.
       In BitBucket repository, bitbucket.org/yoavgo/word2vecf
.. [5] RaRe Technologies. 2016. Gensim.
       In GitHub repository, github.com/RaRe-Technologies/gensim
.. [6] Optimizing word2vec in gensim,
       In Radim Rehurek's blog, radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/
"""
import pickle
import threading
import sys
import numpy as np
import os

from queue import Queue
from timeit import default_timer
from scipy.special import expit
from numpy.random import rand, randint
from numpy import array, zeros, uint32, full, iinfo, intersect1d, \
    float32 as real, empty

from ..pocs import HDF5POcs, pocs2vocab

BATCH_SIZE = 10000  # TODO refactor so that we can get rid of this fancy param
BATCH_TYPE = uint32
MASK_VALUE = iinfo(BATCH_TYPE).max
QUEUE_FACTOR = 2
FINISH = None
RECOMPILE = True  # for cython backend debugging

try:
    if RECOMPILE:
        import pyximport
        models_dir = os.path.dirname(__file__) or os.getcwd()
        pyximport.install(setup_args={"include_dirs": [models_dir, np.get_include()]})
    from pocs2vec_inner import train_batch as train_batch_fast
except ImportError:
    train_batch_fast = None


def iter_sentences(pocs, phrase2i):
    """
    Groups a stream of sorted pocs by sentences
    and replaces each phrase's text with its ID.

    If a phrase's ID is not found in the given
    dictionary, the poc is skipped.
    """
    sentence = list()
    current_i = None

    for phrase, sentence_i, positions in pocs:

        if current_i != sentence_i:
            current_i = sentence_i
            if len(sentence) > 0:
                yield sentence
                sentence = list()

        phrase_i = phrase2i.get(phrase, None)
        if phrase_i is not None:
            sentence.append((phrase_i, sentence_i, positions))

    if len(sentence) > 0:
        yield sentence


def bake_batch(pocs, max_len):
    """
    Called from `iter_batches`.

    # Arguments
        pocs: A list of pocs shorter than BATCH_SIZE,
            no text allowed -- only phrase IDs.
        max_len: Maximum phrase length.

    # Returns:
        batch_data: POcs packed in a numpy array,
            first column is reserved for context window,
            second column is phrase ID,
            third column is sentence ID,
            all other columns are (masked) positions.
    """
    assert len(pocs) <= BATCH_SIZE

    batch_data = full(
        shape=(len(pocs), 3 + max_len),
        fill_value=MASK_VALUE,
        dtype=BATCH_TYPE
    )
    for row, (phrase_i, sentence_i, positions) in enumerate(pocs):
        # batch_data[row, 0] reserved for random context window
        batch_data[row, 1] = phrase_i
        batch_data[row, 2] = sentence_i
        batch_data[row, 3:3 + len(positions)] = positions

    return batch_data


def iter_batches(pocs, nb_pocs, nb_epochs, phrase2i, max_len):
    """
    Packs a stream of sorted pocs into batches.
    """
    # To calculate progress:
    nb_done_pocs = 0
    total_pocs = nb_epochs * nb_pocs

    batch_pocs = list()

    for epoch in range(nb_epochs):
        for sentence in iter_sentences(pocs, phrase2i):

            if len(batch_pocs) + len(sentence) <= BATCH_SIZE:
                # Enough place for this sentence.
                batch_pocs += sentence
                continue

            if len(sentence) > BATCH_SIZE:
                # Too long sentence -- skip it!
                nb_done_pocs += len(sentence)
                continue

            if len(batch_pocs) > 0:
                # Yield the current batch.
                nb_done_pocs += len(batch_pocs)
                progress = float(nb_done_pocs) / total_pocs
                yield bake_batch(batch_pocs, max_len), progress
                # Initialize a new batch with this sentence.
                batch_pocs = sentence

    if len(batch_pocs) > 0:
        # Yield the last batch.
        nb_done_pocs += len(batch_pocs)
        progress = float(nb_done_pocs) / total_pocs
        yield bake_batch(batch_pocs, max_len), progress


def iter_batches_hdf5(pocs, nb_pocs, nb_epochs, max_len):

    # To calculate progress:
    nb_done_pocs = 0
    total_pocs = nb_epochs * nb_pocs

    nb_boundaries = int(pocs.file["boundaries"].shape[0])
    offset = 50  # sentences

    for epoch in range(nb_epochs):
        for b in range(0, nb_boundaries, offset):
            beg = int(pocs.file["boundaries"][b])
            end = int(pocs.file["boundaries"][min(nb_boundaries - 1, b + offset)])
            batch_data = empty(
                shape=(end - beg, 3 + max_len),
                dtype=BATCH_TYPE
            )
            batch_data[:, 1:] = pocs.file["pocs"][beg:end]
            nb_done_pocs += end - beg
            progress = float(nb_done_pocs) / total_pocs
            yield batch_data, progress


def train_pair(target_vecs, context_vecs, cum_table, negative, i, j, lr):
    """
    Does a single SGD step towards optimizing SG+NS objective.
    Called from `train_batch_numpy` function.
    """
    s = 1 - expit(context_vecs[j].dot(target_vecs[i]))
    h = lr * s * context_vecs[j]
    context_vecs[j] += lr * s * target_vecs[i]

    for _ in range(negative):
        # Draw a negative sample:
        k = cum_table.searchsorted(randint(cum_table[-1]))

        s = 0 - expit(context_vecs[k].dot(target_vecs[i]))
        h = lr * s * context_vecs[k]
        context_vecs[k] += lr * s * target_vecs[i]

    target_vecs[i] += h


def train_batch_numpy(target_vecs, context_vecs, cum_table, thresholds,
                      data, lr, window, negative, work):
    """
    Takes a batch of pocs, randomly down-samples frequent phrases,
    randomly chooses context window for each poc
    and then generates target-context pairs, whose are
    further passed to `train_pair` function.

    The function serves a demonstrative purpose
    and should not be used as it is too slow.
    Use cythonized train_batch() from pocs2vec_inner.pyx!
    """
    downsampled = data[
        array([thresholds[i] for i in data[:, 1]]) <
        rand(len(data)) * 2 ** 32
    ]
    # Random context window for each poc:
    downsampled[:, 0] = randint(1, 1 + window, len(downsampled))

    for a in range(len(downsampled)):

        a_w = downsampled[a, 0]  # context window
        a_i = downsampled[a, 1]  # phrase ID
        a_s = downsampled[a, 2]  # sentence ID
        a_p = downsampled[a, 3:]  # positions
        a_p = a_p[a_p != MASK_VALUE]

        for b in range(a + 1, len(downsampled)):

            b_w = downsampled[b, 0]  # context window
            b_i = downsampled[b, 1]  # phrase ID
            b_s = downsampled[b, 2]  # sentence ID
            b_p = downsampled[b, 3:]  # positions

            if a_s != b_s:
                break  # crossing sentence boundary

            # Distance -- number of tokens between a and b.
            d = real(b_p[0]) - a_p[-1]

            if d >= max(a_w, b_w):
                break  # beyond context window radii

            # The following mess tests whether pocs a, b overlap
            # and could be reduced to ```len(intersect1d(a_p, b_p)) > 0```,
            # but this is a critical loop that needs to be
            # optimized for performance.
            if d == 0:
                continue  # overlapping on a_p[-1] position
            elif d < 0:
                if len(a_p) == 1:
                    if a_p[0] == b_p[0]:
                        continue  # overlapping on a_p[0] position
                elif len(intersect1d(a_p, b_p)) > 0:
                    continue  # some other overlapping

            # If we got here, it's time to train!
            if d < a_w:
                train_pair(
                    target_vecs, context_vecs, cum_table,
                    negative, a_i, b_i, lr
                )
            if d < b_w:
                train_pair(
                    target_vecs, context_vecs, cum_table,
                    negative, b_i, a_i, lr
                )

    nb_tc = len(data) - len(downsampled)
    nb_ds = len(downsampled)
    return nb_tc, nb_ds


def get_cum_table(freqs, domain=2 ** 31 - 1):

    cum_table = zeros(len(freqs), dtype=uint32)

    total = np.sum(freqs)
    cumulative = 0.0
    for i in range(len(freqs)):
        cumulative += freqs[i]
        cum_table[i] = round(cumulative / total * domain)

    assert cum_table[-1] == domain
    return cum_table


if train_batch_fast is None:
    sys.stderr.write("Unable to use cythonized version => ~70x slower.")
    train_batch = train_batch_numpy
else:
    train_batch = train_batch_fast


def pocs2vec(pocs, output_name,
             dim=256,
             sample=int(2e3),
             window=5,
             negative=7,
             nb_epochs=5,
             max_lr=0.05,
             min_lr=0.00001,
             nb_workers=4,
             report_interval=1.0):

    if isinstance(pocs, HDF5POcs):
        freqs = pocs.file["freqs"].value
        max_len = int(pocs.file["pocs"].shape[1] - 2)
        nb_pocs = freqs.sum()
        batches = iter_batches_hdf5(pocs, nb_pocs, nb_epochs, max_len)
        phrase2i = None
    else:
        phrase2i, freqs, max_len = pocs2vocab(pocs)
        nb_pocs = np.sum(freqs)
        batches = iter_batches(pocs, nb_pocs, nb_epochs, phrase2i, max_len)

    vocab_size = len(freqs)
    total_pocs = nb_epochs * nb_pocs

    # Phrases with frequency higher than `sample`
    # will be randomly down-sampled with a probability:
    #
    #   1 - sqrt(sample / frequency)
    #
    thresholds = 1.0 - np.sqrt(float(sample) / freqs)
    thresholds = (thresholds.clip(min=0.0) * 2 ** 32).astype(uint32)

    # Negative samples are drawn from unigram distribution
    # (typically raised to 0.75) by choosing a random number
    # and finding its insertion point in a cumulative
    # distribution table.
    cum_table = get_cum_table(freqs ** 0.75)

    # Vector initialization
    target_vecs = ((rand(vocab_size, dim) - 0.5) / dim).astype(real)
    context_vecs = zeros((vocab_size, dim), dtype=real)

    rep_data = dict(progress=0.0, counter=0, nb_ds=0)

    def worker_loop():
        # per-thread private work memory
        work = zeros(dim, dtype=real)
        while True:
            job = job_queue.get()
            if job is FINISH:  # no more jobs => quit this worker
                progress_queue.put(True)
                break
            _nb_tc, _nb_ds = train_batch(
                target_vecs, context_vecs, cum_table, thresholds,
                job["data"], job["lr"], window, negative, work
            )
            progress_queue.put((_nb_tc, _nb_ds))

    def job_producer():
        learning_rate = max_lr
        for batch_data, progress in batches:
            job_queue.put(
                dict(
                    data=batch_data,
                    lr=learning_rate
                )
            )
            learning_rate = max(
                min_lr,
                max_lr - (max_lr - min_lr) * progress
            )
            rep_data["progress"] = progress
        # give the workers heads up that they can finish -- no more work!
        for _ in range(nb_workers):
            job_queue.put(FINISH)

    job_queue = Queue(maxsize=QUEUE_FACTOR * nb_workers)
    progress_queue = Queue(maxsize=(QUEUE_FACTOR + 1) * nb_workers)

    workers = [threading.Thread(target=worker_loop) for _ in range(nb_workers)]
    unfinished_worker_count = len(workers)
    workers.append(threading.Thread(target=job_producer))

    for thread in workers:
        thread.daemon = True  # make interrupting the process with ctrl+c easier
        thread.start()

    start = default_timer()
    next_report = report_interval
    report = ""

    while unfinished_worker_count > 0:

        finished = progress_queue.get()  # blocks if workers too slow

        if finished is True:  # a thread reporting that it finished
            unfinished_worker_count -= 1
            if report != "":
                sys.stdout.write("\n")
                report = ""
            sys.stdout.write(
                "worker thread finished; awaiting finish of %i more threads\n"
                % unfinished_worker_count
            )
            continue

        nb_tc, nb_ds = finished
        rep_data["counter"] += nb_tc
        rep_data["nb_ds"] += nb_ds

        # log progress once every report_delay seconds
        elapsed = default_timer() - start
        if elapsed >= next_report:
            sys.stdout.write("\b" * len(report))
            sys.stdout.write("\r")
            report = (
                "[[ PROGRESS: {:.2f}% | {:,.0f} pocs/s | {:,.0f} tc/s | "
                "{:,.0f} downsampled/s ]]".format(
                    100.0 * rep_data["progress"],
                    (rep_data["progress"] * total_pocs) / elapsed,
                    float(rep_data["counter"]) / elapsed,
                    float(rep_data["nb_ds"]) / elapsed
                )
            )
            sys.stdout.write(report)
            sys.stdout.flush()
            next_report = elapsed + report_interval

    # ... and persist the trained model to disk:

    if phrase2i is not None:
        with open("%s.phrase2i.pickle" % output_name, "w") as phrase2i_file:
            pickle.dump(phrase2i, phrase2i_file)

    np.save("%s.freqs.npy" % output_name, freqs)
    np.save("%s.target.npy" % output_name, target_vecs)
    np.save("%s.context.npy" % output_name, context_vecs)
