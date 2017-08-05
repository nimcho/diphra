#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2017 Lukas Banic <lukas.banic@protonmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# This file incorporates work covered by the following copyright:
#
#   Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
#   Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.math cimport log
from libc.string cimport memset

# scipy <= 0.15
try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

randint = np.random.randint
REAL = np.float32

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

# for when fblas.sdot returns a double
cdef REAL_t our_dot_double(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>dsdot(N, X, incX, Y, incY)

# for when fblas.sdot returns a float
cdef REAL_t our_dot_float(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>sdot(N, X, incX, Y, incY)

# for when no blas available
cdef REAL_t our_dot_noblas(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    # not a true full dot()-implementation: just enough for our cases
    cdef int i
    cdef REAL_t a
    a = <REAL_t>0.0
    for i from 0 <= i < N[0] by 1:
        a += X[i] * Y[i]
    return a

# for when no blas available
cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil:
    cdef int i
    for i from 0 <= i < N[0] by 1:
        Y[i * (incY[0])] = (alpha[0]) * X[i * (incX[0])] + Y[i * (incY[0])]


# to support random draws from negative-sampling cum_table
cdef inline unsigned long long bisect_left(np.uint32_t *a, unsigned long long x, unsigned long long lo, unsigned long long hi) nogil:
    cdef unsigned long long mid
    while hi > lo:
        mid = (lo + hi) >> 1
        if a[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo

# this quick & dirty RNG apparently matches Java's (non-Secure)Random
# note this function side-effects next_random to set up the next number
cdef inline unsigned long long random_int32(unsigned long long *next_random) nogil:
    cdef unsigned long long this_random = next_random[0] >> 16
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return this_random


cdef inline unsigned long long train_pair(
        REAL_t *target_vecs,
        REAL_t *context_vecs,
        REAL_t *work,
        np.uint32_t *cum_table,
        REAL_t lr,
        unsigned long long negative,
        const int vector_size,
        unsigned long long cum_table_len,
        unsigned long long modulo,
        unsigned long long i,
        unsigned long long j,
        unsigned long long next_random
) nogil:

    cdef unsigned long long pi, pj, d
    cdef REAL_t f, g, label

    memset(work, 0, vector_size * cython.sizeof(REAL_t))
    pi = vector_size * i

    for d in range(negative + 1):

        if d == 0:
            pj = vector_size * j
            label = ONEF
        else:
            pj = vector_size * bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            label = <REAL_t>0.0
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo

        f = our_dot(&vector_size, &target_vecs[pi], &ONE, &context_vecs[pj], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * lr
        our_saxpy(&vector_size, &g, &context_vecs[pj], &ONE, work, &ONE)
        our_saxpy(&vector_size, &g, &target_vecs[pi], &ONE, &context_vecs[pj], &ONE)

    our_saxpy(&vector_size, &ONEF, work, &ONE, &target_vecs[pi], &ONE)
    return next_random


def train_batch(
        _target_vecs,
        _context_vecs,
        _cum_table,
        _thresholds,
        _data,
        _lr,
        _window,
        _negative,
        _work
):

    cdef REAL_t *target_vecs = <REAL_t *>(np.PyArray_DATA(_target_vecs))
    cdef REAL_t *context_vecs = <REAL_t *>(np.PyArray_DATA(_context_vecs))
    cdef REAL_t *work = <REAL_t *>np.PyArray_DATA(_work)

    cdef np.uint32_t *cum_table = <np.uint32_t *>(np.PyArray_DATA(_cum_table))
    cdef np.uint32_t *thresholds = <np.uint32_t *>(np.PyArray_DATA(_thresholds))
    cdef np.uint32_t *data = <np.uint32_t *>(np.PyArray_DATA(_data))

    cdef REAL_t lr = _lr
    cdef unsigned long long window = _window
    cdef unsigned long long negative = _negative

    # ---

    cdef unsigned long long cum_table_len = len(_cum_table)
    cdef unsigned long long vector_size = _target_vecs.shape[1]
    cdef unsigned long long nb_pocs = _data.shape[0]
    cdef unsigned long long nb_cols = _data.shape[1]
    cdef unsigned long long nb_poss = nb_cols - 3
    cdef unsigned long long mask_value = np.iinfo(np.uint32).max
    cdef unsigned long long modulo = 281474976710655ULL
    cdef unsigned long long next_random = (2**24) * randint(0, 2**24) + randint(0, 2**24)
    cdef unsigned long long counter = 0

    cdef REAL_t x, y, z
    cdef unsigned long long i, j, k, ii, jj, pi, pj
    cdef unsigned long long a, pa, a_nb_poss
    cdef unsigned long long b, pb, nb_downsampled
    cdef long long d

    with nogil:

        i = 0

        # Down-sample freq. phrases
        # and choose a random context window.
        for j in range(nb_pocs):
            pj = j * nb_cols
            if thresholds[data[pj + 1]] < random_int32(&next_random):
                pi = i * nb_cols
                if i != j:
                    for k in range(nb_cols):
                        data[pi + k] = data[pj + k]
                data[pi] = random_int32(&next_random) % window + 1  # context window
                i += 1

        nb_downsampled = nb_pocs - i
        nb_pocs = i  # after down-sampling

        # ---

        for a in range(nb_pocs):

            pa = a * nb_cols  # first index in data array

            # determine number of positions for poc `a`
            # (silently assuming data correctness,
            # i.e. at least one position for each poc)
            a_nb_poss = nb_poss
            for i in range(nb_poss):
                if data[pa + 3 + i] == mask_value:
                    a_nb_poss = i
                    break

            for b in range(a + 1, nb_pocs):

                pb = b * nb_cols

                if data[pa + 2] != data[pb + 2]:
                    break  # crossing sentence boundary

                # Distance -- number of tokens between a and b.
                d = data[pb + 3] - data[pa + 2 + a_nb_poss]

                if d >= max(<long long>data[pa], <long long>data[pb]):
                    break  # beyond context window radii

                # Check overlaps
                if d == 0:
                    continue  # overlapping on a_p[-1] position
                elif d < 0:
                    if a_nb_poss == 1:
                        if data[pa + 3] == data[pb + 3]:
                            continue  # overlapping on a_p[0] position
                    else:
                        # check some more obscure overlaps
                        i, j = 3, 3
                        k = 0  # if overlap ===> set to 1
                        while True:
                            pi = data[pa + i]
                            pj = data[pb + j]
                            if pi == pj:
                                k = 1  # message that there is an overlap
                                break
                            elif pi < pj:
                                i += 1
                                if i == nb_cols:
                                    break  # no overlap
                                elif data[pa + i] == mask_value:
                                    break  # no overlap
                            else:
                                j += 1
                                if j == nb_cols:
                                    break  # no overlap
                                elif data[pb + j] == mask_value:
                                    break  # no overlap
                        if k == 1:
                            continue  # some obscure overlap

                if d <= data[pa]:
                    next_random = train_pair(
                        target_vecs, context_vecs, work, cum_table,
                        lr, negative, vector_size, cum_table_len, modulo,
                        data[pa + 1], data[pb + 1], next_random
                    )
                    counter += 1
                if d <= data[pb]:
                    next_random = train_pair(
                        target_vecs, context_vecs, work, cum_table,
                        lr, negative, vector_size, cum_table_len, modulo,
                        data[pb + 1], data[pa + 1], next_random
                    )
                    counter += 1

    # ---

    return counter, nb_downsampled


def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.  Also calculate log(sigmoid(x)) into LOG_TABLE.

    """
    global our_dot
    global our_saxpy

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
        LOG_TABLE[i] = <REAL_t>log( EXP_TABLE[i] )

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        our_dot = our_dot_double
        our_saxpy = saxpy
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        our_dot = our_dot_float
        our_saxpy = saxpy
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        our_dot = our_dot_noblas
        our_saxpy = our_saxpy_noblas
        return 2

FAST_VERSION = init()  # initialize the module
