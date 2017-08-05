#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Lukas Banic <lukas.banic@protonmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import pickle
import numpy as np

from os.path import join
from collections import defaultdict
from copy import deepcopy


def get_head(tree):
    """
    Returns the first item from pre-order tree traversal.
    """
    if type(tree) is tuple:
        return get_head(tree[0]) if len(tree) > 0 else None
    else:
        return tree


class PType(object):
    """
    Phrase type (used by Lexicon).
    """

    def __init__(self, i, tree):
        self.i = i
        self.tree = tree

    def __getattr__(self, item):
        if item == "head":
            return get_head(self.tree)

    def __eq__(self, other):
        return self.i == other.i


class LexItem(object):

    def __init__(self, i, ptype, lex_ids, text, freq):

        self.i = int(i)
        self.text = text
        self.ptype = ptype
        self.lex_ids = lex_ids
        self.freq = freq
        self.head = None  # must be calculated explicitly from Lexicon

    def __eq__(self, other):

        return self.i == other.i


class Lexicon(object):

    @staticmethod
    def load(directory):

        ptypes_fn = join(directory, "phrase_types.pickle")
        lexicon_fn = join(directory, "lexicon.vert")

        with open(ptypes_fn) as pt_file:
            ptypes = [
                PType(i, pt) for i, pt
                in enumerate(pickle.load(pt_file))
            ]

        phrases = []
        with open(lexicon_fn) as lexicon_file:
            for i, line in enumerate(lexicon_file):
                parts = line.strip().split("\t")
                ptype = ptypes[int(parts[1])]
                lex_ids = tuple(map(int, parts[2].split(",")))
                text = parts[3]
                freq = int(parts[4])
                phrases.append(LexItem(i, ptype, lex_ids, text, freq))

        return Lexicon(ptypes, phrases)

    def __init__(self, ptypes, phrases):

        self.ptypes = ptypes
        self.phrases = phrases

        self.tree2i = {
            ptype.tree: ptype.i for ptype in self.ptypes
        }
        self.text2phrase = defaultdict(lambda: list())
        for phrase in self.phrases:
            self.text2phrase[phrase.text].append(phrase)

        self.vecs = None
        self.head_index = None

    def __getitem__(self, item):

        if type(item) is int:
            return self.phrases[item]
        elif type(item) in (str, unicode):
            return self.text2phrase[item]

    def __len__(self):
        return len(self.phrases)

    def set_heads(self):

        # Currently, a head is strictly a single word expression.
        # This could be changed.

        mapping = {
            (phrase.ptype.i, phrase.lex_ids): phrase
            for phrase in self.phrases
        }
        self.head_index = defaultdict(lambda: list())
        for phrase in self.phrases:

            head_ptype_tree = (phrase.ptype.head, )
            head_ptype_i = self.tree2i.get(head_ptype_tree, None)
            if head_ptype_i is None:
                continue  # no registered phrases for such ptype

            phrase.head = mapping.get(
                (head_ptype_i, (phrase.lex_ids[0], )),
                None
            )
            if phrase.head is not None:
                self.head_index[phrase.head.i].append(phrase)

    def load_vecs(self, fn):
        self.vecs = np.load(fn)
        self.vecs /= np.linalg.norm(self.vecs, axis=1, ord=2, keepdims=True)
        assert len(self.vecs) == len(self.phrases)

    def similarity(self, a, b):
        return self.vecs[a.i].dot(self.vecs[b.i])

    def most_similar(self, positive, negative=None, topn=10):

        if isinstance(positive, LexItem):
            positive = [positive]

        assert len(positive) > 0

        if isinstance(negative, LexItem):
            negative = [negative]
        elif negative is None:
            negative = []

        vec = deepcopy(self.vecs[positive[0].i])
        for x in positive[1:]:
            vec += self.vecs[x.i]
        for x in negative:
            vec -= self.vecs[x.i]

        vec /= np.linalg.norm(vec)

        excl_set = set([x.i for x in positive + negative])
        ids = np.flip(self.vecs.dot(vec).argsort()[-(topn + len(excl_set) + 1):], axis=0)
        return [self[int(i)] for i in ids if i not in excl_set][:topn]
