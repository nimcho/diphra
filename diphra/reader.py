#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Lukas Banic <lukas.banic@protonmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from os.path import splitext
from collections import defaultdict
from gzip import GzipFile

file_openers = defaultdict(lambda: open)

file_openers.update({
    ".gz": lambda name, mode, buffering: GzipFile(fileobj=open(name, mode, buffering)),
})


def our_open(name, mode="r", buffering=-1):
    """
    Auto-decompressing open function.
    """
    return file_openers[splitext(name)[1]](name, mode, buffering)
