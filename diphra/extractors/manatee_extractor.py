#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Lukas Banic <lukas.banic@protonmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
"""
Memory-Friendly Phrase Extraction with Shallow Parsing
======================================================

This code uses Manatee corpus manager (nlp.fi.muni.cz/trac/noske) [1]_
and subsequently Corpus Query Language (CQL) formalism
(sketchengine.co.uk/documentation/corpus-querying).

## References

.. [1] Rychlý, Pavel. Manatee/Bonito - A Modular Corpus Manager.
       In 1st Workshop on Recent Advances in Slavonic Natural Language Processing.
       Brno : Masaryk University, 2007. p. 65-70. ISBN 978-80-210-4471-5.
"""

import json
import pickle
import sys

import manatee

from string import Formatter
from collections import defaultdict, namedtuple
from os import path, makedirs, remove
from os.path import exists
from glob import glob
from itertools import chain, islice, combinations
from more_itertools import peekable

from sortedcontainers import SortedListWithKey
from more_itertools import collapse

from ..utils import bounded_count, SelfClosingFile
from ..reader import our_open


class MissingLabels(Exception):
    """
    Raised when a CQL query does not contain labels
    that are required by subsequent structures.
    """
    pass


class NoOccurrenceFound(Exception):
    """
    Raised when a CQL query yields no results in cases
    where at least a single result is indispensable.
    """
    pass


class WorkspaceReserved(Exception):
    """
    Raised when preparing a workspace in a reserved directory
    (i.e. a configuration file already exists).
    """
    pass


class WorkspaceNotFound(Exception):
    """
    Raised when trying to open a non-existing workspace
    (i.e. in a directory without a configuration file).
    """
    pass


class CollidingPhraseType(Exception):
    """
    Raised when trying to start a phrase extraction that
    has already happend and skipping/overwriting was not
    explicitly set up.
    """
    pass


def analyse_template(template):
    """
    Takes a template for a phrase pattern
    and returns a dictionary {label: set_of_attrs}

    # Examples
        >>> ta = analyse_template("{1.lemma}/{1.tag} {2.word}")
        >>> ta == {1: {"lemma", "tag"}, 2: {"word"}}
        True
        >>> ta = analyse_template("1.lemma/1.tag 2.word")
        Traceback (most recent call last):
         ...
        ValueError: A template must contain at least one variable...
    """
    field_names = [
        field_name
        for _, field_name, __, ___
        in Formatter().parse(template)
        if field_name is not None
    ]
    if len(field_names) == 0:
        raise ValueError(
            'A template must contain at least one variable! '
            'Your template: %s' % template
        )
    label2attrs = defaultdict(lambda: set())
    for field_name in field_names:
        label, attr = field_name.split(".")
        label2attrs[int(label)].add(attr)

    return dict(label2attrs)


def iter_label2pos(corpus, query):
    """
    Low-level function to evaluate a labeled CQL query
    on a given corpus (called from `match_cql` function).

    # Arguments
        corpus: `manatee.Corpus` object
        query: labeled CQL query

    # Yields
        {label: position} dictionaries
    """
    results = corpus.eval_query(query)
    while not results.end():
        beg = results.peek_beg()  # query's first token's position
        colls = manatee.IntVector()
        results.collocs(colls)
        yield {
            colls[i]: colls[i + 1] + beg
            for i in range(0, len(colls), 2)
        }
        results.next()


def match_cql(corpus, struct, lex_attr, query, template, cql_attr=None):
    """
    Finds occurrences of a language pattern
    described by a given CQL query in a given corpus.

    # Arguments
        corpus: A corpus to find occurrences in.
            You may provide a corpus name
            or directly pass a `manatee.Corpus` object.
        struct: A structure to work with (typically sentence).
            You may provide a structure name
            or directly pass a `manatee.Structure` object.
        lex_attr: Lexical attribute ... (typically lemma)
            You may provide an attribute name
            or directly pass a `manatee.PosAttr` object.
        query: CQL query with at least one label,
            i.e. '1:[tag="N.*"]' is a valid query,
            while '[tag="N.*"]' is NOT.
        template: A string with placeholders linked
            to labels mentioned in the query
            and to attributes available in the corpus,
            e.g.: "{1.lemma}/{1.tag} {2.word}"
        cql_attr: Default CQL attribute.
            It can be used to shorten queries,
            e.g. you can use "N.*" instead of [tag="N.*"]
            if the default attribute is set to "tag".

    # Yields
        (struct_id, label2pos, label2lex_attr, text) tuples,
        each one representing a single phrase occurrence.

        `struct_id` is a structure number.
        `label2pos` maps each label to a position within that structure.
        `label2lex_attr` maps each label to a lexical attribute index.
        `text` is a filled in template.

    # Example
        ```python
        iter_ocs(
            corpus='bnc', struct='s', lex_attr='lemma',
            cql_attr='tag', query='2:"A.*" 1."N.*"',
            template='{2.lemma} {1.lemma}'
        )
        ```
    """
    if type(corpus) == str:
        corpus = manatee.Corpus(corpus)
    if type(struct) == str:
        struct = corpus.get_struct(struct)
    if type(lex_attr) == str:
        lex_attr = corpus.get_attr(lex_attr)

    if cql_attr is not None:
        corpus.set_default_attr(cql_attr)

    template = template.strip()
    label2attrs = analyse_template(template)

    # We will use manatee.PosAttr objects
    # for each attribute mentioned in `template`
    manatee_attrs = {
        attr: corpus.get_attr(attr)
        for attr in set(chain.from_iterable(label2attrs.values()))
    }

    # To enable templates like "{1.lemma}/{1.tag} {2.word}"
    namedtuples = {
        label: namedtuple("NTLabel%i" % label, tuple(attrs))
        for label, attrs in label2attrs.items()
    }

    for label2pos in iter_label2pos(corpus, query):

        label2lex_id = {
            label: lex_attr.pos2id(pos)
            for label, pos in label2pos.items()
        }

        # List that will be used to fill in the template
        label2nt = [None for _ in range(max(label2attrs) + 1)]

        for label, attrs in label2attrs.items():
            try:
                pos = label2pos[label]
            except KeyError:
                raise MissingLabels(
                    "Unable to fill in the template.",
                    dict(query=query, template=template, label=label)
                )
            else:
                label2nt[label] = namedtuples[label](
                    **{attr: manatee_attrs[attr].pos2str(pos)
                       for attr in attrs}
                )

        text = template.format(*label2nt)
        struct_id = struct.num_at_pos(min(label2pos.values()))

        # Positions relative to the structure
        struct_beg = struct.beg(struct_id)
        label2pos = {
            label: pos - struct_beg
            for label, pos in label2pos.items()
        }

        yield struct_id, label2pos, label2lex_id, text


class RawPOc(object):
    """
    Used by ManateeExtractor class.

    Positions and lexical IDs are sorted so as to correspond
    to nodes of a phrase type tree in pre-order traversal.
    """
    @staticmethod
    def from_line(line):
        parts = line.split("\t")
        return RawPOc(
            lex_ids=tuple(map(int, parts[0].split(","))),
            text=parts[1],
            struct_id=int(parts[2]),
            positions=tuple(map(int, parts[3].split(",")))
        )

    def __init__(self, lex_ids, text, struct_id, positions):
        self.lex_ids = lex_ids
        self.text = text
        self.struct_id = struct_id
        self.positions = positions

    def __str__(self):
        return "\t".join([
            ",".join(map(str, self.lex_ids)),
            self.text,
            str(self.struct_id),
            ",".join(map(str, self.positions)),
        ])


class ManateeExtractor(object):

    FN_CONFIG = "config.json"
    DN_RAW_POCS = "raw_pocs"
    DN_RAW_VOCABS = "raw_vocabs"

    @staticmethod
    def make(directory, corpus, struct, lex_attr, cql_attr=None):
        """
        Prepares a workspace for phrase extraction in a given directory.

        # Arguments
            directory: All intermediate data generated during
                phrase extraction will be saved to this directory.
            corpus: Name of a compiled corpus.
            struct: Name of a structure (typically sentence).
            lex_attr: Lexical attribute (typically lemma).
            cql_attr: Default CQL attribute.
                It can be used to shorten queries,
                e.g. you can use "N.*" instead of [tag="N.*"]
                if the default attribute is set to "tag".

        # Raises
            WorkspaceReserved: If there already exists
                a configuration file in the given directory.
        """
        fn_config = path.join(directory, ManateeExtractor.FN_CONFIG)
        if path.exists(fn_config):
            raise WorkspaceReserved(
                "Configuration file `%s` already exists."
                % fn_config
            )

        # Test whether the provided arguments
        # can be actually used.
        c = manatee.Corpus(corpus)
        c.get_struct(struct)
        c.get_attr(lex_attr)
        if cql_attr is not None:
            c.get_attr(cql_attr)

        # Create CQLExtractor's base directory
        if not path.exists(directory):
            makedirs(directory)

        # Create additional subdirectories
        sub_directories = [
            path.join(directory, sd)
            for sd in [ManateeExtractor.DN_RAW_POCS,
                       ManateeExtractor.DN_RAW_VOCABS]
        ]
        for sub_directory in sub_directories:
            if not path.exists(sub_directory):
                makedirs(sub_directory)

        # Create configuration file containing
        # basic info about this CQLExtractor
        config = dict(
            corpus=corpus,
            struct=struct,
            lex_attr=lex_attr,
            cql_attr=cql_attr,
        )
        with our_open(fn_config, "w") as config_file:
            json.dump(config, config_file)

    def __init__(self, directory, cache_size=int(1e7)):
        """
        Opens a phrase extractor located in a given directory.

        `cache_size` determines the maximum number of vocabulary items
        that can be held in memory.
        """
        self.directory = directory
        self.cache_size = cache_size

        # Filenames of outputs of build_lexicon(...)
        self.fn_phrase_types = path.join(self.directory, "phrase_types.pickle")
        self.fn_lexicon = path.join(self.directory, "lexicon.vert")
        self.fn_discarded = path.join(self.directory, "discarded.vert")

        # Load configuration
        fn_config = path.join(self.directory, ManateeExtractor.FN_CONFIG)
        if not path.isfile(fn_config):
            raise WorkspaceNotFound(
                "Configuration file `%s` does not exist."
                "Did you forget to `ManateeExtractor.prepare` the directory?"
                % fn_config
            )
        with our_open(fn_config) as f:
            self.config = json.load(f)

        # Initialize Manatee objects
        self.corpus = manatee.Corpus(self.config["corpus"])
        self.struct = self.corpus.get_struct(self.config["struct"])
        self.lex_attr = self.corpus.get_attr(self.config["lex_attr"])

        if self.config["cql_attr"] is not None:
            self.corpus.set_default_attr(self.config["cql_attr"])

        self.struct_size = self.struct.size()

    def pt2fn_raw_pocs(self, phrase_type):
        return path.join(
            self.directory,
            ManateeExtractor.DN_RAW_POCS,
            str(phrase_type) + ".vert.gz"
        )

    def pt2fn_raw_vocab(self, phrase_type):
        return path.join(
            self.directory,
            ManateeExtractor.DN_RAW_VOCABS,
            str(phrase_type) + ".vert.gz"
        )

    def extract_words(self, word_category2regexp, attr, min_coverage=.75):
        """
        Somewhat hacky function that makes extraction of single word
        expressions several times faster than if through extract_phrases.

        It would be better to get rid of this.
        """
        if type(attr) == str:
            attr = self.corpus.get_attr(attr)
        else:
            assert type(attr) is manatee.PosAttr

        vals = [
            dict(
                text=attr.id2str(i),
                freq=attr.freq(i),
                word_categories=list()  # will be filled in later
            )
            for i in range(attr.id_range())  # position is an ID
        ]
        for wc, regexp in word_category2regexp.items():
            # All attributes defined by the regexp:
            ids_stream = attr.regexp2ids(regexp, False)
            if ids_stream.end():
                raise ValueError("Empty word category: %s" % wc)
            while not ids_stream.end():
                vals[ids_stream.next()]["word_categories"].append(wc)

        if any(len(val["word_categories"]) > 1 for val in vals):
            raise ValueError(
                "Multiple word categories per single %s" % attr.name,
                {val["text"]: val["word_categories"]
                 for val in vals if len(val["word_categories"]) > 1}
            )

        nb_tokens = sum(val["freq"] for val in vals)
        nb_covered = sum(
            val["freq"] for val in vals
            if len(val["word_categories"]) > 0
        )
        coverage = float(nb_covered) / nb_tokens
        if coverage < min_coverage:
            raise ValueError(
                "Low coverage: %.2f%%" % (100 * coverage),
                dict(
                    not_covered={
                        v["text"] for v in vals
                        if len(v["word_categories"]) == 0
                    }
                )
            )

        # Open a file for each word category.  We will go
        # through the corpus and write to all those files
        # simultaneously.
        wc2obj = {
            word_category: SelfClosingFile(
                self.pt2fn_raw_pocs((word_category, )),
                "w"
            )
            for word_category in word_category2regexp
        }
        val_i2obj = {
            val_i: (
                wc2obj[val["word_categories"][0]]
                if len(val["word_categories"]) == 1
                else None
            )
            for val_i, val in enumerate(vals)
        }

        for struct_i in range(self.struct.size()):

            beg = self.struct.beg(struct_i)
            end = self.struct.end(struct_i)

            for pos in range(beg, end):

                obj = val_i2obj[attr.pos2id(pos)]
                if obj is None:
                    continue

                cols = [
                    str(self.lex_attr.pos2id(pos)),  # lex ID
                    self.lex_attr.pos2str(pos),      # text
                    str(struct_i),                   # sentence ID
                    str(pos - beg),                  # in-sentence positions
                ]
                line = "\t".join(cols) + "\n"
                obj.write(line)

                if pos % 1000000 == 0:
                    print "{:,}".format(pos)

        for wc, obj in wc2obj.items():
            obj.close()

        for word_category in word_category2regexp:
            self._build_raw_vocab((word_category, ))

    def check_grammar(self, grammar, shortcuts=None, nb_outputs=0):
        """
        Tries to execute all CQL queries and fill in corresponding templates.
        If there is a problem, a raised exception localizes it
        (in which shortcut / pattern).

        If everything runs OK, it prints the first `nb_outputs` phrases
        for each query, so that you can check the sanity of your grammar.
        """
        if shortcuts is None:
            shortcuts = dict()

        # Check shortcuts
        for name, query in shortcuts.items():
            try:
                self.corpus.eval_query(query)
            except RuntimeError as error:
                raise RuntimeError(
                    "Error when evaluating CQL shortcut `%s`." % name,
                    dict(error=error)
                )

        # Check patterns
        for phrase_type, phrase_data in grammar.items():
            for pattern_i, pattern in enumerate(phrase_data["patterns"]):

                query = pattern["query"].format(**shortcuts)
                template = pattern["template"]

                try:
                    matches = match_cql(
                        corpus=self.corpus,
                        struct=self.struct,
                        lex_attr=self.lex_attr,
                        query=query,
                        template=template
                    )
                except RuntimeError as re:
                    # probably invalid CQL
                    # ---> propagate it up
                    raise RuntimeError(
                        "RuntimeError while trying "
                        "to get the first occurrence.",
                        dict(
                            phrase_type=phrase_type,
                            pattern_i=pattern_i,
                            error=re
                        )
                    )
                except MissingLabels as e:
                    # unable to fill in a template
                    # ---> make it more informative and re-raise
                    del e.args[1]["query"]
                    del e.args[1]["template"]
                    e.args[1]["phrase_type"] = phrase_type
                    e.args[1]["pattern_i"] = pattern_i
                    raise e

                matches = peekable(matches)

                try:
                    _, label2pos, __, ___ = matches.peek()
                except StopIteration:
                    # no occurrence found
                    # ---> cannot check filling in templates
                    # ---> raise error
                    raise NoOccurrenceFound(dict(
                        phrase_type=phrase_type,
                        pattern_i=pattern_i,
                    ))

                # If a phrase type tree has N nodes,
                # then the query must contain labels 1, 2, ..., N.
                nb_nodes = len(tuple(collapse(phrase_type)))
                mandatory_labels = set(range(1, nb_nodes + 1))
                missing_labels = {
                    label for label in mandatory_labels
                    if label not in label2pos
                }
                if len(missing_labels) > 0:
                    raise MissingLabels(
                        "Labels %i, ..., %i required",
                        dict(
                            phrase_type=phrase_type,
                            pattern_i=pattern_i,
                            missing_labels=missing_labels,
                        )
                    )
                # Everything is OK ---> print some phrases
                if nb_outputs > 0:
                    sys.stdout.write(
                        "{pt} / #{pattern_i}\n\n".format(
                            pt=str(phrase_type),
                            pattern_i=pattern_i + 1
                        )
                    )
                    for _, __, ___, text in islice(matches, nb_outputs):
                        sys.stdout.write(
                            "\t{phrase}\n".format(phrase=text)
                        )
                    sys.stdout.write("\n")

    def _get_colliding_pts(self, grammar):
        """
        Returns the set of phrase types from the grammar
        for whose the extraction already happened.
        """
        return {
            phrase_type for phrase_type in grammar
            if path.exists(self.pt2fn_raw_vocab(phrase_type))
        }

    def extract_phrases(self, grammar, shortcuts=None, mode="stop"):
        """
        Extracts phrases matching patterns defined by the given grammar.

        `mode` argument determines what should happen if the grammar contains
        phrase types for whose the extraction alredy happend.
        Possible values are:
            - stop:      raise CollidingPhraseType exception,
                         do not extract anything
            - skip:      ignore such phrase types
            - overwrite: re-extract and overwrite old data
        """
        if mode not in {"stop", "skip", "overwrite"}:
            raise ValueError(
                '`mode` argument must be set to stop|skip|overwrite.'
            )
        if shortcuts is None:
            shortcuts = dict()

        sys.stdout.write("Number of phrase types: %i\n" % len(grammar))
        self.check_grammar(grammar, shortcuts)

        colliding_pts = self._get_colliding_pts(grammar)
        if len(colliding_pts) > 0:
            sys.stdout.write(
                "There are colliding phrase types (%i):\n"
                % len(colliding_pts)
            )
            # Print a list of colliding phrase types
            for cpt in sorted(colliding_pts):
                sys.stdout.write("\t- %s\n" % str(cpt))
            # Take action according to `mode` argument
            if mode == "stop":
                raise CollidingPhraseType(
                    dict(colliding_pts=colliding_pts)
                )
            elif mode == "skip":
                sys.stdout.write(
                    "Colliding phrase types will be skipped.\n"
                )
                # Filter out colliding phrase types
                grammar = {
                    phrase_type: phrase_data
                    for phrase_type, phrase_data in grammar.items()
                    if phrase_type not in colliding_pts
                }
            elif mode == "overwrite":
                sys.stdout.write(
                    "Colliding phrase types will be overwritten.\n"
                )
                for cpt in colliding_pts:
                    for fn in [
                        self.pt2fn_raw_pocs(cpt),
                        self.pt2fn_raw_vocab(cpt)
                    ]:
                        if path.exists(fn):
                            remove(fn)

        sys.stdout.write("\n")
        for phrase_type, phrase_data in grammar.items():
            fn_pt = self.pt2fn_raw_pocs(phrase_type)
            if exists(fn_pt):
                remove(fn_pt)
            for pattern_i, pattern in enumerate(phrase_data["patterns"]):
                sys.stdout.write(
                    "{pt} / #{pattern_i}\n".format(
                        pt=str(phrase_type),
                        pattern_i=pattern_i + 1
                    )
                )
                query = pattern["query"].format(**shortcuts)
                template = pattern["template"]
                matches = match_cql(
                    corpus=self.corpus,
                    struct=self.struct,
                    lex_attr=self.lex_attr,
                    query=query,
                    template=template
                )
                self._save_raw_pocs(
                    raw_pocs=self._matches2raw_pocs(matches, phrase_type),
                    phrase_type=phrase_type
                )
                sys.stdout.write("\n")

            self._build_raw_vocab(phrase_type)

        sys.stdout.write(
            "Phrase extraction succesfully finished!\n"
        )

    def _matches2raw_pocs(self, matches, phrase_type):

        nb_nodes = len(tuple(collapse(phrase_type)))

        for struct_id, label2pos, label2lex_id, text in matches:
            positions = [
                label2pos[label]
                for label in range(1, nb_nodes + 1)
            ]
            lex_ids = [
                label2lex_id[label]
                for label in range(1, nb_nodes + 1)
            ]
            yield RawPOc(lex_ids, text, struct_id, positions)

    def _save_raw_pocs(self, raw_pocs, phrase_type):

        # Note on Progress Reporting
        # ---
        # Progress reporting uses struct IDS assuming that Manatee
        # yields results in the correct order.

        counter = 0  # number of processed occurrences
        report = ""  # last report

        fn = self.pt2fn_raw_pocs(phrase_type)
        with our_open(fn, "a", buffering=int(16 * 1024**2)) as f:
            for raw_poc in raw_pocs:

                f.write(str(raw_poc) + "\n")
                counter += 1

                if counter % 100000 == 0:
                    # Report progress
                    sys.stdout.write("\b" * len(report))
                    sys.stdout.write("\r")
                    progress = 100.0 * raw_poc.struct_id / self.struct_size
                    report = (
                        "[[ Progress: {:.2f}%, Occurrences: {:,} ]]"
                        .format(progress, counter)
                    )
                    sys.stdout.write(report)
                    sys.stdout.flush()

        # Final report
        sys.stdout.write("\b" * len(report))
        sys.stdout.write("\r")
        sys.stdout.write(
            "[[ Progress: 100%, Occurrences: {:,} ]]"
            .format(counter)
        )
        sys.stdout.write("\n")

    def _build_raw_vocab(self, phrase_type):
        """
        Counts (lex_ids, text) pair frequencies in a raw
        phrase occurrence file in a memory-friendly way.
        """
        fn_raw_pocs = self.pt2fn_raw_pocs(phrase_type)

        def iter_keys():
            with our_open(fn_raw_pocs) as f:
                for line in f:
                    yield "\t".join(line.strip().split("\t", 2)[:2])

        bounded_count(
            keys=iter_keys(),
            filename=self.pt2fn_raw_vocab(phrase_type),
            cache_size=self.cache_size
        )

    def _iter_raw_vocab(self, fn):

        with our_open(fn) as f:
            for line in f:
                parts = line.strip().split("\t")
                lex_ids = tuple(map(int, parts[0].split(",")))
                text = parts[1]
                freq = int(parts[2])
                yield (lex_ids, text), freq

    def _iter_unified(self, items):

        key = None
        sum_count = 0
        max_count = 0
        max_text = 0

        for (lex_ids, text), count in items:

            if key != lex_ids:
                if key is not None:
                    yield (key, max_text), sum_count
                key = lex_ids
                max_text = text
                max_count = count
                sum_count = count
            else:
                sum_count += count
                if count > max_count:
                    max_text = text
                    max_count = count

        if key is not None:
            yield (key, max_text), sum_count

    def _get_vassals(self, lexicon, vassal_coeff=0.75):

        # Considers only lexical IDs, phrase types are ignored.

        counter = 0  # number of processed phrases
        report = ""  # last report

        # Mapping from sorted `lex_ids` to lists of
        # (phrase ID, phrase frequency) pairs.
        sli = defaultdict(lambda: list())
        for i, _, lex_ids, __, count in lexicon:
            sli[tuple(sorted(lex_ids))].append((i, count))

        vassal_ids = set()

        for i, _, lex_ids, __, count in lexicon:
            lex_ids = sorted(lex_ids)
            counter += 1

            # Cycle over all possible sub-phrases.
            for k in range(1, len(lex_ids)):
                for sub in combinations(lex_ids, k):
                    for sub_i, sub_count in sli[sub]:

                        # Is this sub-phrase a vassal?
                        if sub_count * vassal_coeff < count:
                            vassal_ids.add(sub_i)

            if i % 1000 == 0:
                # Report progress
                sys.stdout.write("\b" * len(report))
                sys.stdout.write("\r")
                progress = 100.0 * float(i) / len(lexicon)
                report = (
                    "[[ Progress: {:.2f}%, Phrases: {:,}, Vassals: {:,} ]]"
                    .format(progress, counter, len(vassal_ids))
                )
                sys.stdout.write(report)
                sys.stdout.flush()

        # Final report
        sys.stdout.write("\b" * len(report))
        sys.stdout.write("\r")
        sys.stdout.write(
            "[[ Progress: 100%, Phrases: {:,}, Vassals: {:,} ]]"
            .format(counter, len(vassal_ids))
        )
        sys.stdout.write("\n")

        return vassal_ids

    def build_lexicon(self, min_count, max_items=sys.maxsize,
                      vassal_coeff=0.75):
        """
        Unifies large raw vocabularies of individual phrase types.

        The function removes "vassals".  A vassal is a phrase that occurs
        too often as a constituent of a certain longer phrase.

        # Generated files

            phrase_types.pickle
                A list of phrase types.

            lexicon.vert
                A tab-delimited file with 5 columns:
                 - phrase_id,
                 - phrase_type_id,
                 - lex_ids (space-delimited),
                 - textual_representation,
                 - frequency

            removed.vert
                A list of vassals.
        """
        sys.stdout.write("Collecting phrases from raw vocabularies ...\n")
        fns_raw_vocabs = glob(
            "{workspace}/{raw_vocabs}/*.vert.gz".format(
                workspace=self.directory,
                raw_vocabs=ManateeExtractor.DN_RAW_VOCABS
            )
        )
        phrase_types = [
            eval(fn.split("/")[-1][:-8])
            for fn in fns_raw_vocabs
        ]
        lexicon_dev = SortedListWithKey(key=lambda (p, c): -c)
        for pt_i, pt in enumerate(phrase_types):
            sys.stdout.write("\tPhrase Type: %s \n" % str(pt))

            # Gather sorted ((lex_ids, text), freq) tuples.
            # There may be multiple textual representations
            # per single lex_ids.
            fn = self.pt2fn_raw_vocab(pt)
            raw_vocab_items = self._iter_raw_vocab(fn)

            # Unify sorted stream of such tuples,
            # i.e. select the most common textual repr.
            # and sum up frequencies.
            unified_vocab_items = self._iter_unified(raw_vocab_items)

            # Extend lexicon.
            for (lex_ids, text), count in unified_vocab_items:
                if count >= min_count:
                    lexicon_dev.add(((pt_i, lex_ids, text), count))
                    if len(lexicon_dev) > max_items:
                        lexicon_dev.pop()

        sys.stdout.write("\n")

        # Final lexicon structure -- 5 columns
        lexicon = [
            (i, pt_i, lex_ids, text, count)
            for i, ((pt_i, lex_ids, text), count)
            in enumerate(lexicon_dev)
        ]
        del lexicon_dev

        sys.stdout.write("Detecting vassals (coeff=%.2f) ...\n"
                         % vassal_coeff)
        vassal_ids = self._get_vassals(lexicon, vassal_coeff)
        sys.stdout.write("\n")

        sys.stdout.write("Persisting lexicon to disk ...\n")

        with our_open(self.fn_phrase_types, "w") as f:
            pickle.dump(phrase_types, f)

        new_i = 0
        with our_open(self.fn_lexicon, "w") as f_lexicon:
            for old_i, pt_i, lex_ids, text, count in lexicon:
                if old_i not in vassal_ids:
                    cols = [
                        str(new_i),
                        str(pt_i),
                        ",".join(map(str, lex_ids)),
                        text,
                        str(count)
                    ]
                    f_lexicon.write("\t".join(cols) + "\n")
                    new_i += 1

        with our_open(self.fn_discarded, "w") as f_removed:
            for old_i, __, ___, text, ____ in lexicon:
                if old_i in vassal_ids:
                    f_removed.write(text + "\n")

    def iter_pocs(self, use_ids=False):
        """
        Yields unsorted pocs of phrases from the lexicon.

        If `use_ids` is set on, textual representations
        are replaced by lexicon IDs.
        """
        with our_open(self.fn_phrase_types) as pt_file:
            phrase_types = pickle.load(pt_file)

        mapping = dict()  # {(pt_i, lex_ids): phrase}
        with our_open(self.fn_lexicon) as lex_file:
            for line in lex_file:
                parts = line.strip().split("\t")
                phrase_i = int(parts[0])
                pt_i = int(parts[1])
                lex_ids = tuple(map(int, parts[2].split(",")))
                text = parts[3]
                mapping[(pt_i, lex_ids)] = (
                    phrase_i if use_ids
                    else text
                )

        for pt_i, phrase_type in enumerate(phrase_types):
            fn_ocs = self.pt2fn_raw_pocs(phrase_type)
            with our_open(fn_ocs) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    lex_ids = tuple(map(int, parts[0].split(",")))
                    phrase = mapping.get((pt_i, lex_ids), None)
                    if phrase is not None:
                        sentence_id = int(parts[2])
                        # A valid poc has sorted positions.
                        positions = tuple(sorted(map(int, parts[3].split(","))))
                        yield phrase, sentence_id, positions
