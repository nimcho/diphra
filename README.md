# DiPhra: Extraction and Interpretation of Phrases

![python: 2.7](https://img.shields.io/badge/python-2.7-brightgreen.svg)
[![license](https://img.shields.io/badge/license-LGPL--2.1-blue.svg)](https://github.com/nimcho/diphra/blob/master/LICENSE)
![status: early development](https://img.shields.io/badge/status-early%20development-red.svg)

DiPhra is a library for modeling the semantics of phrases,
written in Python, with critical parts optimized with Cython. 
Use DiPhra if you want to turn your >1B corpora into models
capable of resolving idiomatic expression by modeling
phrase synonymy.

DiPhra combines two simple and powerful concepts:

- Shallow syntactic parsing for phrase extraction
  (usually just a simple set of patterns over POS tag sequences)
- Distributional semantics for phrase interpretation
  (co-occurrence statistics -- phrases occurring in similar contexts
   tend to have similar meanings)

Extraction is independent of interpretation,
so it is possible to combine DiPhra with existing phrase extractors
and named entity recognizers.

## Installation

Dependencies:

 - [NumPy and SciPy]
 - [HDF5], [h5py]
 - [Manatee]

It is also recommended to install a fast BLAS library (such as [ATLAS] or [OpenBLAS])
before installing NumPy &mdash; speeds up performance by as much as an order of magnitude.

The easiest way to install DiPhra is using pip:

```
sudo pip install git+https://github.com/nimcho/diphra
```

## Data Structure: Phrase Occurrences (POcs)

DiPhra goes beyond mere merging tokens.  You can work with phrases
that occur nested, overlapping or discontinuous within corpora.

Let's say we have a single-sentence corpus:

```
The police kept close tabs on him during the holidays .
```

The corresponding POcs could look like:

| Phrase          | Sentence ID   | In-Sentence Positions |
| -------------   |:-------------:| --------------------- |
| the             | 0             | 0                     |
| police          | 0             | 1                     |
| keep            | 0             | 2                     |
| keep close tabs | 0             | 2, 3, 4               |
| keep tabs       | 0             | 2, 4                  |
| keep tabs on    | 0             | 2, 4, 5               |
| close           | 0             | 3                     |
| close tabs      | 0             | 3, 4                  |
| tabs            | 0             | 4                     |
| on              | 0             | 5                     |
| he              | 0             | 6                     |
| during          | 0             | 7                     |
| during holidays | 0             | 7, 9                  |
| the             | 0             | 8                     |
| holidays        | 0             | 9                     |
| .               | 0             | 10                    |

The third column (in-sentence positions) must be sorted,
so that a consistent sorting/merging of POcs datasets is possible.

With such format, you can easily combine multiple phrase extractors,
named entity recognizers or whatever.  In the end,
all you need is a large sorted POcs file.

### Vertical POcs

A simple format to store pocs to disk is a tab-separated vertical
(with variable number of columns).

```
...
keep               0    2
keep close tabs    0    2    3    4
keep tabs          0    2    4
keep tabs on       0    2    4    5
...
```

The vertical format is human-readable and makes it easy to solve
some basic problems with standard unix tools.  

If we know that the max. number of positions covered by a single poc
is 3, then the following command will sort the file:

```bash
sort pocs.vert -n -t$'\t' -k2,2 -k3,3 -k4,4 -k5,5
```

When we combine multiple phrase extractors, we want to merge 
the sorted files:

```bash
sort pocs1.vert pocs2.vert -m -n -t$'\t' -k2,2 -k3,3 -k4,4 -k5,5
```

For machine learning purposes, we may want to randomize sentences:

```bash
sort pocs.vert -R -s -k2,2 -t$'\t'
```

In python, you can use `diphra.pocs.VerticalPOcs` class to read and
write vertical pocs.

### HDF5 POcs

Parsing pocs from a plain text file may be quite slow.  It would become
a main bottleneck for machine learning algorithms passing the whole dataset
several times.  A workaround is to compile them into HDF5 file.  

```python
from diphra.pocs import VerticalPOcs, HDF5POcs

HDF5POcs.build(
    pocs=VerticalPOcs("my_pocs.vert"),
    output_name="my_pocs.hdf5"
)

pocs = HDF5POcs("my_pocs.hdf5")
```

## POcs2Vec: Word2Vec Meets Phrase Occurrences

POcs2Vec is a modified implementation of Skip-Gram with Negative Sampling,
one of the method from the well-known [Word2Vec](https://code.google.com/archive/p/word2vec/)
system for efficient estimation of distributed representations
of words and phrases.

POcs2Vec consumes pocs (phrase occurrences), which can be nested,
overlapping and discontinuous.

A typical pipeline looks like:

 1. **Create a large pocs dataset.**  
    Employ any phrase extraction tools you like,
    identify a diverse set of phrases.
 1. **Bound the vocabulary.**  
    There is no `min_count` parameter in POcs2Vec as the vocabulary size
    for multi-word expressions may easily go to hundreds of millions.
    It is up to you to bound it to some meaningful value, e.g. keep only
    1 million most frequent items.
 1. **Sort & Merge.**  
    If you use multiple phrase extractors, you probably have several
    pocs files, sort them and merge them together.  The easiest
    solution is `VerticalPocs.pocs_sort(...)` function, which provides
    an external sort and is happy with gzipped files.
 1. **HDF5-ify.**  
    To speed up the training, compile your pocs into HDF5.
 1. **Now train a POcs2Vec model.**  

## Phrase Extraction with Manatee

DiPhra includes a light wrapper over Manatee corpus manager.  Use it 
to extract phrases by the means of explicit rules (grammar patterns).
See [the tutorial](./examples/phrase-extraction-with-manatee.ipynb).

More tutorials will come soon.

## Bottlenecks

While POcs2Vec fed with HDF5POcs runs as fast as (maybe even faster than)
original Word2Vec, the phrase extraction is very slow and there are lots
of things to improve.

 - Treating words as phrases with length 1 is a desired conceptual
   minimalism, but "extracting" words with the same procedure
   is super slow and completely unnecessary.  Current (hacky) solution
   is using `ManateeExtractor.extract_words(...)`, which is orders
   of magnitude faster than extracting words with
   `ManateeExtractor.extract_phrases(...)`.  
   
 - ManateeExtractor does not leverage the order of results
   returned by Manatee.  In the end, POcs are smashed together
   and have to be sorted.
   
 - Certain parts could be cythonized and parallelized.

  [HDF5]: https://support.hdfgroup.org/HDF5/
  [h5py]: http://www.h5py.org/
  [Manatee]: https://nlp.fi.muni.cz/trac/noske
  [NumPy and SciPy]: http://www.scipy.org/Download
  [ATLAS]: http://math-atlas.sourceforge.net/
  [OpenBLAS]: http://xianyi.github.io/OpenBLAS/
