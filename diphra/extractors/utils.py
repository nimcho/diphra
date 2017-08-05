#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Lukas Banic <lukas.banic@protonmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from string import Formatter


def resolve_shortcuts(shortcuts):
    """
    Resolves dependencies among shortcuts,
    (cases where one shortcut uses another one).

    Shortcuts are processed in topological order.

    # Arguments
        shortcuts: A {name: query} dictionary, where
            some queries contain placeholders, i.e.
            depend on other shortcuts.

    # Returns
        A new {name: query} dictionary with all
            placeholders properly filled in.

    # Raises
        ValueError: If there is an unmet or cyclic dependency.

    # Examples
        >>> rs = resolve_shortcuts(dict(x="a", y="{x}b", z="{y}c"))
        >>> rs == {'x': 'a', 'y': 'ab', 'z': 'abc'}
        True
        >>> rs = resolve_shortcuts(dict(x="a{x}"))
        Traceback (most recent call last):
         ...
        ValueError: Shortcuts contain a cyclic dependency.
        >>> rs = resolve_shortcuts(dict(x="a{y}"))
        Traceback (most recent call last):
         ...
        ValueError: Unmet dependency `y`.
    """
    # Uses Kahn's algorithm for topological sorting
    # ---

    # Set of outgoing edges for each shortcut.
    # An edge A ---> B means that the shortcut B
    # depends on the shortcut A.
    edges = {name: set() for name in shortcuts.keys()}

    # Number of incoming edges for each shortcut.
    name2nb_deps = {name: 0 for name in shortcuts.keys()}

    # Fill in `edges` and `name2nb_deps`
    for name, query in shortcuts.items():
        dependencies = [
            field_name for _, field_name, __, ___
            in Formatter().parse(query)
            if field_name is not None
        ]
        name2nb_deps[name] = len(dependencies)
        for dependency in dependencies:
            if dependency not in edges:
                raise ValueError("Unmet dependency `%s`." % dependency)
            edges[dependency].add(name)

    topological_order = []
    candidates = {
        name for name, nb_deps
        in name2nb_deps.items()
        if nb_deps == 0
    }
    while len(candidates) > 0:
        candidate = candidates.pop()
        topological_order.append(candidate)
        for name in edges[candidate]:
            name2nb_deps[name] -= 1
            if name2nb_deps[name] == 0:
                candidates.add(name)

    if len(topological_order) != len(shortcuts):
        raise ValueError("Shortcuts contain a cyclic dependency.")

    new_shortcuts = dict()
    for name in topological_order:
        new_shortcuts[name] = shortcuts[name].format(**new_shortcuts)

    assert len(shortcuts) == len(new_shortcuts)
    return new_shortcuts
