# -*- coding: utf-8 -*-
# Coh-Metrix-Dementia - Automatic text analysis and classification for dementia.
# Copyright (C) 2014  Andre Luiz Verucci da Cunha
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import unicode_literals, print_function, division
import text_metrics.resource_pool


class DelafStemmer(object):

    """Stemmer backed by the `delaf_words` Postgres table.

    Lemmas are deterministic given (word, pos), so results are memoized in a
    process-wide dict to avoid redundant DB queries — every guten.py metric
    re-stems the same content-word list, which on natural text drives many
    tens of thousands of repeated lookups for the same handful of types.
    """

    def __init__(self):
        self._cache = {}

    def get_lemma(self, word, pos=None):
        if pos == "ADJ":
            pos = "A"

        word = word.lower()
        key = (word, pos)

        if key in self._cache:
            return self._cache[key]

        helper = text_metrics.resource_pool.rp.db_helper()
        delaf_word = helper.get_delaf_word(word, pos)
        if delaf_word is None:
            delaf_word = helper.get_delaf_word(word)

        lemma = delaf_word.lemma if delaf_word is not None else None
        self._cache[key] = lemma
        return lemma
