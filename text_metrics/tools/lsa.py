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

import pickle

import numpy as np


class LsaSpace(object):
    """Represents an LSA space, that can be used to compute similarities
    between text fragments (texts, paragraphs, sentences, and so on).
    """

    # Fallback for words not in the LSA model. Read-only at call sites,
    # so a single shared array is enough.
    _UNKNOWN_WORD_VECTOR = np.full(300, 0.001)

    def __init__(self, model_path):
        """Load an LSA space from a file.

        :model_path: path to the model file.
        """
        f = open(model_path, 'rb')
        self._lsa_model = pickle.load(f)
        f.close()

        # The same sentence is fed to LSA many times within a run:
        # `LsaSentenceAllMean` makes N*(N-1)/2 sentence-pair calls (each
        # sentence appears in N-1 pairs), `LsaSpanBase` recomputes every
        # past sentence's vector at every span step (~N^2/2 calls total).
        # `get_vector` is a pure function of the token sequence, so
        # caching by `tuple(tokens)` collapses the redundancy.
        self._vector_cache = {}

    def word2vec(self, word):
        return self._lsa_model.get(word.lower(), LsaSpace._UNKNOWN_WORD_VECTOR)

    def get_vector(self, tokens):
        """Return the dense num_topics-dimensional LSA vector for a
        sentence, memoized per process by token tuple. Returns an
        all-zero vector if no token survives the len(word) > 2 filter.
        """
        key = tuple(tokens)
        cached = self._vector_cache.get(key)
        if cached is not None:
            return cached

        word_vec = [self.word2vec(w) for w in tokens if len(w) > 2]
        if not word_vec:
            dense = np.zeros(self.num_topics)
        else:
            dense = np.array(word_vec).mean(axis=0)

        self._vector_cache[key] = dense
        return dense

    @property
    def num_topics(self):
        """Return the number of topics in the model."""
        return 300

    @staticmethod
    def compute_similarity(v1, v2):
        """Cosine similarity on dense numpy arrays. Returns 0.0 if
        either input has zero norm (matches gensim's `cossim` on
        empty sparse vectors).
        """
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))
