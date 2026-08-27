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
import re

import nlpnet
from text_metrics.tools.tag.api import Tagger
from text_metrics.tools.tag.macmorpho import MacMorphoTagSet
from text_metrics.conf import config

# Matches a token made up entirely of dash characters (travessão, en-dash,
# horizontal bar, hyphen) and runs thereof.
_DASH_ONLY = re.compile(r'^[—–―-]+$')


class NLPNetTagger(Tagger):

    """Docstring for NLPNetTagger. """

    def __init__(self, data_dir=None):
        self.tagset = MacMorphoTagSet()
        self._data_dir = data_dir
        self._tagger = None

    def load_tagger(self):
        if not self._data_dir:
            self._data_dir = config['NLPNET_DATA_DIR']

        nlpnet.set_data_dir(self._data_dir)
        self._tagger = nlpnet.POSTagger()

    def tag(self, tokens):
        if not self._tagger:
            self.load_tagger()
        assert self._tagger is not None
        tags = self._tagger.tag_tokens(tokens)
        return [(tok, self._clean_tag(tok, tag))
                for tok, tag in zip(tokens, tags)]

    @staticmethod
    def _clean_tag(token, tag):
        """Fix systematic nlpnet mis-taggings for a single token.

        nlpnet tags some standalone symbols as nouns, which inflates word/noun
        counts and poisons the min-frequency metrics (the symbol is treated as a
        rare unknown word). Force them to "PU":

        - dash-only tokens — a sentence-initial travessão would otherwise add
          ~one noun per dialogue turn;
        - a standalone "&" — in the corpus it is the connector in firm names and
          paired titles ("Mawson & Williams"), semantically "e", never a noun.

        Tokens that carry letters keep their real tag, so "bem-vindo" never
        matches the dash rule. (The tokenizer does split "AT&T" into AT / & / T,
        so that inner "&" is normalized too; harmless — it de-inflates an
        already-fragmented brand's noun count rather than the reverse.)

        Add more corrections here as further mis-taggings are found.
        """
        if _DASH_ONLY.match(token) or token == '&':
            return 'PU'
        return tag
