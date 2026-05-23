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
import os
from text_metrics.tools.dependency.api import DependencyParser
from text_metrics.conf import config
from nltk.parse.malt import MaltParser as NltkMaltParser
from text_metrics.profiling import timed_block


class MaltParser(DependencyParser):

    """Docstring for MaltParser. """

    def __init__(self, tagger=None):
        self.tagger = tagger

    def tagger_func(self, sent):
        return self.tagger.tag(sent)

    def parse_sents(self, sents):
        with timed_block("jvm.malt"):
            os.environ['MALT_PARSER'] = config['MALT_WORKING_DIR'] + '/malt.jar'
            parser = NltkMaltParser(parser_dirname=config['MALT_WORKING_DIR'],
                                    model_filename=config['MALT_MCO'],
                                    additional_java_args=config['MALT_JAVA_ARGS'],
                                    tagger=self.tagger_func)
            graphs = [list(graph)[0] for graph in parser.parse_sents(sents)]

        # Sometimes, there is an empty graph at the end of the list. Delete it.
        if len(graphs) > 0 and len(graphs[-1].nodes) == 1:
            del graphs[-1]

        return graphs
