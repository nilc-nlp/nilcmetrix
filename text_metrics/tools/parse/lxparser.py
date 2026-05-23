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

from text_metrics.conf import config
from text_metrics.tools.parse.api import Parser, TagSet
from nltk.tree import Tree
import subprocess
from text_metrics.profiling import timed_block
import tempfile
import codecs
import re
import os


class LxParser(Parser):

    """A simple interface for LXParser. This parser needs two options to be
        configured:

        * LX_STANFORD_PATH: the path to the folder containing the
            'stanford-parser.jar' file.
        * LX_MODEL_PATH: the path to the model file to be used (e.g.,
            cintil.ser.gz for Portuguese).
    """

    def __init__(self):
        self.tagset = LxTagSet()

    def _cmd(self, filename):
        return ['java', '-Xmx500m', '-cp',
                config['LX_STANFORD_PATH'] + '/stanford-parser.jar',
                'edu.stanford.nlp.parser.lexparser.LexicalizedParser',
                '-escaper', 'edu.stanford.nlp.process.PTBEscapingProcessor'
                '-tokenized', '-sentences', 'newline', '-outputFormat',
                'oneline', '-uwModel',
                'edu.stanford.nlp.parser.lexparser.BaseUnknownWordModel',
                config['LX_MODEL_PATH'], filename]

    @staticmethod
    def normalize(sent):
        """Normalize the sentence to make it suitable for analysis by
        the parser.

        Required arguments:
        :sent: a string containing the sentence to normalize.
        """
        # This is not currently being used. Instead, we are passing
        # "-escaper edu.stanford.nlp.process.PTBEscapingProcessor"
        # to the parser.

        sent = sent.replace('(', ' -LRB- ')
        sent = sent.replace(')', ' -RRB- ')

        sent = sent.replace('[', ' -LSB- ')
        sent = sent.replace(']', ' -RSB- ')

        sent = sent.replace('{', ' -LCB- ')
        sent = sent.replace('}', ' -RCB- ')

        return re.sub(r' +', ' ', sent)

    def parse_sents(self, sents):
        """Parse a list of strings.

        :sents: a list of strings to parse.
        :returns: a list of nltk.tree.Tree objects, one for each tree generated
            by LXParser.
        """
        fdesc, input_file_path = tempfile.mkstemp(text=True)
        os.close(fdesc)

        with codecs.open(input_file_path, mode='w', encoding='utf-8') as infile:
            infile.write('\n'.join(sents))

        return_value = self.run(input_file_path)
        os.remove(input_file_path)

        return return_value

    def run(self, filename):
        """Runs the parser for a file.

        :filename: the file containing the sentences to be analyzed.
        :returns: a list of nltk.tree.Tree objects, one for each tree generated
            by LXParser.
        """

        with timed_block("jvm.stanford"):
            p = subprocess.Popen(self._cmd(filename),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            raw_lines = p.communicate()[0].decode('utf-8').split('\n')
        tree_lines = [line for line in raw_lines if line.startswith('(')]
        trees = [Tree.fromstring(line) for line in tree_lines]

        return trees


class LxTagSet(TagSet):

    """The tagset used by the LXParser. """

    sentence_node_labels = ['S']
