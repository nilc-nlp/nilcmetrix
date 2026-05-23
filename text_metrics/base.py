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
from text_metrics.utils import is_valid_id
from text_metrics.resource_pool import rp as default_rp
from text_metrics.profiling import timed_block
import numpy as np
import codecs
import collections
import logging
from lxml import etree
import re

logger = logging.getLogger(__name__)


class Text(object):
    """Represents a text: its content and metadata."""

    def __init__(self, content='', revised_content='',
                 filepath='', revised_filepath='', encoding='utf-8',
                 raw_as_xml=False, **meta):
        """Form a text. The text's content can be informed via a string or a
        file path to be read. A text can also be revised.

        To create a Text object from a string, use the attribute 'content'.

        > t = Text(content='The book is on the table.')

        Or simply:

        > t = Text('The book is on the table.')

        To create a Text from the contents of a file, use 'filepath'.

        > t = Text(filepath='./text.txt')

        If the file is not encoded in UTF-8, use 'encoding':

        > t = Text(filepath='./text.txt', encoding='ISO-8859-1')

        If the text is revised, the revised content can be informed both
        as a string or as a file path. For the first case, use
        'revised_content'; for the second one, 'revised_filepath'.

        Examples:

        > t = Text('the book is on the table',
                   revised_content='The book is on the table.')

        > t = Text(filepath='./text.txt',
                   revised_filepath='./text_rev.txt')

        It's possible to inform the raw content as a string and the
        revised content as a file path, and vice-versa.

        It's possible to inform as many keyword arguments as wanted. Their
        values will be stored in the 'meta' attribute:

        > t = Text('...', author='John Doe', date='today')
        > t.meta
            {'author': 'John Doe', 'date': 'today'}
        """

        self.meta = meta
        self.filepath = filepath
        # Set raw content.
        # if not filepath:
        #     self.raw_content = content
        # else:
        #     with codecs.open(filepath, mode='r', encoding=encoding)\
        #             as input_file:
        #         self.raw_content = input_file.read()
        if filepath:
            with codecs.open(filepath, mode='r', encoding=encoding)\
                    as input_file:
                content = input_file.read()

        # 2021: Esse bloco abaixo não faz muito sentido, e remove as quebras de linhas em textos acima de 1000 tokens.
        # content_tokens = content.split()
        # if len(content_tokens) > 1000:
        #     index = [content_tokens[1000:].index(token) for token in content_tokens[1000:] if token[-1] in ['.', '!', '?']]
        #     if index:
        #         index = index[0]
        #         content = ' '.join(content_tokens[:1000 + index + 1])
        #     content = ' '.join(content_tokens)

        # Set title.
        title = re.search('<title>.*?</title>', content)
        if title:
            self.title = title.group(0)
        else:
            self.title = ''
        # Set number of subtitles.
        subtitles = re.findall('<subtitle>.*?</subtitle>', content, re.DOTALL)
        self.subtitles = len(subtitles)
        # Set url.
        url = re.search('<url>.*?</url>', content)
        if url:
            self.url = url.group(0)
        else:
            self.url = ''
        # Set raw content.
        expression = re.compile('''
                  <(?P<name>.*)>.*?</(?P=name)>| # remove marcações <x>...</x>
                  <.*?>|                   # remove marcações <x>
                  \[...\]|                      # remove [...]
                  \(\)|                         # remove parêntese vazio ()
                  •|                            # remove o caracter •
                                ''',
                                re.VERBOSE | re.DOTALL)

        self.raw_content = re.sub(expression, '', content)
        replace_list = [
            #  ('“', '"'),
            ('“', ' “ '),
            ('”', ' ” '),
            ('‘', ' ‘ '),
            ('’', ' ’ '),
            ('"', ' " '),
            ("'", " ' "),
            ('``', ' `` '),
            ('´´', ' ´´ '),
            ('°', 'o'),
            ('º', 'o'),
            ('è', 'e'),
            ('ë', 'e'),
            ('Ë', 'E'),
            ('È', 'E'),
            ('î', 'i'),
            ('Î', 'I'),
            ('Ï', 'I'),
            ('ï', 'i'),
            ('ñ', 'n'),
            ('ò', 'o'),
            ('ø', 'o'),
            ('μ', 'u'),
            ('µ', 'u'),
            ('ü', 'u'),
            ('û', 'u'),
            ('ö', 'o'),
            ('Ö', 'O'),
            ('ä', 'a'),
            ('Ä', 'A'),
            ('Å', 'A'),
            ('Ñ', 'N'),
            ('α', 'a'),
            ('\u0456', 'i'),
            ('\u0430', 'a'),
            ('ˆ', ''),
            ('√', 'r'),
            ('π', 'pi'),
            ('ω', 'w'),
            ('λ', 'l'),
            ('Δ', 'D'),
            ('β', 'B'),
            ('ς', 'S'),
            ('ο', 'o'),
            ('∈', 'E'),
            ('≤', '<='),
            ('ē', 'e'),
            ('δ', 's'),
            ('τ', 't'),
            ('α', 'a'),
            ('θ', 'T'),
            ('Θ', 'T'),
            ('η', 'n'),
            ('ρ', 'p'),
            ('·', '.'),
            ('ε', 'e'),
            ('ψ', 'p'),
            ('σ', 's'),
            ('ÿ', 'y'),
            ('ß', 'B'),
            ('ª', 'a'),
            ('ù', 'u'),
            ('œ', ''),
            ('Œ', ''),
            ('ì', 'i'),
            ('Ì', 'I'),
            ('þ', 'b'),
            ('æ', 'a'),
            ('Æ', 'a'),
            ('ø', 'o'),
            ('Ø', 'O'),
            ('…', '...'),
            #  ('``', '"'),
            #  ('``', '"'),
            ('\ufeff', ''),  # remove BOM marks
            ('\xc2\xa0', '')    # remove BOM marks
        ]
        for old, new in replace_list:
            self.raw_content = self.raw_content.replace(old, new)
        while '\n\n' in self.raw_content:
            self.raw_content = self.raw_content.replace('\n\n', '\n')
        # Set revised content.
        revised = False
        if revised_content:
            self.revised_content = revised_content
            revised = True
        elif revised_filepath:
            with codecs.open(filepath, mode='r', encoding=encoding)\
                    as revised_file:
                self.revised_content = revised_file.read()
            revised = True

        if revised and raw_as_xml:
            # We need to reinterpret the raw content as XML.
            tree = etree.fromstring("<raw-content>{}</raw-content>"
                                    .format(self.raw_content))

            empty_utterances = tree.findall('empty')
            disf_utterances = tree.findall('disf')

            self.meta['empty'] = empty_utterances
            self.meta['disf'] = disf_utterances

            raw_content = ' '.join([phrase.strip()
                                    for phrase in tree.itertext()]).strip()

            self.raw_content = raw_content

        _content = self.revised_content if revised else self.raw_content

        self.paragraphs = [line.strip() for line in _content.split('\n')
                           if line and not line.isspace()]

    @staticmethod
    def load_xml(filepath):
        """Load a file in Coh-Metrix-Dementia's XML format."""

        tree = etree.parse(filepath)
        texts = []
        for etext in tree.findall('text'):
            metadata = {}
            xml_metadata = etext.findall('meta')
            for metadatum in xml_metadata:
                metadata[metadatum.get('type')] = metadatum.text

            empty_utterances = etext.find('raw-content').findall('empty')
            disf_utterances = etext.find('raw-content').findall('disf')

            metadata['empty'] = empty_utterances
            metadata['disf'] = disf_utterances

            raw = etext.find('raw-content')
            raw_content = ' '.join([phrase.strip()
                                    for phrase in raw.itertext()]).strip()

            revised = etext.find('revised-content')
            revised_content = '\n'.join(
                [phrase.strip() for phrase in revised.itertext()]).strip()

            text = Text(content=raw_content,
                        revised_content=revised_content,
                        **metadata)
            texts.append(text)

        return texts

    def __repr__(self):
        return '<Text: "%s">' % ((self.paragraphs[0][:70] + '...')
                                 if len(self.paragraphs[0][:70]) == 70
                                 else self.paragraphs[0][:70])


class Category(object):
    """Represents a set of taxonomically related metrics.
    """
    def __init__(self, metrics=None, name=None, table_name=None, desc=None):
        """Form a category.

        Keyword arguments:
        :name: A succint name of the category (e.g., 'Basic Counts'). If
            no name is provided, the class name is used. (default None).
        :table_name: The name of the table in text_metrics_user_data that contains
            the values of this category on the users's texts. If no value is
            specified, Coh-Metrix-Port will check whether 'name' is a valid
            table name; if so, 'name' is used as the table name. (default None)
        :desc: A longer description of the category. Used for UI purposes.
            If no value is passed, the docstring of the class is used.
            (default None)
        """
        self.metrics = metrics

        if name is None and hasattr(self.__class__, 'name'):
            name = self.__class__.name

        if table_name is None and hasattr(self.__class__, 'table_name'):
            table_name = self.__class__.table_name

        if name is None:
            name = self.__class__.__name__
        self.name = name

        if table_name is None:
            if is_valid_id(name):
                table_name = name
            else:
                raise ValueError('No valid table name provided.')
        self.table_name = table_name

        if desc is None:
            desc = self.__doc__

        self.desc = desc

    def _set_metrics_from_module(self, module):
        """Set self.metrics as the list of Metric subclasses declared in
        a module.

        Required arguments:
        :module: the name of module that will be scanned for metrics.
        """
        import sys
        import inspect

        self.metrics = [obj() for _, obj
                        in inspect.getmembers(sys.modules[module])
                        if inspect.isclass(obj) and issubclass(obj, Metric)]

    def values_for_text(self, text, rp=default_rp):
        """Calculate the value of each metric in a text and return it in a
        ResultSet.

        Required arguments:
        :text: the text whose metrics will be extracted.

        :returns: a ResultSet containing the calculated metrics.
        """
        values = []
        for m in self.metrics:
            #print(m)
            try:
                logger.info('Calculating metric %s.', m.name)
                with timed_block("metric." + m.column_name):
                    values.append((m, round(m.value_for_text(text), 5)))
            except ZeroDivisionError:
                values.append((m, 0))

        # metrics_values = ResultSet([(m, m.value_for_text(text))
        #                             for m in self.metrics])
        metrics_values = ResultSet(values)

        return metrics_values

    def values_for_texts(self, texts, rp=default_rp):
        """Calculate the value of each metric in a set of texts and return them
        as a ResultSet.

        :texts: a list of Text objects.
        :rp: the resource pool to be used.
        :returns: a ResultSet containing the calculated metrics for each text.

        """
        return ResultSet((t, self.values_for_text(t, rp)) for t in texts)

    def __repr__(self):
        return '<Category: %s: %s>' % \
            (self.name, str([m.name for m in self.metrics]))

    def __getattr__(self, attr):
        # A metric's column name can be used as an attribute to access its
        # object in self.metrics.
        for m in self.metrics:
            if m.column_name == attr:
                return m
        raise AttributeError('%s: no such metric.' % attr)

    def __getitem__(self, key):
        # A metric's column name and its name can be used as an index
        # to access its object in self.metrics.
        for m in self.metrics:
            if m.column_name == key or m.name == key:
                return m
        raise KeyError('%s: no such metric.' % key)


class Metric(object):
    """A metric is a textual characteristic.
    """

    def __init__(self, name=None, column_name=None, desc=None):
        """Form a metric.

        Keyword arguments:
        :name: A succint name of the metric (e.g., 'Flesch index'). If
            no name is provided, the class name is used. (default None)
        :table_name: The name of the column in the table corresponding to
            the category of this metric in text_metrics_user_data. If no value is
            specified, Coh-Metrix-Port will check whether 'name' is a valid
            table name; if so, 'name' is used as the table name. (default None)
        :desc: A longer description of the metric. Used for UI purposes.
            (default None)
        """
        if name is None and hasattr(self.__class__, 'name'):
            name = self.__class__.name

        if column_name is None and hasattr(self.__class__, 'column_name'):
            column_name = self.__class__.column_name

        if name is None:
            name = self.__class__.__name__
        self.name = re.sub(r"(\s{2,})", " ", name)

        if column_name is None:
            if is_valid_id(name):
                column_name = name
            else:
                raise ValueError('No valid column name provided.')
        self.column_name = column_name

        if desc is None:
            desc = self.__doc__

        self.desc = desc

    def value_for_text(self, text, rp=default_rp):
        """Calculate the value of the metric in the text.

        Required arguments:
        :text: The text to be analyzed.

        :returns: an integer value, corresponding to the metric.
        """
        raise NotImplementedError('Subclasses should implement this method!')

    def __repr__(self):
        return '<Metric: %s> ' % (self.name)


class MetricsSet(object):
    def __init__(self, categories):
        self.categories = categories
        self.names = []
        for _category in self.categories:
            for _metric in _category.metrics:
                self.names.append(_metric.name)
        self.column_names = []
        for _category in self.categories:
            for _metric in _category.metrics:
                self.column_names.append(_metric.column_name)
        self.descs = []
        for _category in self.categories:
            for _metric in _category.metrics:
                self.descs.append(_metric.desc)
        self.all_info = []
        for _category in self.categories:
            for _metric in _category.metrics:
                self.all_info.append((_metric.name,
                                     _metric.column_name,
                                     _metric.desc))

    def _set_categories_from_module(self, module):
        """Set self.categories as the list of Category subclasses
            declared in a module.

        Required arguments:
        :module: the name of module that will be scanned for categories.
        """
        import sys
        import inspect

        self.categories = [obj() for _, obj
                           in inspect.getmembers(sys.modules[module])
                           if inspect.isclass(obj)
                           and issubclass(obj, Category)]

    def values_for_text(self, text, rp=default_rp):
        values = []

        for cat in self.categories:
            logger.info('Calculating category %s.', cat.name)
            values.append((cat, cat.values_for_text(text, rp)))

        # return ResultSet([(c, c.values_for_text(t)) for c in self.categories])
        return ResultSet(values)

    def values_for_texts(self, texts, rp=default_rp):
        """Calculate the value of each metric in a set of texts and return them
        as a ResultSet.

        :texts: a list of Text objects.
        :rp: the resource pool to be used.
        :returns: a ResultSet containing the calculated metrics for each text.

        """

        values = []
        ntexts = len(texts)
        for i, text in enumerate(texts):
            logger.info('Analyzing text %d/%d: %s.', i + 1, ntexts, text)
            values.append((text, self.values_for_text(text, rp)))

        return ResultSet(values)


class ResultSet(collections.OrderedDict):
    """A dictionary structure that represents the values of a set of metrics
    extracted from a text.
    """

    def __getitem__(self, key):
        # If the key is a string, use table/column name.
        if isinstance(key, str):
            for _key, value in self.items():
                if isinstance(_key, Category):
                    if _key.table_name == key:
                        return value
                elif isinstance(_key, Metric):
                    if _key.column_name == key:
                        return value

            raise KeyError(key)

        return super(collections.OrderedDict, self).__getitem__(key)

    @property
    def names(self):
        """Return a list containing the name of each category/metric."""

        return [key.name for key in self.keys()]

    def as_json(self, use_names=True, text_key='title'):
        """Return a JSON representation of this ResultSet."""

        import json
        return json.dumps(self.as_dict(use_names, text_key))

    def as_dict(self, use_names=True, text_key='title'):
        """Return a dictionary representation, with the categories/metrics
        names as keys.

        :use_names: if True, use names as keys; otherwise,
            use table/column names.
        """

        d = collections.OrderedDict()  # was d = {}
        for key, value in self.items():
            if isinstance(key, Text):
                d[key.meta[text_key]] = value.as_dict(use_names, text_key)
            if isinstance(key, Category):
                attr = 'name' if use_names else 'table_name'
                d[getattr(key, attr)] = value.as_dict(use_names, text_key)
                # is_metric_dict = False
            elif isinstance(key, Metric):
                attr = 'name' if use_names else 'column_name'
                d[getattr(key, attr)] = value

        return d

    def as_flat_dict(self, use_names=True, text_key='title'):
        """Return a flat dictionary representation, with the metric values

        :use_names: if True, use names as keys; otherwise,
            use table/column names.
        """

        d = collections.OrderedDict()  # was d = {}
        for category in self.keys():
            for metric in self[category].keys():
                d[metric.column_name] = self[category][metric]
        return d

    def as_table(self):
        """Return a string representation that uses tables to facilitate
        reading.
        """

        from prettytable import PrettyTable

        table = PrettyTable(['Metric', 'Value'])

        table.align['Metric'] = 'r'
        table.align['Value'] = 'r'
        table.padding_width = 1

        def add_lines_for_category(cat, cvalues):
            table.add_row(['', ''])
            table.add_row([cat.name.upper(),
                          '------------------'])
            table.add_row(['', ''])
            for m, mvalue in cvalues.items():
                table.add_row([m.name, mvalue])

        for key, value in self.items():
            if isinstance(key, Text):
                table.add_row(['', ''])
                table.add_row(['~ ' + key.title.upper() + ' ~',
                               '##################'])
                table.add_row(['', ''])

                for k, v in value.items():
                    if isinstance(k, Category):
                        add_lines_for_category(k, v)
                    elif isinstance(k, Metric):
                        table.add_row([k.name, v])
            elif isinstance(key, Category):
                add_lines_for_category(key, value)
            elif isinstance(key, Metric):
                table.add_row([key.name, value])

        return table.get_string()

    def _get_multi_text_arff_data(self, text_key='title'):
        d = self.as_dict(use_names=False, text_key=text_key)

        # TODO: add all metadata that are strings.
        attrs = [(text_key, 'STRING')]

        for cat, cvalues in list(d.values())[0].items():
            for m, mvalue in cvalues.items():
                attrs.append((cat + ':' + m, 'NUMERIC'))

        data = []
        for title, tvalues in d.items():
            curr_data = [('title', '"%s"' % title)]

            for cat, cvalues in tvalues.items():
                for m, mvalue in cvalues.items():
                    curr_data.append((cat + ':' + m, mvalue))

            data.append(curr_data)

        return attrs, data

    def _get_single_text_arff_data(self, text_key='title'):
        d = self.as_dict(use_names=False, text_key=text_key)

        attrs = []
        for cat, cvalues in d.items():
            for m in cvalues.keys():
                attrs.append((cat + ':' + m, 'NUMERIC'))

        if isinstance(list(self.keys())[0], Category):
            data = []
            for cat, cvalues in d.items():
                for m, mvalue in cvalues.items():
                    data.append((cat + ':' + m, mvalue))
        else:
            data = d.items()

        return attrs, [data]

    def as_arff(self, relation_name='corpus', text_key='title', class_attr='class'):
        """Return a string representation in the Attribute-Relation File
        Format (ARFF), suitable for use in tools such as Weka."""

        from datetime import datetime

        if isinstance(list(self.keys())[0], Text):
            attrs, data = self._get_multi_text_arff_data(text_key)

            # Add class information.

            _texts = list(self.keys())

            # Attribute
            class_values = set()
            for _text in _texts:
                class_values.add(_text.meta[class_attr])
            attrs.append((class_attr, '{%s}' % ','.join(class_values)))

            # Values
            for i, datum in enumerate(data):
                datum.append((class_attr, _texts[i].meta[class_attr]))
        else:
            attrs, data = self._get_single_text_arff_data(text_key)

        lines = ['%% File generated by Coh-Metrix-Dementia on %s' %
                 str(datetime.now()),
                 '@RELATION %s' % relation_name,
                 '']

        for attr_name, attr_type in attrs:
            lines.append('@ATTRIBUTE %s %s' % (attr_name, attr_type))

        lines.extend(['',
                      '@DATA'])

        for datum in data:
            lines.append(','.join([str(v) for _, v in datum]))

        return '\n'.join(lines)

    def as_csv(self, text_key='title', class_attr='class'):
        """Return a string representation in CSV format."""

        if isinstance(list(self.keys())[0], Text):
            attrs, data = self._get_multi_text_arff_data(text_key)

            # Add class information.

            _texts = list(self.keys())

            # Attribute
            class_values = set()
            for _text in _texts:
                class_values.add(_text.meta[class_attr])
            attrs.append((class_attr, '{%s}' % ','.join(class_values)))

            # Values
            for i, datum in enumerate(data):
                datum.append((class_attr, _texts[i].meta[class_attr]))
        else:
            attrs, data = self._get_single_text_arff_data(text_key)

        lines = [','.join(attr_name for attr_name, _ in attrs)]

        for datum in data:
            lines.append(','.join([str(v) for _, v in datum]))

        return '\n'.join(lines)

    def as_array(self, text_key='title'):
        """Return a numpy.ndarray representing the data."""

        if isinstance(list(self.keys())[0], Text):
            _, data = self._get_multi_text_arff_data(text_key)
        else:
            _, data = self._get_single_text_arff_data(text_key)

        return np.array([[value for _, value in line[1:]] for line in data],
                        dtype=np.float64)
