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

import nltk

from text_metrics.tools import senter, word_tokenize,\
    pos_tagger, stemmer, parser, dep_parser, univ_pos_tagger, palavras_flat,\
    positive_words, negative_words, simple_words, discourse_markers,\
    ambiguous_discourse_markers, getTemporalExpressions, pronomes_indefinidos,\
    palavras_dificeis, calc_log, conjuncoes_fund1, conjuncoes_fund2,\
    translate, concreteness, load_psicolinguistico, palavras_tree
from text_metrics.tools.lsa import LsaSpace
from text_metrics.tools.lm import KenLmLanguageModel
from text_metrics.utils import is_valid_id, ilen
from text_metrics.database import create_engine, create_session, Helper
from text_metrics.conf import config
from text_metrics.tools.freq_corpora import brwac_frequencies, brasileiro_frequencies

import re
import logging
from itertools import chain
from os.path import basename, isfile, join
from numpy import mean
from functools import reduce
from itertools import filterfalse


logger = logging.getLogger(__name__)


class ResourcePool(object):
    """A resource pool is a repository of methods for producing application
    resources. It centralizes tasks like PoS-tagging and sentence splitting,
    allowing synchronization among threads and use of multiple tools for
    the same task (e.g., taggers). It also allows the creation and reuse of
    database connections and similar resources.
    """

    def __init__(self, cache_limit=300):
        """Form a new resource pool.

        Optional arguments:
        :cache_limit: maximum number of unpinned items in the cache.
        """

        if cache_limit < 0:
            raise ValueError('Invalid cache limit %d. Must be >= 0.' % cache_limit)

        # The resource hooks, in the form {<suffix> : <hook>}.
        self._hooks = {}

        # Resources already asked for, in the form
        # [(<suffix>, <args>, <data>)].

        self._unpinned_cache = []
        self._pinned_cache = []

        self._pinned = set()
        self._cache_limit = cache_limit

    def register(self, suffix, hook, pinned=False):
        """Register a new resource.

        Required arguments:
        :suffix: A string identifying the resource type.
        :hook: The method that, when called, generates the resource data.

        Optional arguments:
        :pinned: True if the resource should be pinned in the cache.

        :returns: None.
        """

        if suffix in self._hooks:
            logger.warning("Resource \"%s\" already registered.", suffix)

        if pinned:
            self._pinned.add(suffix)

        self._hooks[suffix] = hook
        if is_valid_id(suffix):
            setattr(self, suffix, lambda *args: self.get(suffix, *args))

    @staticmethod
    def _get_index(cache, suffix, args):
        """Return the index of the element in the cache,
        or None if not present."""

        for i, elem in enumerate(cache):
            if elem[0] == suffix and elem[1] == args:
                return i
        return None

    def get(self, suffix, *args):
        """Get a resource.

        Required arguments:
        :suffix: The type of the resource to be extracted.
        :args: (Optional) arguments to be passed to the resource's hook.

        :returns: The resource data (as returned by the resource's hook.)
        """

        if suffix not in self._hooks:
            raise ValueError('Resource \"{0}\" not registered.'.format(suffix))

        if suffix in self._pinned:
            index = self._get_index(self._pinned_cache, suffix, args)
            if index is None:
                self._pinned_cache.append((suffix, args, self._hooks[suffix](*args)))
                index = len(self._pinned_cache) - 1

            return self._pinned_cache[index][2]
        else:
            index = self._get_index(self._unpinned_cache, suffix, args)
            if index is None:
                self._unpinned_cache.append((suffix, args, self._hooks[suffix](*args)))
                index = len(self._unpinned_cache) - 1
            value = self._unpinned_cache[index][2]

            if len(self._unpinned_cache) > self._cache_limit:
                del self._unpinned_cache[0]

            return value


class DefaultResourcePool(ResourcePool):
    """A resource pool that uses the standard tools.
    """
    def __init__(self):
        """Registers the default resources."""
        super(DefaultResourcePool, self).__init__()

        # Tools and helpers.
        self.register('pos_tagger', lambda: pos_tagger, pinned=True)
        self.register('univ_pos_tagger', lambda: univ_pos_tagger, pinned=True)
        self.register('parser', lambda: parser, pinned=True)
        self.register('dep_parser', lambda: dep_parser, pinned=True)
        self.register('stemmer', lambda: stemmer, pinned=True)
        self.register('db_helper', self._db_helper, pinned=True)
        self.register('idd3_engine', self._idd3_engine, pinned=True)

        # Basic text info.
        # TODO: these methods need renaming for better readability.
        self.register('raw_content', lambda t: t.raw_content)
        self.register('raw_words', self._raw_words)
        self.register('paragraphs', lambda t: t.paragraphs)
        self.register('sentences', self._sentences)
        self.register('sentence_lengths', self._sentence_lengths)
        self.register('num_clauses', self._num_clauses)
        self.register('tokens', self._tokens)
        self.register('words_in_sents', self._words_in_sents)
        self.register('all_tokens', self._all_tokens)
        self.register('all_words', self._all_words)
        self.register('lower_words', self._lower_words)
        self.register('mattr', self._mattr)
        self.register('mattr_relative', self._mattr_relative)
        self.register('tagged_sentences', self._tagged_sentences)
        self.register('tagged_tokens', self._tagged_tokens)
        self.register('tagged_words', self._tagged_words)
        self.register('tagged_words_in_sents', self._tagged_words_in_sents)

        # Derived text info.
        self.register('content_words', self._content_words)
        self.register('stemmed_content_words', self._stemmed_content_words)
        self.register('content_words_with_tags', self._content_words_with_tags)
        self.register('words_with_tags_in_sents', self._words_with_tags_in_sents)
        self.register('cw_freq', self._cw_freq)
        self.register('cw_freq_brwac', self._cw_freq_brwac)
        self.register('freq_brwac', self._freq_brwac)
        self.register('cw_freq_brasileiro', self._cw_freq_brasileiro)
        self.register('freq_brasileiro', self._freq_brasileiro)
        self.register('token_types', self._token_types)
        self.register('translation', self._translate)

        # Parse structures.
        self.register('parse_trees', self._parse_trees)
        self.register('dep_trees', self._dep_trees)
        self.register('palavras_flat', self._palavras_flat)
        self.register('palavras_tree', self._palavras_tree)

        self.register('toplevel_nps_per_sentence', self._toplevel_nps_per_sentence)
        self.register('leaves_in_toplevel_nps', self._leaves_in_toplevel_nps)

        # LSA spaces
        self.register('lsa_space', self._lsa_space, pinned=True)

        # Language models
        self.register('language_model', self._language_model, pinned=True)

        # Dictionaries
        self.register('positive_words', self._positive_words, pinned=True)
        self.register('negative_words', self._negative_words, pinned=True)
        self.register('simple_words', self._simple_words, pinned=True)
        self.register('discourse_markers',
                      self._discourse_markers, pinned=True)
        self.register('ambiguous_discourse_markers',
                      self._ambiguous_discourse_markers, pinned=True)
        self.register('pronomes_indefinidos',
                      self._pronomes_indefinidos, pinned=True)
        self.register('palavras_dificeis',
                      self._palavras_dificeis, pinned=True)
        self.register('conjuncoes_fund1', self._conjuncoes_fund1, pinned=True)
        self.register('conjuncoes_fund2', self._conjuncoes_fund2, pinned=True)
        self.register('concreteness', self._concreteness, pinned=True)
        self.register('psicolinguistico', self.load_psicolinguistico, pinned=True)
        self.register('brwac_frequencies', self._brwac_frequencies, pinned=True)
        self.register('brasileiro_frequencies', self._brasileiro_frequencies, pinned=True)

        # Temporal expression
        self.register('temporal_expressions', self._temporal_expressions)

        # Calcs
        self.register('log_for_words', self._log_for_words)

    def _db_helper(self):
        """Creates a database session and returns a Helper associated with it.
        """
        engine = create_engine()
        session = create_session(engine)
        helper = Helper(session)

        return helper

    def _idd3_engine(self):
        """Create an IDD3 Engine, configure it, and return it.
        """
        import idd3
        from idd3.rules import pt

        idd3.use_language(pt)

        engine = idd3.Engine(idd3.all_rulesets, idd3.all_transformations)

        return engine

    def _raw_words(self, text):
        """TODO: Docstring for raw_words.

        :text: TODO
        :returns: TODO

        """

        clean_patterns = [re.compile(r'\(\([\w\d\s]*\)\)'),  # Metainfo
                          re.compile(r'\.\.\.'),  # Short pauses
                          re.compile(r'::+'),  # Vowel stretching
                          ]
        split_pattern = re.compile('\s+')

        content = text.raw_content
        for clean_pattern in clean_patterns:
            content = re.sub(clean_pattern, ' ', content)

        content = re.sub(split_pattern, ' ', content)

        return [word for word in content.split(' ') if word]

    def _sentences(self, text):
        """Return a list of strings, each one being a sentence of the text.
        """
        paragraphs = self.get('paragraphs', text)
        sentences = chain.from_iterable(senter.tokenize(p) for p in paragraphs)
        return list(sentences)

    def _sentence_lengths(self, text):
        """Return a list with the lengths, in words, of each sentence of the text.
        """
        sentences = self.get('tagged_sentences', text)
        return [ilen(filterfalse(self.get("pos_tagger").tagset.is_punctuation, s)) for s in sentences]

    def _num_clauses(self, text):
        """
        Return the number of clauses found on text.

        Number of clauses is defined as the number of main verbs: using PALAVRAS,
        it's the words with "V" tags and without "<aux>" tags
        """
        flat = self.get('palavras_flat', text)
        return flat.count(' V ') - flat.count('<aux>')

    def _tokens(self, text):
        """Return a list of lists of strings, where each list of strings
            corresponds to a sentence, and each string in the list is a token.
        """
        sentences = self.get('sentences', text)
        return [word_tokenize(sent) for sent in sentences]

    def _words_in_sents(self, text):
        """Return a list of lists of strings, where each list of strings
            corresponds to a sentence, and each string in the list is a word.
        """
        tagged_sents = self.get('tagged_sentences', text)
        tagset = self.get('pos_tagger').tagset
        words = list(tagged_sents)
        for i in range(len(tagged_sents)):
            words[i] = [word for (word, tag) in tagged_sents[i]
                        if not tagset.is_punctuation((word, tag))]
        return words


    def _all_tokens(self, text):
        """Return all tokens of the text in a single list.
        """
        tokens = self.get('tokens', text)
        return list(chain.from_iterable(tokens))

    def _all_words(self, text):
        """Return all non-punctuation tokens of the text in a single list.
        """
        tagged_words = self.get('tagged_words', text)
        return [word[0] for word in tagged_words]

    def _lower_words(self, text):
        """
        Return all non-punctuation tokens of the text in a single list.

        Tokens are lowered
        """
        return [w.lower() for w in self.get('all_words', text)]

    def _tagged_sentences(self, text):
        """Return a list of lists of pairs (string, string), representing
            the sentences with tagged tokens.
        """
        tokens = self.get('tokens', text)
        return pos_tagger.tag_sents(tokens)

    def _tagged_tokens(self, text):
        """Return a list of pair (string, string), representing the tokens
            not separated in sentences.
        """
        tagged_sentences = self.get('tagged_sentences', text)
        return list(chain.from_iterable(tagged_sentences))

    def _tagged_words(self, text):
        """Return a list of pairs (string, string), representing the
            non-punctuation tokens not separated in sentences.
        """
        tagged_tokens = self.get('tagged_tokens', text)
        tagset = self.get('pos_tagger').tagset
        tagged_words = [token for token in tagged_tokens
                        if not tagset.is_punctuation(token)]
        return tagged_words

    def _tagged_words_in_sents(self, text):
        """Return a list of lists of pairs (string, string),
            representing the non-punctuation tokens separated
            in sentences.
        """

        tagged_tokens = self.get('tagged_tokens', text)
        tagset = self.get('pos_tagger').tagset
        tagged_words = [token for token in tagged_tokens
                        if not tagset.is_punctuation(token)]


        tagged_sents = self.get('tagged_sentences', text)
        tagset = self.get('pos_tagger').tagset
        tagged_words = [[token for token in tagged_sent
                         if not tagset.is_punctuation(token)]
                        for tagged_sent in tagged_sents]
        return tagged_words

    def _content_words(self, text):
        """Return the content words of the text, separated in sentences.

        :text: @todo
        :returns: @todo

        """
        tagged_sents = self.get('tagged_sentences', text)
        content_words = list(tagged_sents)
        for i in range(len(tagged_sents)):
            content_words[i] = [word for (word, tag) in tagged_sents[i]
                                if pos_tagger.tagset.is_content_word(
                                    (word, tag))]
        return content_words

    def _stemmed_content_words(self, text):
        """Return the stem of each content word in the text, separated in
            sentences.

        :text: @todo
        :returns: @todo

        """
        tagged_sents = self.get('tagged_sentences', text)
        tagset = self.get('pos_tagger').tagset
        stemmed_content_words = []
        stemmer = self.get('stemmer')
        for sentence in tagged_sents:

            curr_sentence = []
            for token in sentence:
                if tagset.is_content_word(token):
                    # TODO: add 'tag' to stemmer.get_lemma call after
                    #   tag normalization.
                    lemma = stemmer.get_lemma(token[0])
                    lemma = lemma if lemma else token[0]
                    curr_sentence.append(lemma)

            stemmed_content_words.append(curr_sentence)
            # stemmed_content_words[i] = [stemmer.get_lemma(word)
            #                             for (word, tag) in tagged_sents[i]
            #                             if pos_tagger.tagset.is_content_word(
            #                             (word, tag))]
        return stemmed_content_words

    def _words_with_tags_in_sents(self, text):
        """Return the content words of the text, separated in sentences, but with _tag.
        :text: @todo
        :returns: @todo
        """
        tagged_sents = self.get('tagged_sentences', text)
        tagset = self.get('pos_tagger').tagset
        words = list(tagged_sents)
        for i in range(len(tagged_sents)):
            words[i] = ['%s_%s'%(word,tag) for (word, tag) in tagged_sents[i]
                        if not tagset.is_punctuation((word,tag))]
        return words


    def _content_words_with_tags(self, text):
        """Return the content words of the text, separated in sentences, but with _tag.
        :text: @todo
        :returns: @todo
        """
        tagged_sents = self.get('tagged_sentences', text)
        content_words = list(tagged_sents)
        for i in range(len(tagged_sents)):
            content_words[i] = ['%s_%s'%(word,tag) for (word, tag) in tagged_sents[i]
                                if pos_tagger.tagset.is_content_word(
                                    (word, tag))]
        return content_words


    def _cw_freq(self, text):
        """Return the frequency of each content word in the text, separated
        by sentences.

        :text: @todo
        :returns: @todo

        """
        content_words = self.get('content_words', text)
        db_helper = self.get('db_helper')
        all_words = [word.lower() for sent in content_words for word in sent]
        freq_dict = db_helper.get_frequencies_batch(all_words)

        frequencies = []
        for sentence in content_words:
            sent_freqs = []
            for word in sentence:
                # now its possible to do only 1 query and use the get method on the dict to get the frequencies
                freq_obj = freq_dict.get(word.lower())
                sent_freqs.append(freq_obj.freq if freq_obj is not None else 0)
            frequencies.append(sent_freqs)
        
        return frequencies


    def _freq_brwac(self, text):
        """Return the brWaC corpus frequency of all words separated by sentences.

        :text: @todo
        :returns: @todo
        """
        words = self.get('words_with_tags_in_sents', text)
        brwac_freq = self.get('brwac_frequencies')
        frequencies = list(words)
        for i in range(len(frequencies)):
            frequencies[i] = [brwac_freq[word.lower()] if word.lower() in brwac_freq else 0
                              for word in words[i]]

        # for i in range(len(frequencies)):
        #     for j in range(len(frequencies[i])):
        #         print(words[i][j], ":", frequencies[i][j])

        return frequencies


    def _cw_freq_brwac(self, text):
        """Return the brWaC corpus frequency of each content word separated by sentences.

        :text: @todo
        :returns: @todo
        """
        content_words = self.get('content_words_with_tags', text)
        brwac_freq = self.get('brwac_frequencies')
        frequencies = list(content_words)

        for i in range(len(frequencies)):
            frequencies[i] = [brwac_freq[word.lower()] if word.lower() in brwac_freq else 0
                              for word in content_words[i]]

        return frequencies


    def _freq_brasileiro(self, text):
        """Return the Brasileiro corpus frequency of all words separated by sentences.

        :text: @todo
        :returns: @todo
        """
        words = self.get('words_in_sents', text)
        bra_freq = self.get('brasileiro_frequencies')
        frequencies = list(words)

        for i in range(len(frequencies)):
            frequencies[i] = [bra_freq[word.lower()] if word.lower() in bra_freq else 0
                              for word in words[i]]

        return frequencies


    def _cw_freq_brasileiro(self, text):
        """Return the Brasileiro corpus frequency of each content word separated by sentences.

        :text: @todo
        :returns: @todo
        """
        content_words = self.get('content_words', text)
        bra_freq = self.get('brasileiro_frequencies')
        frequencies = list(content_words)

        for i in range(len(frequencies)):
            frequencies[i] = [bra_freq[word.lower()] if word.lower() in bra_freq else 0
                              for word in content_words[i]]

        return frequencies


    def _token_types(self, text):
        """Return the token types of the text, as a set.

        :text: TODO
        :returns: TODO

        """
        words = [word.lower() for word in self.get('all_words', text)]
        return set(words)

    def _mattr(self, tokens, w=100):
        """Return the Moving Average Type-Token Ratio of a list of tokens.

        :text: TODO
        :returns: TODO

        """

        p = 0
        n = len(tokens)
        wft = dict()
        ttr = []

        if w > n:
            w = n

        for i in range(p, w):
            if tokens[i] in wft:
                wft[tokens[i]] += 1
            else:
                wft[tokens[i]] = 1
        ttr.append(len(wft.keys()) / sum(wft.values()))

        while n - w > p:
            p += 1
            if wft[tokens[p - 1]] == 1:
                wft.pop(tokens[p - 1])
            else:
                wft[tokens[p - 1]] -= 1
            if tokens[p + w - 1] in wft:
                wft[tokens[p + w - 1]] += 1
            else:
                wft[tokens[p + w - 1]] = 1
            ttr.append(len(wft.keys()) / sum(wft.values()))

        return mean(ttr)

    def _mattr_relative(self, tokens, filtered, w=100):
        """Return the Moving Average Type-Token Ratio of a list of filtered
        tokens by a list of tokens.

        :text: TODO
        :returns: TODO

        """

        p = 0
        n = len(tokens)
        wft = dict()
        ttr = []

        if w > n:
            w = n

        for i in range(p, w):
            if tokens[i] in filtered:
                if tokens[i] in wft:
                    wft[tokens[i]] += 1
                else:
                    wft[tokens[i]] = 1
        ttr.append(sum(wft.values()) / w)

        while n - w > p:
            p += 1
            if tokens[p - 1] in wft:
                if wft[tokens[p - 1]] == 1:
                    wft.pop(tokens[p - 1])
                else:
                    wft[tokens[p - 1]] -= 1

            if tokens[p + w - 1] in filtered:
                if tokens[p + w - 1] in wft:
                    wft[tokens[p + w - 1]] += 1
                else:
                    wft[tokens[p + w - 1]] = 1
            ttr.append(sum(wft.values()) / w)

        return mean(ttr)


    def _parse_trees(self, text):
        """Return the parse tree of each sentence in the text.

        :text: TODO
        :returns: TODO
        """
        tokens = self.get('tokens', text)
        sentences = [' '.join(sent) for sent in tokens]
        return parser.parse_sents(sentences)

    def _dep_trees(self, text):
        """Return the dependency tree of each sentence in the text.

        :text: TODO
        :returns: TODO
        """
        sents = self.get('tokens', text)
        return self.get('dep_parser').parse_sents(sents)

    def _toplevel_nps_per_sentence(self, text):
        """
        Returns the NPs that are not contained in any other NP
        in the parse tree for each sentence.

        This depends on the LX-Parser syntax tree.
        
        :rtype: List[List[nltk.Tree]].
        """
        def toplevel_nps(tree):
            """
            Generator over the NPs that are not contained in any
            other NP in the parse tree.
            """
            if tree.label() == 'NP':
                yield tree
            else:
                for child in tree:
                    if isinstance(child, nltk.Tree):
                        yield from toplevel_nps(child)
        parse_trees = self.get('parse_trees', text)
        return [list(toplevel_nps(tree)) for tree in parse_trees]

    def _leaves_in_toplevel_nps(self, text):
        """
        Get the leaves of the toplevel NPs, ignoring all punctuation, for each
        sentence.

        :rtype: List[List[List[str]]].
        """
        def extract_leaves(toplevel):
            """Given a toplevel NP, extract the leaves that are not punctuation"""
            leaves = toplevel.subtrees(lambda t: t.label() != 'PNT' and t.height() == 2)
            return [l[0] for l in leaves]
        return [[extract_leaves(toplevel) for toplevel in toplevels_in_sentence]
                for toplevels_in_sentence in self.get('toplevel_nps_per_sentence', text)]
                

    def _lsa_space(self):
        """Return the default LSA space.

        :returns: an LsaSpace.
        """
        space = LsaSpace(config['LSA_MODEL_PATH'])
        return space

    def _language_model(self):
        """Return the default language model (a 3-gram model
        generated using KenLM).

        :returns: a kenlm.LanguageModel.
        """
        model = KenLmLanguageModel(config['KENLM_LANGUAGE_MODEL'])
        return model

    def _palavras_flat(self, text):
        """Return a PALAVRAS flat tree for a given text.

        :returns: a string.
        """
        treepath = join(config['FLAT_TREES'], basename(text.filepath))
        if text.filepath and isfile(treepath):
            with open(treepath) as fp:
                flat = fp.read()
        else:
            flat = palavras_flat(text)
        return flat

    def _palavras_tree(self, text):
        """Return a PALAVRAS flat tree for a given text.

        :returns: a string.
        """
        return palavras_tree(text)

    def _positive_words(self):
        """Return LIWC's list of positive words.

        :returns: a string.
        """
        pos = positive_words()
        return pos

    def _negative_words(self):
        """Return LIWC's list of negative words.

        :returns: a string.
        """
        neg = negative_words()
        return neg

    def _simple_words(self):
        """Return Biderman's list of simple words.

        :returns: a string.
        """
        sw = simple_words()
        return sw

    def _brwac_frequencies(self):
        """Return list of frequencies from brWaC corpus.

        :returns: a dict.
        """
        bf = brwac_frequencies()
        return bf

    def _brasileiro_frequencies(self):
        """Return list of frequencies from corpus Brasileiro.

        :returns: a dict.
        """
        bf = brasileiro_frequencies()
        return bf

    def _discourse_markers(self):
        """Return a AIC's list of discourse markers.

        :returns: a string.
        """
        dm = discourse_markers()
        return dm

    def _ambiguous_discourse_markers(self):
        """Return a AIC's list of ambiguous discourse markers.

        :returns: a string.
        """
        adm = ambiguous_discourse_markers()
        return adm

    def _temporal_expressions(self, text):
        """Return a PALAVRAS flat tree for a given text.

        :returns: a string.
        """
        return getTemporalExpressions(text.raw_content)

    def _pronomes_indefinidos(self):
        """Return a AIC's list of ambiguous discourse markers.

        :returns: a list.
        """
        indef = pronomes_indefinidos()
        return indef

    def _palavras_dificeis(self):
        """Return dicts of dificult words which frequency if lower than 25,
        50, 100 and 200.

        :returns: a dict.
        """
        dificeis = palavras_dificeis()
        return dificeis

    def _log_for_words(self, word, dic):
        """Return log of the frequency of a word in the difficult
        words dictionary word-freq.

        :returns: a number.
        """
        log = calc_log(word, dic)
        return log

    def _conjuncoes_fund1(self):
        """Return a list of easy conjunctions

        :returns: a list.
        """
        conj = conjuncoes_fund1()
        return conj

    def _conjuncoes_fund2(self):
        """Return a list of easy conjunctions

        :returns: a list.
        """
        conj = conjuncoes_fund2()
        return conj

    def _translate(self, t):
        """Return a english translation of portuguese entry text

        :returns: a list.
        """
        tokens = rp._all_tokens(t)
        chunks = (tokens[0 + i:100 + i] for i in range(0, len(tokens), 100))
        translation = [translate(' '.join(chunk), 'en', 'pt') for chunk in chunks]
        return reduce(lambda x, y: x + ' ' + y, translation)

    def _concreteness(self):
        """Return a dictionary of concreteness ratio of english words

        :returns: a dict.
        """
        return concreteness()

    def load_psicolinguistico(self):
        """Return a dictionary of concreteness ratio of english words

        :returns: a dict.
        """
        return load_psicolinguistico()

rp = DefaultResourcePool()
