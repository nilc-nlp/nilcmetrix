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
from sqlalchemy import create_engine as _create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Boolean
from sqlalchemy.orm import sessionmaker


Base = declarative_base()

DEFAULT_OPTIONS = {
    'dialect': 'postgresql',
    'driver': 'psycopg2',
    'username': 'cohmetrix',
    'password': 'cohmetrix',
    'host': 'pgs_cohmetrix',
    'port': '5432',
    'database': 'cohmetrix',
}


def create_engine(options=DEFAULT_OPTIONS, echo=False):
    connect_string =\
        '{dialect}+{driver}://{username}:{password}@{host}:{port}/{database}'\
        .format(**options)
    return _create_engine(connect_string, echo=echo)


def create_session(engine):
    return sessionmaker(bind=engine)()


class DelafVerb(Base):
    __tablename__ = 'delaf_verbs'

    word = Column(String, primary_key=True)
    lemma = Column(String, primary_key=True)
    pos = Column(String, primary_key=True)
    tense = Column(String, primary_key=True)
    person = Column(String, primary_key=True)

    def __repr__(self):
        return ('<DelafVerb: word={0}, lemma={1}, pos={2}, tense={3},' +
                ' person={4}>')\
            .format(self.word, self.lemma, self.pos, self.tense, self.person)


class DelafNoun(Base):
    __tablename__ = 'delaf_nouns'

    word = Column(String, primary_key=True)
    lemma = Column(String, primary_key=True)
    pos = Column(String, primary_key=True)
    morf = Column(String, primary_key=True)

    def __repr__(self):
        return '<DelafNoun: word={0}, lemma={1}, pos={2}, morf={3}>'\
            .format(self.word, self.lemma, self.pos, self.morf)


class DelafWord(Base):
    __tablename__ = 'delaf_words'

    word = Column(String, primary_key=True)
    lemma = Column(String, primary_key=True)
    pos = Column(String, primary_key=True)

    def __repr__(self):
        return '<DelafWord: word={0}, lemma={1}, pos={2}>'\
            .format(self.word, self.lemma, self.pos)


class TepWord(Base):
    __tablename__ = 'tep_words'

    group = Column(Integer, primary_key=True)
    word = Column(String, primary_key=True)
    pos = Column(String)
    antonym = Column(Integer)

    def __repr__(self):
        return '<TepWord: group={0}, word={1}, pos={2}, antonym={3}>'\
            .format(self.group, self.word, self.pos, self.antonym)


class Frequency(Base):
    __tablename__ = 'frequencies'

    id = Column(Integer, primary_key=True)
    word = Column(String)
    freq = Column(Integer)
    freq_perc = Column(Float)
    texts = Column(Integer)
    texts_perc = Column(Float)

    def __repr__(self):
        return '<Frequency: word=%s, freq=%s, freq_perc=%s, texts=%s, texts_perc=%s>'\
            % (self.word, str(self.freq), str(self.freq_perc), str(self.texts),
               str(self.texts_perc))


class Hypernym(Base):
    __tablename__ = 'hypernyms_verbs'

    word = Column(String, primary_key=True)
    category = Column(String, primary_key=True)
    grammar_attrs = Column(String)
    hyper_levels = Column(Integer)

    def __repr__(self):
        return '<Hypernym: word={0}, cat={1}, attrs={2}, levels={3}>'\
            .format(self.word, self.category, self.grammar_attrs,
                    self.hyper_levels)


class Connective(Base):
    __tablename__ = 'connectives'

    connective = Column(String, primary_key=True)
    additive_pos = Column(Boolean)
    additive_neg = Column(Boolean)
    temporal_pos = Column(Boolean)
    temporal_neg = Column(Boolean)
    causal_pos = Column(Boolean)
    causal_neg = Column(Boolean)
    logic_pos = Column(Boolean)
    logic_neg = Column(Boolean)

    def __repr__(self):
        attrs = []
        if self.additive_pos:
            attrs.append('add pos')
        if self.additive_neg:
            attrs.append('add neg')
        if self.temporal_pos:
            attrs.append('tmp pos')
        if self.temporal_neg:
            attrs.append('tmp neg')
        if self.causal_pos:
            attrs.append('cau pos')
        if self.causal_neg:
            attrs.append('cau neg')
        if self.logic_pos:
            attrs.append('log pos')
        if self.logic_neg:
            attrs.append('log neg')

        return '<Connective: conn={0}, {1}>'.format(self.connective,
                                                    ', '.join(attrs))


class Helper(object):

    def __init__(self, session):
        """@todo: Docstring for __init__.

        :session: @todo
        :returns: @todo

        """
        self._session = session

    def get_frequency(self, word):
        return self._session.query(Frequency).filter_by(word=word).first()

    def get_frequencies_batch(self, words):
        """Get frequencies for multiple words in a single query.

        :words: list of words to query
        :returns: dict mapping word -> Frequency object (or None if not found)
        """
        if not words:
            return {}
        # remove duplicates
        unique_words = list(set(w.lower() for w in words))
        # single db lookup
        results = self._session.query(Frequency).filter(
            Frequency.word.in_(unique_words)
        ).all()
        # build final dict
        return {f.word: f for f in results}

    def get_hypernyms(self, verb):
        """@todo: Docstring for get_hypernyms.

        :verb: @todo
        :returns: @todo

        """
        return self._session.query(Hypernym).filter_by(word=verb).first()

    def get_delaf_verb(self, verb):
        """@todo: Docstring for get_verb.

        :verb: @todo
        :returns: @todo

        """
        return self._session.query(DelafVerb).filter_by(word=verb).first()

    def get_delaf_noun(self, noun):
        """@todo: Docstring for get_noun.

        :noun: @todo
        :returns: @todo

        """
        return self._session.query(DelafNoun).filter_by(word=noun).first()

    def get_delaf_word(self, word, pos=None):
        """@todo: Docstring for get_word.

        :word: @todo
        :pos: @todo
        :returns: @todo

        """
        if pos is None:
            # Ignore PoS
            result = self._session.query(DelafWord).filter_by(word=word).first()
        else:
            result = self._session.query(DelafWord)\
                .filter_by(word=word, pos=pos).first()

        return result

    def get_tep_word(self, word, pos=None):
        """@todo: Docstring for get_tep_word.

        :word: @todo
        :pos: @todo
        :returns: @todo

        """
        if pos is None:
            # Ignore PoS
            result = self._session.query(TepWord).filter_by(word=word).first()
        else:
            result = self._session.query(TepWord)\
                .filter_by(word=word, pos=pos).first()

        return result

    def get_all_tep_words(self, word, pos=None):
        """@todo: Docstring for get_all_tep_words.

        :word: @todo
        :pos: @todo
        :returns: @todo

        """
        if pos is None:
            # Ignore PoS
            result = self._session.query(TepWord).filter_by(word=word).all()
        else:
            result = self._session.query(TepWord)\
                .filter_by(word=word, pos=pos).all()

        return result

    def get_tep_words_count(self, word, pos=None):
        """@todo: Docstring for get_tep_words_count.

        :word: @todo
        :pos: @todo
        :returns: @todo

        """
        if pos is None:
            # Ignore PoS
            result = self._session.query(TepWord).filter_by(word=word).count()
        else:
            result = self._session.query(TepWord)\
                .filter_by(word=word, pos=pos).count()

        return result

    def get_connective(self, connective):
        """TODO: Docstring for get_connective.

        :connective: TODO
        :returns: TODO

        """
        return self._session.query(Connective).filter_by(connective=connective)\
            .first()

    def get_all_connectives(self):
        """TODO: Docstring for get_connective.

        :connective: TODO
        :returns: TODO

        """
        return self._session.query(Connective).all()


if __name__ == '__main__':
    engine = create_engine()
    session = create_session(engine)
    helper = Helper(session)

    print(helper.get_frequency('abacaxi'))
    print(helper.get_frequency('maçã'))
    print(helper.get_hypernyms('dar'))
    print(helper.get_hypernyms('abalançar'))
    print(helper.get_delaf_verb('apareceu'))
    print(helper.get_delaf_verb('abraçarão'))
    print(helper.get_delaf_noun('abraço'))
    print(helper.get_delaf_noun('carrinho'))
    print(helper.get_delaf_noun('carrão'))
    print(helper.get_delaf_word('bonito'))
    print(helper.get_delaf_word('finalmente'))
    print(helper.get_delaf_word('canto', pos='N'))
    print(helper.get_delaf_word('canto', pos='V'))
    print(helper.get_tep_word('cantar', pos='Substantivo'))
    print(helper.get_tep_word('cantar', pos='Verbo'))
    print(helper.get_all_tep_words('cantar'))
    print(helper.get_tep_words_count('cantar'))
    print(helper.get_all_tep_words('cantar', pos='Verbo'))
    print(helper.get_tep_words_count('cantar', pos='Verbo'))
    print(helper.get_connective('na realidade'))
    print(helper.get_connective('além disso'))
    # print(helper.get_all_connectives())
