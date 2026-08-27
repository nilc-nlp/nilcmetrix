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
from text_metrics import base
from text_metrics.resource_pool import rp as default_rp
from text_metrics.utils import adjacent_pairs, all_pairs
import numpy as np
from text_metrics.tools import senter, word_tokenize
from itertools import chain


class LsaBase(base.Metric):
    """A base class for LSA-derived metrics."""

    def get_pairs(self, t, rp):
        """Return an iterator that yields pair of lists of strings."""

        raise NotImplementedError('Subclasses should override this method')

    def get_value(self, similarities):
        """Given a list of similarities between pairs, return the value of
        the metric.
        """

        raise NotImplementedError('Subclasses should override this method')

    def value_for_text(self, t, rp=default_rp):
        space = rp.lsa_space()
        similarities = []
        for s1, s2 in self.get_pairs(t, rp):
            v1 = space.get_vector(s1)
            v2 = space.get_vector(s2)
            similarities.append(space.compute_similarity(v1, v2))

        if not similarities:
            return 0
        return round(self.get_value(similarities), 5)


class LsaSentenceAdjacentMean(LsaBase):
    """
        ## LSA: média entre sentenças adjacentes

        Média de similaridade entre pares de sentenças adjacentes no texto.

        O espaço LSA utilizado na versão atual do sistema foi gerado a partir
        do mesmo corpus empregado na geração do modelo de língua utilizado
        pela métrica de entropia cruzada: um corpus de 120.813.620
        *tokens*, que consiste na união dos corpora Wikipedia, PLN-BR,
        LácioWeb, e Revista Pesquisa FAPESP.

        ### Exemplo:

        *"Foi o senador Flávio Arns (PT-PR) quem sugeriu a inclusão da peça
            entre os itens do uniforme de alunos dos ensinos Fundamental e
            Médio nas escolas municipais, estaduais e federais. Ele defende a
            medida como forma de proteger crianças e adolescentes dos males
            provocados pelo excesso de exposição aos raios solares. Se a idéia
            for aprovada, os estudantes receberão dois conjuntos anuais,
            completados por calçado, meias, calça e camiseta."*

        O exemplo possui 3 sentenças, e, portanto, 2 pares de sentenças
        adjacentes. A similaridade LSA entre a primeira e a segunda sentenças,
        segundo o modelo utilizado na versão atual do Coh-Metrix-Dementia,
        é 0,084, e a similaridade entre a segunda e a terceira sentenças é
        0,063. Nesse caso, a média entre esses valores é de 0,0735.
    """

    name = 'LSA sentence adjacent mean'
    column_name = 'lsa_adj_mean'

    def get_pairs(self, t, rp):
        tokens = rp.tokens(t)
        tokens = [[token.lower() for token in sentence] for sentence in tokens]

        return adjacent_pairs(tokens)

    def get_value(self, similarities):
        return sum(similarities) / len(similarities) if similarities else 0


class LsaSentenceAdjacentStd(LsaBase):
    """
        ## LSA: desvio padrão entre sentenças adjacentes

        Desvio padrão de similaridade entre pares de sentenças adjacentes no
        texto.

        ### Exemplo:

        *"Foi o senador Flávio Arns (PT-PR) quem sugeriu a inclusão da peça
            entre os itens do uniforme de alunos dos ensinos Fundamental e
            Médio nas escolas municipais, estaduais e federais. Ele defende a
            medida como forma de proteger crianças e adolescentes dos males
            provocados pelo excesso de exposição aos raios solares. Se a idéia
            for aprovada, os estudantes receberão dois conjuntos anuais,
            completados por calçado, meias, calça e camiseta."*

        O exemplo possui 3 sentenças, e, portanto, 2 pares de sentenças
        adjacentes. A similaridade LSA entre a primeira e a segunda sentenças
        é 0,084, e a similaridade entre a segunda e a terceira sentenças é
        0,063. O desvio padrão entre esses valores é de 0,0105.
    """

    name = 'LSA sentence adjacent std'
    column_name = 'lsa_adj_std'

    def get_pairs(self, t, rp):
        tokens = rp.tokens(t)
        tokens = [[token.lower() for token in sentence] for sentence in tokens]

        return adjacent_pairs(tokens)

    def get_value(self, similarities):
        return np.array(similarities).std()


class LsaSentenceAllMean(LsaBase):
    """
        ## LSA: média entre sentenças todos os pares de sentenças

        Média de similaridade entre todos os pares de sentenças no texto.

        ### Exemplo:

        *"Foi o senador Flávio Arns (PT-PR) quem sugeriu a inclusão da peça
            entre os itens do uniforme de alunos dos ensinos Fundamental e
            Médio nas escolas municipais, estaduais e federais. Ele defende a
            medida como forma de proteger crianças e adolescentes dos males
            provocados pelo excesso de exposição aos raios solares. Se a idéia
            for aprovada, os estudantes receberão dois conjuntos anuais,
            completados por calçado, meias, calça e camiseta."*

        O exemplo possui 3 sentenças, e, portanto, 3 pares de sentenças.
        A similaridade LSA entre a primeira e a segunda sentenças
        é 0,084, a similaridade entre a segunda e a terceira sentenças é
        0,063, e a similaridade entre a primeira e a terceira é 0,362.
        A média entre esses valores é 0,17.
    """

    name = 'LSA sentence all mean'
    column_name = 'lsa_all_mean'

    def get_pairs(self, t, rp):
        tokens = rp.tokens(t)
        tokens = [[token.lower() for token in sentence] for sentence in tokens]

        return all_pairs(tokens)

    def get_value(self, similarities):
        return sum(similarities) / len(similarities) if similarities else 0


class LsaSentenceAllStd(LsaBase):
    """
        ## LSA: desvio padrão entre sentenças todos os pares de sentenças

        Desvio padrão de similaridade entre todos os pares de sentenças no
        texto.

        ### Exemplo:

        *"Foi o senador Flávio Arns (PT-PR) quem sugeriu a inclusão da peça
            entre os itens do uniforme de alunos dos ensinos Fundamental e
            Médio nas escolas municipais, estaduais e federais. Ele defende a
            medida como forma de proteger crianças e adolescentes dos males
            provocados pelo excesso de exposição aos raios solares. Se a idéia
            for aprovada, os estudantes receberão dois conjuntos anuais,
            completados por calçado, meias, calça e camiseta."*

        O exemplo possui 3 sentenças, e, portanto, 3 pares de sentenças.
        A similaridade LSA entre a primeira e a segunda sentenças
        é 0,084, a similaridade entre a segunda e a terceira sentenças é
        0,063, e a similaridade entre a primeira e a terceira é 0,362.
        O desvio padrão entre esses valores é 0,14.
    """

    name = 'LSA sentence all (within paragraph) std'
    column_name = 'lsa_all_std'

    def get_pairs(self, t, rp):
        for paragraph in rp.paragraphs(t):
            sentences = senter.tokenize(paragraph)
            tokens = [word_tokenize(sent) for sent in sentences]

            for s1, s2 in all_pairs(tokens):
                yield s1, s2

    def get_value(self, similarities):
        return np.array(similarities).std()


def all_tokens(paragraph):
    """Return all tokens inside a paragraph in a list."""

    sentences = senter.tokenize(paragraph)
    tokens = [word_tokenize(sent) for sent in sentences]

    return list(chain.from_iterable(tokens))


class LsaParagraphAdjacentMean(LsaBase):
    """
        ## LSA: média entre parágrafos adjacentes

        Média de similaridade entre pares de parágrafos adjacentes no texto.
        Esta métrica é calculada do mesmo modo que a média entre sentenças
        adjacentes, mas utilizando-se parágrafos, ao invés de sentenças, como
        unidades.
    """

    name = 'LSA paragraph adjacent mean'
    column_name = 'lsa_paragraph_mean'

    def get_pairs(self, t, rp):
        paragraphs = [all_tokens(par) for par in rp.paragraphs(t)]
        return adjacent_pairs(paragraphs)

    def get_value(self, similarities):
        return sum(similarities) / len(similarities) if similarities else 0


class LsaParagraphAdjacentStd(LsaBase):
    """
        ## LSA: desvio padrão entre parágrafos adjacentes

        Desvio padrão de similaridade entre pares de parágrafos adjacentes no
        texto. Esta métrica é calculada do mesmo modo que o desvio padrão entre
        sentenças adjacentes, mas utilizando-se parágrafos, ao invés de
        sentenças, como unidades.
    """

    name = 'LSA paragraph adjacent std'
    column_name = 'lsa_paragraph_std'

    def get_pairs(self, t, rp):
        paragraphs = [all_tokens(par) for par in rp.paragraphs(t)]
        return adjacent_pairs(paragraphs)

    def get_value(self, similarities):
        return np.array(similarities).std()


class LsaGivennessBase(LsaBase):

    def get_pairs(self, t, rp):
        tokens = rp.tokens(t)
        tokens = [[token.lower() for token in sentence] for sentence in tokens]

        for i in range(1, len(tokens)):
            past_sentences = tokens[:i]
            past_tokens = list(chain.from_iterable(past_sentences))

            yield tokens[i], past_tokens


class LsaGivennessMean(LsaGivennessBase):
    """
        ## LSA: média de *givenness* das sentenças

        Média do *givenness* da cada sentença do texto, a partir da segunda.
        Se o texto possui apenas uma sentença, define-se a métrica como 0,0.
        Define-se o *givenness* de uma sentença como a similaridade LSA entre
        a sentença e todo o texto que a precede.

        ### Exemplo:

        *"Foi o senador Flávio Arns (PT-PR) quem sugeriu a inclusão da peça
            entre os itens do uniforme de alunos dos ensinos Fundamental e
            Médio nas escolas municipais, estaduais e federais. Ele defende a
            medida como forma de proteger crianças e adolescentes dos males
            provocados pelo excesso de exposição aos raios solares. Se a idéia
            for aprovada, os estudantes receberão dois conjuntos anuais,
            completados por calçado, meias, calça e camiseta."*

        O exemplo possui 3 sentenças. A similaridade LSA entre a primeira e a
        segunda sentenças é 0,084, e a similaridade entre a terceira e o
        conjunto formado pela primeira e a segunda é 0,286. A média entre esses
        valores é 0,185.
    """

    name = 'LSA sentence givenness mean'
    column_name = 'lsa_givenness_mean'

    def get_value(self, similarities):
        return sum(similarities) / len(similarities) if similarities else 0


class LsaGivennessStd(LsaGivennessBase):
    """
        ## LSA: desvio padrão de *givenness* das sentenças

        Desvio padrão do *givenness* da cada sentença do texto, a partir da
        segunda. Se o texto possui apenas uma sentença, define-se a métrica
        como 0,0. Define-se o *givenness* de uma sentença como a similaridade
        LSA entre a sentença e todo o texto que a precede.

        ### Exemplo:

        *"Foi o senador Flávio Arns (PT-PR) quem sugeriu a inclusão da peça
            entre os itens do uniforme de alunos dos ensinos Fundamental e
            Médio nas escolas municipais, estaduais e federais. Ele defende a
            medida como forma de proteger crianças e adolescentes dos males
            provocados pelo excesso de exposição aos raios solares. Se a idéia
            for aprovada, os estudantes receberão dois conjuntos anuais,
            completados por calçado, meias, calça e camiseta."*

        O exemplo possui 3 sentenças. A similaridade LSA entre a primeira e a
        segunda sentenças é 0,084, e a similaridade entre a terceira e o
        conjunto formado pela primeira e a segunda é 0,286. O desvio padrão
        entre esses valores é 0,101.
    """

    name = 'LSA sentence givenness std'
    column_name = 'lsa_givenness_std'

    def get_value(self, similarities):
        return np.array(similarities).std()


class LsaSpanBase(base.Metric):
    """Base class for LSA span metrics.

    `lsa_span_mean` and `lsa_span_std` differ only in their aggregation
    over the same span array; the array itself is cached as the
    `lsa_spans` ResourcePool resource so the two metrics share the work.
    """

    def get_value(self, spans):
        """Given the span array, return the value of the metric."""

        raise NotImplementedError('Subclasses should override this method')

    def value_for_text(self, t, rp=default_rp):
        spans = rp.lsa_spans(t)
        if spans is None:
            return 0
        return round(self.get_value(spans), 5)


def compute_lsa_spans(t, rp):
    """Hook for the `lsa_spans` ResourcePool resource.

    Returns the per-sentence LSA span array (cosine of each sentence
    against the LSA projection of all preceding sentences), or `None`
    for texts with fewer than two sentences.
    """
    space = rp.lsa_space()
    num_topics = space.num_topics

    tokens = rp.tokens(t)
    tokens = [[token.lower() for token in sentence] for sentence in tokens]

    if len(tokens) < 2:
        return None

    spans = np.zeros(len(tokens) - 1)
    for i in range(1, len(tokens)):
        past_sentences = tokens[:i]
        span_dim = len(past_sentences)

        if span_dim > num_topics - 1:
            # It's not clear, from the papers I read, what should be done
            # in this case. I did what seemed to not imply in loosing
            # information.
            beginning = past_sentences[0:span_dim - num_topics]
            past_sentences[0] = list(chain.from_iterable(beginning))

        past_vectors = [space.get_vector(sent) for sent in past_sentences]
        curr_vector = space.get_vector(tokens[i])

        # Project curr onto span(past_vectors). Equivalent to applying
        # the full projection matrix A · pinv(AᵀA) · Aᵀ to curr, but
        # without ever forming it: lstsq returns the coefficients, then
        # A @ coef gives the projection directly. Cosine is then a
        # straight dense computation.
        A = np.asarray(past_vectors).T
        coef, *_ = np.linalg.lstsq(A, curr_vector, rcond=None)
        projection = A @ coef

        n1 = np.linalg.norm(curr_vector)
        n2 = np.linalg.norm(projection)
        spans[i - 1] = float(curr_vector @ projection / (n1 * n2)) if n1 and n2 else 0.0

    return spans


class LsaSpanMean(LsaSpanBase):
    """
        ## LSA: média do *span* das sentenças

        Média do *span* da cada sentença do texto, a partir da segunda.
        Se o texto possui apenas uma sentença, define-se a métrica como 0,0.
        O *span* de uma sentença, assim como o *givenness*, é uma forma de
        medir a proximidade entre uma sentença e o contexto que a precede.
        A diferença, em termos simples, consiste no fato de que o *span*
        procura capturar a similaridade não apenas com o conteúdo explícito
        apresentado anteriormente no texto, mas também com tudo o que se pode
        inferir com base nesse conteúdo.

        ### Exemplo:

        *"Foi o senador Flávio Arns (PT-PR) quem sugeriu a inclusão da peça
            entre os itens do uniforme de alunos dos ensinos Fundamental e
            Médio nas escolas municipais, estaduais e federais. Ele defende a
            medida como forma de proteger crianças e adolescentes dos males
            provocados pelo excesso de exposição aos raios solares. Se a idéia
            for aprovada, os estudantes receberão dois conjuntos anuais,
            completados por calçado, meias, calça e camiseta."*

        O exemplo possui 3 sentenças. O *span* LSA entre a primeira e a
        segunda sentenças, segundo o modelo utilizado na versão atual do
        Coh-Metrix-Dementia, é 0,084, e o *span* entre a terceira e o
        conjunto formado pela primeira e a segunda é 0,223. A média desses
        valores é 0,1535.
    """

    name = 'LSA sentence span mean'
    column_name = 'lsa_span_mean'

    def get_value(self, spans):
        return spans.mean()


class LsaSpanStd(LsaSpanBase):
    """
        ## LSA: desvio padrão do *span* das sentenças

        Desvio padrão do *span* da cada sentença do texto, a partir da segunda.
        Se o texto possui apenas uma sentença, define-se a métrica como 0,0.

        ### Exemplo:

        *"Foi o senador Flávio Arns (PT-PR) quem sugeriu a inclusão da peça
            entre os itens do uniforme de alunos dos ensinos Fundamental e
            Médio nas escolas municipais, estaduais e federais. Ele defende a
            medida como forma de proteger crianças e adolescentes dos males
            provocados pelo excesso de exposição aos raios solares. Se a idéia
            for aprovada, os estudantes receberão dois conjuntos anuais,
            completados por calçado, meias, calça e camiseta."*

        O exemplo possui 3 sentenças. O *span* LSA entre a primeira e a
        segunda sentenças é 0,084, e o *span* entre a terceira e o
        conjunto formado pela primeira e a segunda é 0,223. O desvio padrão
        desses valores é 0,070.
    """

    name = 'LSA sentence span std'
    column_name = 'lsa_span_std'

    def get_value(self, spans):
        return spans.std()


class Lsa(base.Category):
    name = 'Latent Semantic Analysis'
    table_name = 'lsa'

    def __init__(self):
        super(Lsa, self).__init__()
        self.metrics = [LsaSentenceAdjacentMean(),
                        LsaSentenceAdjacentStd(),
                        LsaSentenceAllMean(),
                        LsaSentenceAllStd(),
                        LsaParagraphAdjacentMean(),
                        LsaParagraphAdjacentStd(),
                        LsaGivennessMean(),
                        LsaGivennessStd(),
                        LsaSpanMean(),
                        LsaSpanStd(),
                        ]
