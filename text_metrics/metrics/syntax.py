# -*- coding: utf-8 -*-
# Coh-Metrix-Dementia - Automatic text analysis and classification for dementia.
# Copyright (C) 2014-2015  Andre Luiz Verucci da Cunha
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

from __future__ import division, print_function, unicode_literals

from nltk.util import trigrams

from text_metrics import base
from text_metrics.resource_pool import rp as default_rp
from text_metrics.utils import reverse_tree


class YngveComplexity(base.Metric):
    """
        **Nome da métrica**: yngve

        **Interpretação**: Quanto maior o valor, maior a complexidade do texto

        **Descrição da métrica**: A complexidade de Yngve baseia-se na premissa de que as árvores
        sintáticas das sentenças da língua inglesa tendem a se ramificar para a direita, e que desvios
        em relação a esse padrão correspondem a uma maior complexidade na linguagem. Dessa
        forma, a complexidade de Yngve procura medir o quanto uma árvore sintática se desvia desse
        padrão de ramificação. Se a forma de calcular a pontuação de cada nó envolver uma pilha, isto
        é, a partir de uma pilha utilizada em uma derivação de cima para baixo, da esquerda para a
        direita, o escore de uma palavra é dado pelo número de elementos que ainda permanecem na
        pilha quando a palavra é finalmente derivada. Alguns trabalhos da literatura, relacionaram o
        tamanho da pilha necessário para processar uma sentença à sua demanda de memória
        operacional, apesar de ele medir diretamente apenas o desvio de uma ramificação à direita.

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: Usando a árvore sintática de constituinte, atribuímos um peso
        para cada nó não-terminal da árvore, da seguinte maneira:

        1. Para cada nó, atribuir peso 0 ao seu filho mais à direita, 1 ao segundo filho mais a
        direita, e assim por diante até chegar ao filho mais a esquerda. Dessa forma, os filhos
        serão numerados com peso 0, 1, 2, ... , da direita para a esquerda.

        2. Em seguida, calcula-se a pontuação de cada palavra, dada pela soma dos pesos dos
        não-terminais existentes no caminho entre a raiz e a palavra.

        3. Uma vez calculada a pontuação das palavras, a complexidade da sentença é calculada
        como a média dos valores encontrados.

        **Recursos de PLN utilizados durante o cálculo**: parser sintático LX-Parser e tokenizador NLTK

        **Limitações da métrica**: a métrica depende da precisão do parser e do tokenizador.

        **Crítica**: Se o parser errar na construção da árvore, necessariamente isso afetará a métrica.

        Quanto maior o tamanho e a complexidade da sentença, maior a probabilidade de o parser

        apresentar erros na construção da árvore de dependências.

        **Projeto**: CMD

        **Teste**: Os brasileiros esperam resultados concretos.

        **Resultado Obtido**: 1,667

        **Status**: dada uma árvore de constituintes construída, o cálculo foi conferido e está correto
    """

    name = 'Yngve Complexity'
    column_name = 'yngve'

    def value_for_text(self, t, rp=default_rp):
        syntax_trees = rp.parse_trees(t)

        sentence_indices = []
        for tree in syntax_trees:
            reverse_tree(tree)

            leaves = tree.leaves()

            word_indices = []
            for i in range(len(leaves)):
                word_indices.append(sum(tree.leaf_treeposition(i)))

            reverse_tree(tree)

            sentence_indices.append((sum(word_indices) / len(word_indices)))

        return sum(sentence_indices) / len(sentence_indices) \
                if sentence_indices else 0


class FrazierComplexity(base.Metric):
    """
        **Nome da métrica**: frazier

        **Interpretação**: Quanto maior o valor, maior a complexidade do texto

        **Descrição da métrica**: Frazier propôs uma abordagem bottom-up para o cálculo da
        complexidade sintática de uma sentença, que parte da palavra e sobe na árvore sintática até
        encontrar um nó que não seja o filho mais à esquerda de seu pai. Cada nó na árvore recebe
        uma pontuação 1, e nós filhos de nós do tipo sentença, 1.5. A pontuação de cada palavra é
        dada pela soma das pontuações dos nós pertencentes a seu ramo.

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: Para calcular a complexidade de uma sentença, é preciso que se
        tenha sua árvore sintática. Por exemplo: considere a oração em inglês "She found a cat with a
        red tail.". Suponha que cada nó tenha um peso. Um peso marcado com [x] indica fim de ramo.
        Nesse exemplo, para a derivação da palavra "She", o nó imediatamente acima teria uma
        etiqueta morfossintática PRP (pronome). Esse nó é incluído na contagem, pois é o filho mais à
        esquerda de NP, recebendo pontuação 1; PRP é filho de NP, que é o filho mais à esquerda de
        seu pai, e como seu pai é um nó do tipo sentença, recebe pontuação 1.5; portanto, a
        pontuação de "She" é 1 + 1.5 = 2.5. Para o cálculo da pontuação de "found", seu nó pai VBD
        (verbo direto) é incluído no cálculo, pois é o filho mais à esquerda de VP, recebendo pontuação
        1; porém, o pai de VBD, VP, não é incluído, pois não é o filho mais à esquerda de S; portanto, a
        derivação para, e a pontuação de "found" é 1. No caso de "cat", nem mesmo seu nó pai NN é
        incluído, pois não é o filho mais à esquerda de NP, recebendo pontuação 0.

        Frazier propôs dividir a sentença em trigramas para fazer o cálculo. Para calcular a
        complexidade da sentença, calcula-se a soma das pontuações das palavras em cada trigrama,
        usando o máximo dessas somas numa varredura da esquerda pra direita. A complexidade de
        um texto é a média da complexidade de Frazier para cada sentença.

        **Recursos de PLN utilizados durante o cálculo**: parser sintático LX-Parser e tokenizador NLTK
        **Limitações da métrica**: a métrica depende da precisão do parser e do tokenizador.

        **Crítica**: Se o parser errar na construção da árvore, necessariamente isso afetará a métrica.
        Quanto maior o tamanho e a complexidade da sentença, maior a probabilidade de o parser
        apresentar erros na construção da árvore de dependências.

        **Projeto**: Coh-Metrix-Dementia

        **Teste**: Os brasileiros esperam resultados concretos.

        **Resultado Obtido**: 5,0

        **Status**: dada uma árvore de constituintes construída, o cálculo foi conferido e está correto
    """

    name = 'Frazier Complexity'
    column_name = 'frazier'

    def value_for_text(self, t, rp=default_rp):
        syntax_trees = rp.parse_trees(t)

        sentence_indices = []
        for tree in syntax_trees:
            if tree.label() == 'ROOT':
                tree = tree[0]

            leaves = tree.leaves()

            word_indices = [0] * len(leaves)
            for i in range(len(leaves)):
                ref_vector = tree.leaf_treeposition(i)

                j = -2
                while j >= -len(ref_vector) and ref_vector[j] == 0:
                    parent_index = len(ref_vector) + j
                    parent_node = tree[ref_vector[:parent_index]]

                    if rp.parser().tagset.is_sentence_node(parent_node):
                        word_indices[i] += 1.5
                    else:
                        word_indices[i] += 1

                    j -= 1

            if len(leaves) < 3:
                sentence_index = sum(word_indices)
            else:
                max_trigrams = 0
                for trigram in trigrams(word_indices):
                    if sum(trigram) > max_trigrams:
                        max_trigrams = sum(trigram)
                sentence_index = max_trigrams

            sentence_indices.append(sentence_index)

        return sum(sentence_indices) / len(sentence_indices) \
                if sentence_indices else 0


class DependencyDistance(base.Metric):
    """
        **Nome da métrica**: dep_distance

        **Interpretação**: Quanto maiores as distâncias de dependência, maior a complexidade do texto

        **Descrição da métrica**: A distância de dependência utiliza uma árvore de dependências para
        realizar o cálculo. A cada relação de dependência está associada uma distância entre as
        palavras na superfícies textual.

        Estudos da literatura mostram que essas distâncias entre palavras nas relações de
        dependência são diretamente proporcionais ao tempo de processamento em tarefas de
        compreensão de sentenças; grandes distâncias entre palavras relacionadas geram overhead de
        memória.

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: A métrica é calculada como a soma das distâncias associadas às

        relações de dependências. O valor final para um texto é a média dos valores de complexidade

        para cada sentença.

        **Recursos de PLN utilizados durante o cálculo**: parser de dependências MaltParser,

        tokenizador NLTK

        **Limitações da métrica**: a precisão da métrica depende do desempenho do tokenizador e do
        parser

        **Crítica**: Se o parser errar na construção da árvore, necessariamente isso afetará a métrica.
        Quanto maior o tamanho e a complexidade da sentença, maior a probabilidade de o parser
        apresentar erros na construção da árvore de dependências.

        **Projeto**: Coh-Metrix-Dementia

        **Teste 1**: Ela encontrou um gato com um rabo vermelho.

        **Resultado Obtido**: 9,0

        **Status**: dada uma árvore de dependências construída, o cálculo foi conferido e está correto
    """

    name = 'Dependency Distance'
    column_name = 'dep_distance'

    def value_for_text(self, t, rp=default_rp):
        graphs = rp.dep_trees(t)

        dep_distances = []
        for graph in graphs:
            dep_distance = 0
            for dep in graph.nodes.values():
                if dep['rel'] not in ('TOP', 'ROOT', 'p', 'root', 'punct', None):
                    dep_distance += abs(dep['address'] - dep['head'])
                    # The 'ROOT' and 'p' tags are the ones returned by
                    # MaltParser, not 'root' and 'punct', but they stay to
                    # avoid breaking anything
            dep_distances.append(dep_distance)

        return sum(dep_distances) / len(dep_distances) \
                if dep_distances else 0


class CrossEntropy(base.Metric):
    """
        ## Entropia Cruzada

        Descrição.

        ### Exemplo:

        *Exemplo.*

        Descrição Exemplo."""

    name = 'Cross Entropy'
    column_name = 'cross_entropy'

    def value_for_text(self, t, rp=default_rp):
        lm = rp.language_model()

        sents = [lm.clean(sent) for sent in rp.sentences(t)]
        # Remove empty sentences before calculating the score. Avoid division by zero.
        sents = [sent for sent in sents if sent]
        scores = [-1/len(sent) * lm.score(sent) for sent in sents]

        return sum(scores) / len(scores) if scores else 0


class SyntacticalComplexity(base.Category):
    """Docstring for SyntacticalComplexity. """

    name = 'Syntactical Complexity'
    table_name = 'syntax'

    def __init__(self):
        super(SyntacticalComplexity, self).__init__()
        self._set_metrics_from_module(__name__)
        self.metrics.sort(key=lambda m: m.name)
