# -*- coding: utf-8 -*-
# Copyright (C) 2015  Nathan Siegle Hartmann
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
from text_metrics.tools import pos_tagger
import re


# class SubordinateClausesIntroducedByConjunctions(base.Metric):
#     """
#         **Nome da Métrica**: subordinate_clauses_per_clauses
#
#         **Interpretação**: quanto maior a proporção, maior a complexidade
#
#         **Descrição da métrica**: Proporção de orações iniciadas por conjunções subordinativas
#
#         **Definição dos termos que aparecem na descrição da métrica**: orações subordinadas são orações introduzidas por
#         uma conjunção subordinativa, por pronomes relativos ou são reduzidas de infinitivo, particípio e gerúndio.
#
#         **Forma de cálculo da métrica**: para descobrir a quantidade total de orações do texto, contam-se todos os
#         verbos e subtraem-se os verbos auxiliares, resultando na quantidade de verbos principais, cada um referente a
#         uma oração. Para descobrir a quantidade de orações subordinadas introduzidas por conjunções subordinativas,
#         contam-se as ocorrências da etiqueta KS do parser Palavras e divide-se o resultado pelo total de orações do
#         texto.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras.
#
#         **Limitações da métrica**: a precisão dos valores depende diretamente do desempenho do parser Palavras e da
#         lista de conjunções subordinativas do parser
#
#         **Crítica**: Esta métrica pretendia capturar todas as orações subordinadas, mas a etiqueta KS captura apenas
#         parte delas. Não estão sendo computadas as subordinadas relativas e as reduzidas de gerúndio, particípio e
#         infinitivo. Por esse motivo, esta métrica deve ser comentada e substituída pela métrica “subordinate_clauses”,
#         do programa guten.py.
#
#         **Projeto**: AIC
#     """
#
#     name = 'Ratio of Subordinate Clauses introduced by Conjunctions per Clauses'
#     column_name = 'subordinate_clauses_per_clauses'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         verbs = flat.count(' V ') - flat.count('<aux>')
#         subordinate_clauses = flat.count(' KS ')
#         try:
#             return subordinate_clauses / verbs
#         except ZeroDivisionError:
#             return 0


class CoordinateClausesPerClauses(base.Metric):
    """
        **Nome da Métrica**: coordinate_conjunctions_per_clauses

        **Interpretação**: as conjunções coordenativas unem orações, palavras e outros constituintes sintáticos. Não
        está clara a sua relação com o nível de complexidade do texto.

        **Descrição da métrica**: proporção de conjunções coordenativas em relação ao total de orações do texto

        **Definição dos termos que aparecem na descrição da métrica**: conjunções coordenativas são uma classe de
        palavras que unem duas orações independentes ou duas palavras de conteúdo ou dois sintagmas nominais. Ex:
        “Inovou e ficou rico”, “João e Maria”, “bonito, porém caro”, “agora e sempre”, “bonitinha, mas ordinária”.

        **Forma de cálculo da métrica**: Divide-se a quantidade de conjunções coordenativas (etiqueta KC do parser
        Palavras) pela quantidade total de orações do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras para identificar as conjunções.

        **Limitações da métrica**:

        **Crítica**: originalmente essa métrica pretendia capturar as orações coordenadas, porém a heurística se mostrou
        falha por dois motivos: 1) há orações coordenadas não introduzidas por conjunções (chamadas assindéticas); 2) a
        etiqueta KC não é exclusiva das conjunções que ligam orações (coordenadas sindéticas). Outros algoritmos foram
        testados para essa finalidade, inclusive com auxílio do criador do parser Palavras, Eckhard Bick, mas não se
        mostraram eficientes.

        **Projeto**: AIC

        **Teste**: O cientista político André Marenco afirma que a Justiça Eleitoral tem sido rigorosa em relação à
        propaganda antes do prazo legal e que a verticalização impôs aos partidos a conciliação de nacionais e regionais.

        **Contagens**: 3 orações (afirma, tem sido, impôs), 2 conjunções coordenativas (e, e)

        **Resultado Esperado**: 0,667

        **Resultado Obtido**: 0,667

        **Status**: correto
    """

    name = 'Ratio of Coordinate Conjunctions per Clauses'
    column_name = 'coordinate_conjunctions_per_clauses'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        clauses = rp.num_clauses(t)
        coordinate_conjunctions = flat.count(' KC ')
        try:
            return coordinate_conjunctions / clauses
        except ZeroDivisionError:
            return 0


class SentencesWithZeroClause(base.Metric):
    """
        **Nome da Métrica**: sentences_with_zero_clause

        **Interpretação**: quanto maior a proporção, menor a complexidade

        **Descrição da métrica**: quantidade de sentenças sem nenhum verbo, ou seja, frases e não orações

        **Definição dos termos que aparecem na descrição da métrica**: sentença sem oração é aquela que não apresenta
        nenhum verbo

        **Forma de cálculo da métrica**: para obter o total de sentenças, utiliza-se o sentenciador do parser; para
        obter a quantidade de sentenças sem orações (sem verbos), verifica-se em cada sentença a quantidade de verbos
        e conta-se as ocorrências de 0 verbos; depois divide-se a quantidade de sentenças sem verbos pela quantidade
        total de sentenças do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser

        **Limitações da métrica**:  a precisão da métrica depende do desempenho do parser. O parser Palavras gera uma
        quebra de sentença sempre que a sentença é longa, utilizando como ponto de quebra a ocorrência de dois-pontos
        ou ponto-e-vírgula. Esse fato pode levar à superestimativa de quantidade de sentenças total do texto e à
        ocorrência artificial de sentenças sem verbo (vide teste abaixo).

        **Crítica**:

        **Projeto**: AIC

        **Teste**: A retirada de chapéus e bonés da cabeça em ambientes fechados.

        **Contagens**: nenhuma oração (observa-se, contudo, que essa sentença é artificial, gerada por um recurso do
        parser de quebrar orações longas em duas partes, usando como divisor os dois pontos. A sentença original era:
        “Professores ainda temem que a distribuição do acessório dificulte a imposição, nas classes, de uma regra da
        boa conduta: a retirada de chapéus e bonés da cabeça em ambientes fechados.”)

        **Resultado Esperado**: 1

        **Resultado Obtido**: 1

        **Status**: correto
    """

    name = 'Ratio of Sentences With Zero Clause'
    column_name = 'sentences_with_zero_clause'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        sentences = flat.split('</s>')[:-2]
        occurences = 0
        for sentence in sentences:
            if sentence.count(' V ') - sentence.count('<aux>') == 0:
                occurences += 1
        try:
            return occurences / len(sentences)
        except ZeroDivisionError:
            return 0


class SentencesWithOneClause(base.Metric):
    """
        **Nome da Métrica**: sentences_with_one_clause

        **Interpretação**: quanto maior a proporção, menor a complexidade

        **Descrição da métrica**: Proporção de sentenças que contenham apenas 1 oração

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: para obter o total de sentenças, utiliza-se o sentenciador do parser; para
        obter a quantidade de sentenças com 1 única oração (com 1 verbo principal), verifica-se em cada sentença a
        quantidade de verbos, subtraem-se os verbos auxiliares e contam-se os resultados que derem 1; depois divide-se
        a quantidade de sentenças com 1 verbo pela quantidade total de sentenças do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser

        **Limitações da métrica**:  a precisão da métrica depende do desempenho do parser. O parser Palavras gera uma
        quebra de sentença sempre que a sentença é longa, utilizando como ponto de quebra a ocorrência de dois-pontos
        ou ponto-e-vírgula. Esse fato pode levar à superestimativa de quantidade de sentenças e à ocorrência artificial
        de sentenças com apenas 1 verbo.

        **Crítica**:

        **Projeto**: AIC

        **Teste**: Em 29 de maio de 2002, Antônio Britto (PPS) cumpria agenda no Vale do Sinos.

        **Contagens**: 1 verbo, 1 oração: cumpria

        **Resultado Esperado**: 1

        **Resultado Obtido**: 1

        **Status**: correto
    """

    name = 'Ratio of Sentences With One Clause'
    column_name = 'sentences_with_one_clause'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        sentences = flat.split('</s>')[:-2]
        occurrences = 0
        for sentence in sentences:
            if sentence.count(' V ') - sentence.count('<aux>') == 1:
                occurrences += 1
        try:
            return occurrences / len(sentences)
        except ZeroDivisionError:
            return 0


class SentencesWithTwoClause(base.Metric):
    """
        **Nome da Métrica**: sentences_with_two_clauses

        **Interpretação**: quanto maior a proporção, maior a complexidade

        **Descrição da métrica**: Proporção de sentenças que contenham 2 orações

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: para obter o total de sentenças, utiliza-se o sentenciador do parser; para
        obter a quantidade de sentenças com 2 orações (com 2 verbos principais), verifica-se em cada sentença a
        quantidade de verbos, subtraem-se os verbos auxiliares  e contam-se os resultados que derem 2; depois divide-se
        a quantidade de sentenças com 2 verbos pela quantidade total de sentenças do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser

        **Limitações da métrica**: a precisão da métrica depende do desempenho do parser. O parser Palavras gera uma
        quebra de sentença sempre que a sentença é longa, utilizando como ponto de quebra a ocorrência de dois-pontos
        ou ponto-e-vírgula. Esse fato pode levar o sistema a superestimar a quantidade total de sentenças, bem como a
        subestimar ou superestimar a quantidade de sentenças com 2 verbos (se a sentença original tiver 2 verbos, a
        quebra gera uma subestimativa; se a sentença original tiver 3 ou mais verbos e a quebra gerar pelo menos 1
        sentença com 2 verbos, temos uma superestimativa).

        **Crítica**:

        **Projeto**: AIC

        **Teste**: Professores ainda temem que a distribuição do acessório dificulte a imposição, nas classes, de uma
        regra da boa conduta:

        **Contagens**: 2 verbos, 2 orações (oração quebrada pelo parser, por isso termina com dois-pontos)

        **Resultado Esperado**: 1

        **Resultado Obtido**: 1

        **Status**: correto
    """

    name = 'Ratio of Sentences With Two Clauses'
    column_name = 'sentences_with_two_clauses'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        sentences = flat.split('</s>')[:-2]
        occurences = 0
        for sentence in sentences:
            if sentence.count(' V ') - sentence.count('<aux>') == 2:
                occurences += 1
        try:
            return occurences / len(sentences)
        except ZeroDivisionError:
            return 0


class SentencesWithThreeClause(base.Metric):
    """
        **Nome da Métrica**: sentences_with_three_clauses

        **Interpretação**: quanto maior a proporção, maior a complexidade

        **Descrição da métrica**: Proporção de sentenças que contenham 3 orações

        **Definição dos termos que aparecem na descrição da métrica**:

            • oração é uma unidade sintática que contém um verbo e os constituintes ligados a ele. O predicado pode
            ser formado por um único verbo (verbo principal) ou por uma locução verbal, isto é, um verbo principal
            acompanhado de um ou mais verbos auxiliares.

            • Sentença é a unidade sintática iniciada por letra maiúscula E terminada por ponto final, ponto de
            exclamação, ponto de interrogação ou reticências.

        **Forma de cálculo da métrica**: para obter o total de sentenças, utiliza-se o sentenciador do parser; para
        obter a quantidade de sentenças com 3 orações (com 3 verbos principais), verifica-se em cada sentença a
        quantidade de verbos, subtraem-se os verbos auxiliares e contam-se os resultados que derem 3; depois divide-se
        a quantidade de sentenças com 3 verbos pela quantidade total de sentenças do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser

        **Limitações da métrica**: a precisão da métrica depende do desempenho do parser. O parser Palavras gera uma
        quebra de sentença sempre que a sentença é longa, utilizando como ponto de quebra a ocorrência de dois-pontos
        ou ponto-e-vírgula. Esse fato pode levar o sistema a subestimar a quantidade de sentenças com 3 verbos.

        **Crítica**:

        **Projeto**: AIC

        **Teste**: Uma parcela critica o uniforme, porque acredita que ele ameaçaria a individualidade de cada um.

        **Contagens**: 3 verbos, 3 orações: critica, acredita, ameaçaria

        **Resultado Esperado**: 1

        **Resultado Obtido**: 1

        **Status**: correto
    """

    name = 'Ratio of Sentences With Three Clauses'
    column_name = 'sentences_with_three_clauses'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        sentences = flat.split('</s>')[:-2]
        occurences = 0
        for sentence in sentences:
            if sentence.count(' V ') - sentence.count('<aux>') == 3:
                occurences += 1
        try:
            return occurences / len(sentences)
        except ZeroDivisionError:
            return 0


class SentencesWithFourClause(base.Metric):
    """
        **Nome da Métrica**: sentences_with_four_clauses

        **Interpretação**: quanto maior a proporção, maior a complexidade

        **Descrição da métrica**: Proporção de sentenças que contenham 4 orações

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: para obter o total de sentenças, utiliza-se o sentenciador do parser; para
        obter a quantidade de sentenças com 4 orações (com 4 verbos principais), verifica-se em cada sentença a
        quantidade de verbos, subtraem-se os verbos auxiliares  e contam-se os resultados que derem 4; depois divide-se
        a quantidade de sentenças com 4 verbos pela quantidade total de sentenças do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser

        **Limitações da métrica**:  a precisão da métrica depende do desempenho do parser. O parser Palavras gera uma
        quebra de sentença sempre que a sentença é longa, utilizando como ponto de quebra a ocorrência de dois-pontos
        ou ponto-e-vírgula. Esse fato pode levar o sistema a subestimar a quantidade de sentenças com 4 verbos.

        **Crítica**:

        **Projeto**: AIC

        **Teste**: Silva adquiriu uma tela de retenção com 150 metros, que deve ser instalada para isolar o trecho da
        barragem mais usado pelos banhistas, com 3,7 mil hectares de área.

        **Contagens**: 4 verbos, 4 orações: principais: adquiriu, instalada, isolar, usado (deve e ser são verbos auxiliares e não contam)

        **Resultado Esperado**: 1

        **Resultado Obtido**: 1

        **Status**: correto
    """

    name = 'Ratio of Sentences With Four Clauses'
    column_name = 'sentences_with_four_clauses'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        sentences = flat.split('</s>')[:-2]
        occurences = 0
        for sentence in sentences:
            if sentence.count(' V ') - sentence.count('<aux>') == 4:
                occurences += 1
        try:
            return occurences / len(sentences)
        except ZeroDivisionError:
            return 0


class SentencesWithFiveClause(base.Metric):
    """
        **Nome da Métrica**: sentences_with_five_clauses

        **Interpretação**: quanto maior a proporção, maior a complexidade

        **Descrição da métrica**: Proporção de sentenças que contenham 5 orações

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: para obter o total de sentenças, utiliza-se o sentenciador do parser; para
        obter a quantidade de sentenças com 5 orações (com 5 verbos principais), verifica-se em cada sentença a
        quantidade de verbos, subtraem-se os verbos auxiliares  e contam-se os resultados que derem 5; depois divide-se
        a quantidade de sentenças com 5 verbos pela quantidade total de sentenças do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser

        **Limitações da métrica**:  a precisão da métrica depende do desempenho do parser. O parser Palavras gera uma
        quebra de sentença sempre que a sentença é longa, utilizando como ponto de quebra a ocorrência de dois-pontos
        ou ponto-e-vírgula. Esse fato pode levar o sistema a subestimar a quantidade de sentenças com 5 verbos.

        **Crítica**:

        **Projeto**: AIC

        **Teste**: Estamos cadastrando ferros-velhos, mas não tivemos um trabalho forte contra a clonagem e agora vamos
        atacar para valer _ admitiu Bacci.

        **Contagens**: 5 verbos, 5 orações: cadastrando, tivemos, atacar, valer, admitiu (estamos e vamos são verbos
        auxiliares e não contam)

        **Resultado Esperado**: 1

        **Resultado Obtido**:  1

        **Status**: correto
    """

    name = 'Ratio of Sentences With Five Clauses'
    column_name = 'sentences_with_five_clauses'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        sentences = flat.split('</s>')[:-2]
        occurences = 0
        for sentence in sentences:
            if sentence.count(' V ') - sentence.count('<aux>') == 5:
                occurences += 1
        try:
            return occurences / len(sentences)
        except ZeroDivisionError:
            return 0


class SentencesWithSixClause(base.Metric):
    """
        **Nome da Métrica**: sentences_with_six_clauses

        **Interpretação**: quanto maior a proporção, maior a complexidade

        **Descrição da métrica**: Proporção de sentenças que contenham 6 orações

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: para obter o total de sentenças, utiliza-se o sentenciador do parser; para
        obter a quantidade de sentenças com 6 orações (com 6 verbos principais), verifica-se em cada sentença a
        quantidade de verbos, subtraem-se os verbos auxiliares  e contam-se os resultados que derem 6; depois divide-se
        a quantidade de sentenças com 6 verbos pela quantidade total de sentenças do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser

        **Limitações da métrica**:  a precisão da métrica depende do desempenho do parser. O parser Palavras gera uma
        quebra de sentença sempre que a sentença é longa, utilizando como ponto de quebra a ocorrência de dois-pontos
        ou ponto-e-vírgula. Esse fato pode levar o sistema a subestimar a quantidade de sentenças com 6 verbos. Além
        disso, o Palavras considera todos as formas de particípio como formas verbais, quando na verdade os particípios
        podem funcionar como substantivos e adjetivos também (vide teste abaixo).

        **Crítica**:

        **Projeto**: AIC

        **Teste**: Uma dúvida que paira hoje na Assembléia é se Ubirajara Amaral Macalão, principal envolvido na compra
        e no desvio de selos, era qualificado para ocupar a direção do Departamento de Serviços Administrativos (DSA).

        **Contagens**: 6 verbos: paira, é, envolvido (adjetivo particípio), era, qualificado (adjetivo particípio),
        ocupar. Como a forma de cálculo das orações é via quantidade de verbos não auxiliares, a quantidade de orações
        é 6.

        **Resultado Esperado**: 1

        **Resultado Obtido**: 1

        **Status**: correto

    """

    name = 'Ratio of Sentences With Six Clauses'
    column_name = 'sentences_with_six_clauses'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        sentences = flat.split('</s>')[:-2]
        occurences = 0
        for sentence in sentences:
            if sentence.count(' V ') - sentence.count('<aux>') == 6:
                occurences += 1
        try:
            return occurences / len(sentences)
        except ZeroDivisionError:
            return 0


class SentencesWithSevenMoreClause(base.Metric):
    """
        **Nome da Métrica**: sentences_with_seven_more_clauses

        **Interpretação**: quanto maior a proporção, maior a complexidade

        **Descrição da métrica**: Proporção de sentenças que contenham 7 ou mais orações

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: para obter o total de sentenças, utiliza-se o sentenciador do parser; para
        obter a quantidade de sentenças com 7 ou mais orações (com 7 ou mais verbos principais), verifica-se em cada
        sentença a quantidade de verbos, subtraem-se os verbos auxiliares e contam-se os resultados que derem mais que
        6; depois divide-se a quantidade de sentenças com mais de 6 verbos pela quantidade total de sentenças do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser

        **Limitações da métrica**:  a precisão da métrica depende do desempenho do parser. O parser Palavras gera uma
        quebra de sentença sempre que a sentença é longa, utilizando como ponto de quebra a ocorrência de dois-pontos
        ou ponto-e-vírgula. Esse fato pode levar o sistema a subestimar a quantidade de sentenças com 7 ou mais verbos.
        Além disso, o Palavras considera todos as formas de particípio como formas verbais, quando na verdade os
        particípios podem funcionar como substantivos e adjetivos também (vide teste abaixo).

        **Crítica**:

        **Projeto**: AIC

        **Teste**: Os aparelhos no teto, explica Simões, são os preferidos pela possibilidade de puxar a tela para cima
        quando não é usada, deixando o equipamento menos visível, dificultando furtos.

        **Contagens**: 7 verbos, 7 orações: explica, são, preferidos (particípio adjetivo), puxar, usada, deixando,
        dificultando.

        **Resultado Esperado**: 1

        **Resultado Obtido**: 1

        **Status**: correto
    """

    name = 'Ratio of Sentences With Seven More Clauses'
    column_name = 'sentences_with_seven_more_clauses'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        sentences = flat.split('</s>')[:-2]
        occurences = 0
        for sentence in sentences:
            if sentence.count(' V ') - sentence.count('<aux>') >= 7:
                occurences += 1
        try:
            return occurences / len(sentences)
        except ZeroDivisionError:
            return 0


class ClausesPerSentece(base.Metric):
    """
        **Nome da Métrica**: clauses_per_sentence

        **Interpretação**: quanto maior o número de orações por sentença, maior a complexidade

        **Descrição da métrica**: média de orações por sentença

        **Definição dos termos que aparecem na descrição da métrica**: oração é a unidade do texto que apresenta um
        verbo principal

        **Forma de cálculo da métrica**: Conta-se a quantidade de orações e divide-se o resultado pelo número de
        sentenças. Para computar a quantidade de orações, somam-se todos os verbos e subtraem-se os verbos auxiliares
        (V – aux)

        **Recursos de PLN utilizados durante o cálculo**: sentenciador, para delimitar sentenças e tagger, para
        etiquetar verbos e verbos auxiliares. O classificador usa o parser Palavras tanto para sentenciar quanto para
        fazer as funções de tagger

        **Limitações da métrica**: o parser não distingue particípios passados nas funções de verbo, substantivo e
        adjetivo, considerando-os todos verbos. Isso pode levar a uma superestimativa do número de orações.

        **Crítica**:

        **Projeto**: AIC

        **Teste**:

        Tendemos a pensar que os hereges são pessoas que nadam contra a corrente, indivíduos inclinados a desafiar o
        conhecimento dominante. Às vezes, porém, um herege é apenas um pensador convencional que permanece olhando na
        mesma direção, ao passo que todos os demais passaram a olhar na direção contrária. Quando, em 1957, John Yudkin
        aventou pela primeira vez a possibilidade de o açúcar representar um perigo para a saúde pública, a hipótese foi
        levada a sério, assim como seu proponente. Ao se aposentar, catorze anos depois, tanto a teoria como seu autor
        haviam sido ridicularizados e marginalizados. Somente agora, postumamente, é que seu trabalho vem sendo
        reconduzido ao pensamento científico consolidado.

        **Contagens**: 5 sentenças, 16 orações

        **Resultado Esperado**: 3,20 (16/5)

        **Resultado Obtido**: 3,00 (o parser reconheceu 15 orações)

        **Status**: correto, considerando a limitação do parser
    """

    name = 'Mean Clauses Per Sentence'
    column_name = 'clauses_per_sentence'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        sentences = flat.split('</s>')[:-2]
        mean = []
        for sentence in sentences:
            mean.append(sentence.count(' V ') - sentence.count('<aux>'))
        try:
            return sum(mean) / len(mean)
        except ZeroDivisionError:
            return 0


class CoordinateConjunctions(base.Metric):
    """
        **Nome da Métrica**: ratio_coordinate_conjunctions

        **Interpretação**: as conjunções coordenativas parecem ser índice de estruturas menos complexas que as
        conjunções subordinativas. Assim, quanto maior o resultado da métrica, menor a complexidade.

        **Descrição da métrica**: proporção de conjunções coordenativas em relação ao total de conjunções do texto

        **Definição dos termos que aparecem na descrição da métrica**: há dois tipos de conjunções: coordenativas e
        subordinativas. Conjunções coordenativas são uma classe de palavras que unem duas orações independentes ou duas
        palavras de conteúdo ou dois sintagmas nominais. Conjunções subordinativas introduzem orações subordinadas.

        **Forma de cálculo da métrica**: Divide-se a quantidade de conjunções coordenativas (etiqueta KC) pela
        quantidade total de conjunções coordenativas e subordinativas do texto (etiquetas KC + KS).

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras para identificar as conjunções.

        **Limitações da métrica**: a precisão da métrica depende da precisão do parser Palavras na identificação das
        conjunções.

        **Crítica**:

        **Projeto**: AIC

        **Teste 1**: Conforme as pesquisas progrediram, a equipe descobriu que a resistência não se estende só ao gambá
        propriamente dito, mas também às cuícas e outros parentes do animal, todos caçadores de cobras, que teriam tido
        vantagens em desenvolver tais defesas bioquímicas.

        **Contagens**: 2 conjunções coordenativas (mas, e) e 2 conjunções subordinativas (conforme, que). O parser,
        porém, não reconheceu “conforme” como KS.

        **Resultado Esperado**: 0,667 (2/3, considerando a limitação do parser)

        **Resultado Obtido**: 0,667

        **Status**: correto

        **Teste 2**: Na zona rural da Venezuela, as pessoas diziam que o gambá era resistente às picadas, mas não se
        sabia como.

        **Contagens**: 1 conjunção coordenativa (mas) e 2 conjunções subordinativas (que, como). O parser, porém,
        reconheceu também o “se” como KS, quando na verdade ele é um pronome, índice de indeterminação do sujeito.

        **Resultado Esperado**: 0,333 (1/3, considerando a limitação do parser)

        **Resultado Obtido**: 0,333

        **Status**: correto
    """

    name = 'Ratio of Coordinate Conjunctions per Number of Conjunctions'
    column_name = 'ratio_coordinate_conjunctions'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            conjuncoes = (flat.count(' KC ') + flat.count(' KS '))
            return flat.count(' KC ') / conjuncoes
        except ZeroDivisionError:
            return 0


class SubordinateConjunctions(base.Metric):
    """
        **Nome da Métrica**: ratio_subordinate_conjunctions

        **Interpretação**: as conjunções subordinativas, por introduzirem orações subordinadas, indicam estruturas mais
        complexas que conjunções coordenativas. Teoricamente, portanto, quanto maior o resultado, maior a complexidade.

        **Descrição da métrica**: proporção de conjunções subordinativas em relação à soma de conjunções subordinativas
        e coordenativas do texto.

        **Definição dos termos que aparecem na descrição da métrica**: há dois tipos de conjunções: coordenativas e
        subordinativas. Conjunções coordenativas são uma classe de palavras que unem duas orações independentes ou duas
        palavras de conteúdo ou dois sintagmas nominais. Conjunções subordinativas introduzem orações subordinadas.

        **Forma de cálculo da métrica**: identificam-se todas as conjunções do texto (etiquetas KC e KS). Divide-se a
        quantidade de conjunções subordinativas (etiqueta KS do parser Palavras) pela quantidade total de conjunções do
        texto (etiquetas KC + KS).

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras.

        **Limitações da métrica**: a precisão da métrica depende da precisão do parser Palavras na identificação das
        conjunções.

        **Crítica**:

        **Projeto**: AIC

        **Teste**: O secretário da Segurança Pública, Enio Bacci, disse que o aumento está ligado à legislação branda
        contra desmanches, ao aumento da frota e ao chamado golpe do seguro -- quando o dono vende o carro a bandidos e
        recebe um novo da seguradora, mas reconheceu que precisa ajustar a repressão.

        **Contagens**: 5 conjunções, 3 coordenativas (e, e, mas) e 2 subordinativas (que, que)

        **Resultado Esperado**: 0,4 (2/5)

        **Resultado Obtido**: 0,4

        **Status**: correto
    """

    name = 'Ratio of Subordinate Conjunctions per Number of Conjunctions'
    column_name = 'ratio_subordinate_conjunctions'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            conjuncoes = (flat.count(' KC ') + flat.count(' KS '))
            return flat.count(' KS ') / conjuncoes
        except ZeroDivisionError:
            return 0


class GerundVerbs(base.Metric):
    """
        **Nome da Métrica**: gerund_verbs

        **Interpretação**: não é clara a relação entre a presença de verbos no particípio e nível de complexidade

        **Descrição da métrica**: Proporção de verbos no particípio em relação a todos os verbos do texto.

        **Definição dos termos que aparecem na descrição da métrica**: verbos no particípio possuem formas regulares e
        irregulares. Alguns verbos têm as duas formas, como salvar: salvado, salvo.

        **Forma de cálculo da métrica**: contam-se todos os verbos (etiqueta V do parser Palavras) e todos os verbos no
         particípio (etiqueta V PCP). Divide-se a quantidade de verbos no particípio pela quantidade total de verbos.

        **Recursos de PLN utilizados durante o cálculo**: tagger do parser Palavras

        **Limitações da métrica**:

        **Crítica**: verbos no particípio são ambíguos com adjetivos e substantivos derivados de particípio. O parser
        quase sempre classifica todos como verbos, o que pode levar a uma superestimativa da quantidade de particípios.

        **Projeto**: AIC

        **Teste**: Estamos fazendo uma inspeção preventiva em todos os sistemas de alarme que tenham sido instalados na
        fábrica há mais de 5 anos, visando detectar mau funcionamento.

        **Contagens**: 8 verbos, 2 no gerúndio (fazendo e visando), mas o parser não reconhece “há” como verbo , então
        são 7 verbos reconhecidos.

        **Resultado Esperado**: 2/7 = 0,286

        **Resultado Obtido**: 0,286

        **Status**: correto
    """

    name = 'Ratio of Gerund Verbs'
    column_name = 'gerund_verbs'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            return flat.count('V GER') / flat.count(' V ')
        except ZeroDivisionError:
            return 0


class ParticipleVerbs(base.Metric):
    """
        **Nome da Métrica**: participle_verbs

        **Interpretação**: não é clara a relação entre a presença de verbos no particípio e nível de complexidade

        **Descrição da métrica**: Proporção de verbos no particípio em relação a todos os verbos do texto.

        **Definição dos termos que aparecem na descrição da métrica**: verbos no particípio são formas não flexionadas,
        que podem ser regulares (teminados em –ado, -ido) ou irregulares. Ex: formado, falado, comido, tido, sido,
        (formas regulares), composto, morto, aberto (irregulares). Além de atuarem como verbos em formas compostas e
        orações reduzidas, as formas do particípio podem constituir substantivos e adjetivos (ex: aposentado, sentido,
        vestido, cansado).

        **Forma de cálculo da métrica**: contam-se todos os verbos (etiqueta V do parser Palavras) e todos os verbos no
        particípio (etiqueta V PCP). Divide-se a quantidade de verbos no particípio pela quantidade total de verbos.

        **Recursos de PLN utilizados durante o cálculo**: tagger do parser Palavras

        **Limitações da métrica**: o parser não distingue a função do particípio (verbo, substantivo, adjetivo), exceto
        quando um artigo o precede (ex: o vestido). Isso pode levar a uma superestimativa dessa forma verbal.

        **Crítica**:

        **Projeto**: AIC

        **Teste**: É importante atentar para os testes que têm sido feitos após a retirada do país da Comunidade
        Europeia.

        **Contagens**: 5 verbos (é, atentar, têm, sido, feitos), 2 no particípio (sido, feitos)

        **Resultado Esperado**: 2/5 = 0,40

        **Resultado Obtido**: 0,40

        **Status**: correto
    """

    name = 'Ratio of Participle Verbs'
    column_name = 'participle_verbs'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            return flat.count('V PCP') / flat.count(' V ')
        except ZeroDivisionError:
            return 0


class InfinitiveVerbs(base.Metric):
    """
        **Nome da Métrica**: infinitive_verbs

        **Interpretação**: não é clara a relação entre a presença de verbos no infinitivo e nível de complexidade

        **Descrição da métrica**: Proporção de verbos no infinitivo em relação a todos os verbos do texto.

        **Definição dos termos que aparecem na descrição da métrica**: verbos no infinitivo são aqueles não flexionados,
        teminados em –ar, -er, -ir e –or: falar, ler, sorrir, compor.

        **Forma de cálculo da métrica**: contam-se todos os verbos (etiqueta V do parser Palavras) e todos os verbos no
        infinitivo (etiqueta V INF). Divide-se a quantidade de verbos no infinitivo pela quantidade total de verbos.

        **Recursos de PLN utilizados durante o cálculo**: tagger do parser Palavras

        **Limitações da métrica**: o parser classifica todas as palavras desconhecidas ou ambíguas como verbos, desde
        que tenham terminação igual à dos verbos. Assim, palavras como “placar” e “[o] jantar” (refeição), são anotadas
        como verbos infinitivos.

        **Crítica**:

        **Projeto**: AIC

        **Teste**: É importante atentar para os testes que têm sido feitos após a retirada do país da Comunidade
        Europeia.

        **Contagens**: 5 verbos (é, atentar, têm, sido, feitos), 1 no infinitivo (atentar)

        **Resultado Esperado**: 1/5 = 0,20

        **Resultado Obtido**: 0,20

        **Status**: correto
    """

    name = 'Ratio of Infinitive Verbs'
    column_name = 'infinitive_verbs'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            return flat.count('V INF') / flat.count(' V ')
        except ZeroDivisionError:
            return 0


class InflectedVerbs(base.Metric):
    """
        **Nome da Métrica**: inflected_verbs

        **Interpretação**: não é clara a relação entre verbos flexionados e complexidade

        **Descrição da métrica**: Proporção de verbos flexionados em relação a todos os verbos do texto

        **Definição dos termos que aparecem na descrição da métrica**: verbos flexionados são aqueles que sofreram
        flexão para adequarem-se à pessoa, ao número, ao modo e ao tempo verbais.

        **Forma de cálculo da métrica**: contam-se todas as ocorrências de VFIN (formas verbais flexionadas) e divide-se
        pela quantidade de V (todos os verbos do texto).

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras na função de POS tagger.

        **Limitações da métrica**:

        **Crítica**:

        **Projeto**: AIC

        **Teste**: É importante observar os testes que têm sido feitos após a retirada do país da Comunidade Europeia.

        **Contagens**: 5 verbos, 2 flexionados (é, têm) e 3 não flexionados (observar, sido, feitos)

        **Resultado esperado**: 2/5 = 0,4

        **Resultado Obtido**: 0,40

        **Status**: correto
    """

    name = 'Ratio of Inflected Verbs'
    column_name = 'inflected_verbs'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            return flat.count('VFIN') / flat.count(' V ')
        except ZeroDivisionError:
            return 0


class PrepositionsPerSentence(base.Metric):
    """
        **Nome da Métrica**: prepositions_per_sentence

        **Interpretação**: não está clara a relação entre o uso de preposições e a complexidade textual

        **Descrição da métrica**: Média de preposições por sentença:

        **Definição dos termos que aparecem na descrição da métrica**: preposições são palavras que pertencem a uma
        classe gramatical fechada (finita). São classificadas como palavras funcionais, ou seja, têm uma função na
        sintaxe, mas não têm conteúdo isoladamente.

        **Forma de cálculo da métrica**: contam-se as ocorrências da etiqueta PRP e divide-se o resultado pelo número
        de sentenças do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras na função de POS tagger

        **Limitações da métrica**:

        **Crítica**:

        **Projeto**: AIC

        **Teste**: Nem é preciso argumentar contra a ineficiência do sistema prisional brasileiro. Ele foi reprovado por
         todas as pessoas para as quais foi solicitada uma avaliação. Nele não se pode confiar e dele não se pode
         esperar nada além do estímulo à violência.

        **Contagens**: 8 preposições, 3 sentenças

        **Resultado Esperado**: 2,66

        **Resultado Obtido**: 2,66

        **Status**: correto
    """

    name = 'Mean of Prepositions Per Sentence'
    column_name = 'prepositions_per_sentence'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        sentences = flat.split('</s>')[:-2]
        preps = []
        for sentence in sentences:
            preps.append(sentence.count('PRP'))
        try:
            return sum(preps) / len(preps)
        except ZeroDivisionError:
            return 0


class PrepositionsPerClause(base.Metric):
    """
        **Nome da Métrica**: prepositions_per_clause

        **Interpretação**: não está clara a relação entre o uso de preposições e a complexidade textual

        **Descrição da métrica**: Média de preposições por sentença:

        **Definição dos termos que aparecem na descrição da métrica**: preposições são palavras que pertencem a uma
        classe gramatical fechada (finita). São classificadas como palavras funcionais, ou seja, têm uma função na
        sintaxe, mas não têm significado isoladamente.

        **Forma de cálculo da métrica**: contam-se as ocorrências da etiqueta PRP e divide-se o resultado pelo número
        de orações do texto. O número de orações é obtido contando-se as ocorrências de verbos menos as ocorrências de
        verbos auxiliares. (V-aux).

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**:

        **Crítica**:

        **Projeto**: AIC

        **Teste**: Nem é preciso argumentar contra a ineficiência do sistema prisional brasileiro. Ele foi reprovado por
        todas as pessoas para as quais foi solicitada uma avaliação. Nele não se pode confiar e dele não se pode esperar
        nada além do estímulo à violência.

        **Contagens**: 8 preposições, 6 orações

        **Resultado Esperado**: 1,33

        **Resultado Obtido**: 1,33

        **Status**: correto
    """

    name = 'Mean Prepositions Per Clause'
    column_name = 'prepositions_per_clause'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            verbos = (flat.count(' V ') - flat.count('<aux>'))
            return flat.count('PRP') / verbos
        except ZeroDivisionError:
            return 0


class RelativeClauses(base.Metric):
    """
        **Nome da Métrica**: relative_clauses

        **Interpretação**: orações relativas estão associadas à maior complexidade textual, portanto, quanto maior a
        proporção, maior a complexidade

        **Descrição da métrica**: proporção de orações relativas em relação ao total de orações do texto

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: Contam-se as ocorrências da etiqueta <rel> e divide-se pelo número de orações
        do texto. O número de orações é obtido contando-se as ocorrências de verbos menos as ocorrências de verbos
        auxiliares. (V-aux).

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**: a etiqueta <rel> marca pronomes relativos, não somente aqueles que introduzem
        orações, o que pode levar a uma superestimativa do número de orações relativas.

        **Crítica**:

        **Projeto**: AIC

        **Teste 1**: O presidente dos EUA, Donald Trump, defendeu pelo Twitter nesta quinta-feira que o terrorista
        uzbeque responsável pelo atentado em Nova York desta semana seja condenado à morte. Sayfullo Saipov pediu que a
        bandeira do Estado Islâmico fosse pendurada no quarto do hospital onde está sendo tratado. O homem de 28 anos,
        que vive há sete anos nos Estados Unidos, ficou ferido no abdômen por um tiro da polícia, que conseguiu
        prendê-lo logo após ele ter atropelado pedestres e ciclistas numa ciclovia de Manhattan na terça-feira.

        **Contagens**: 11 orações, 3 relativas

        **Resultado Esperado**: 0,273

        **Resultado Obtido**: 0,273

        **Status**: correto
    """

    name = 'Mean of Relative Clauses'
    column_name = 'relative_clauses'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            verbos = (flat.count(' V ') - flat.count('<aux>'))
            return flat.count('<rel>') / verbos
        except ZeroDivisionError:
            return 0


class AppositionPerClause(base.Metric):
    """
        **Nome da Métrica**: apposition_per_clause

        **Interpretação**: apostos estão associados a uma maior complexidade textual

        **Descrição da métrica**: quantidade média de apostos por oração

        **Definição dos termos que aparecem na descrição da métrica**: aposto é um constituinte sintático que especifica
        algo sobre um NP. É identificado por meio da etiqueta @APP do parser Palavras.

        **Forma de cálculo da métrica**: Contam-se as ocorrências da etiqueta @APP e divide-se pelo número de orações do
        texto. O número de orações é obtido contando-se as ocorrências de verbos menos as ocorrências de verbos
        auxiliares. (V-<aux>)).

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**: o desempenho do parser Palavras na identificação de apostos é bem fraca, de acordo
        com vários testes que fizemos.

        **Crítica**: Essa métrica é muito forte, por isso é importante que seja bem capturada. A etiqueta APP do parser
        Palavras não está capturando todos os apostos.

        **Projeto**: AIC

        **Teste 1**: O homem, um militar de alta patente, chegou em um carro blindado, prova de que se sente ameaçado.
        Só hoje, segunda-feira, soubemos que ele veio verificar a presença de meliantes, na maioria infantis, na nossa
        escola.

        **Contagens**: 4 apostos, 6 orações (o Palavras identificou 5 orações porque classificou “veio verificar” como
        um só VP)

        **Resultado Esperado**: o correto seria 0,66, mas se considerarmos o erro do Palavras, o esperado é 0,8.

        **Resultado obtido**: 0,0 (o parser não anotou nenhum APP)

        **Status**: incorreto

        **Teste 2**: A Ecologia, ciência que investiga as relações dos seres vivos entre si e com o meio em que vivem,
        adquiriu grande destaque no mundo atual.

        **Contagens**: 1 aposto, 2 orações

        **Resultado Esperado**: 0,5

        **Resultado Obtido**: 0,0 (o parser não anotou nenhum APP)

        **Status**: incorreto

        **Teste 3**: O homem mais rico do mundo, Bill Gates, é um grande filantropo.

        **Contagens**: 1 aposto, 1 oração

        **Resultado Esperado**: 1,0

        **Resultado Obtido**: 1,0

        **Status**: correto
    """

    name = 'Mean Apposition Per Clause'
    column_name = 'apposition_per_clause'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            verbos = (flat.count(' V ') - flat.count('<aux>'))
            return flat.count('APP') / verbos
        except ZeroDivisionError:
            return 0


class AdverbialAdjunctPerClause(base.Metric):
    """
        **Nome da Métrica**: adjunct_per_clause

        **Interpretação**: quanto maior a proporção de adjuntos adverbiais por oração, maior a complexidade textual

        **Descrição da métrica**: média de adjuntos adverbiais por oração

        **Definição dos termos que aparecem na descrição da métrica**: uma oração corresponde a um verbo principal.

        **Forma de cálculo da métrica**: Contam-se todas as ocorrências das etiquetas @ADVL,@>ADVL e @<ADVL e depois
        divide-se o resultado pelo número de orações do texto. Para calcular o número de orações, contam-se as
        ocorrências de verbos e subtraem-se dos resultados as ocorrências de verbos auxiliares (V - aux).

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras na função de POS tagger

        **Limitações da métrica**:

        **Crítica**:

        **Projeto**: AIC

        **Teste**: A resposta de Trump ao primeiro ataque terrorista em solo americano desde o início do seu mandato é
        dura e polêmica. Ontem, ele disse que cogitava enviar Saipov para a prisão de Guantánamo, a mesma que o seu
        antecessor democrata Barack Obama esvaziou e planejava desativar por completo. Guantánamo se tornou famosa
        quando se tornou o presídio dos combatentes capturados no Afeganistão após a invasão liderada pelos Estados
        Unidos depois dos atentados de 11 de setembro de 2001. As condições dos presos mantidos na base naval americana
        foram motivo de indignação internacional e alvo de duras críticas, tanto por parte de governos como de
        organizações humanitárias internacionais.

        **Contagens**: 13 orações (é, disse, cogitava, enviar, esvaziou, planejava, desativar, tornou, tornou,
        capturados, liderada, mantidos, foram) e 8 adjuntos adverbiais (em solo americano, desde o início do seu
        mandato, ontem, por completo, quando se tornou..., no Afeganistão, após a invasão..., depois dos atentados
        de...)

        **Resultado Esperado**: 0,615

        **Resultado obtido**: 0,615

        **Status**: correto
    """

    name = 'Mean of Adverbial Adjunct Per Clause'
    column_name = 'adjunct_per_clause'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            verbs = (flat.count(' V ') - flat.count('<aux>'))
            count_advls = len(re.findall('@[<>]{0,1}ADVL', flat))
            return count_advls / verbs
        except ZeroDivisionError:
            return 0


class FirstPersonPossessivePronouns(base.Metric):
    """
        **Nome da Métrica**: first_person_possessive_pronouns

        **Interpretação**: não está clara a relação da métrica com a complexidade textual

        **Descrição da métrica**: proporção de pronomes possessivos de primeira pessoa em relação à quantidade total de
        pronomes possessivos no texto

        **Definição dos termos que aparecem na descrição da métrica**: pronomes possessivos indicam posse. Os das
        primeiras pessoas são: meu, meus, minha, minhas, nosso, nossos, nossa, nossas.

        **Forma de cálculo da métrica**: contam-se as ocorrências das etiquetas <poss 1S> e <poss 1P>, pronomes
        possessivos de primeira pessoa no singular e no plural, respectivamente. Divide-se o resultado pela soma de
        todos os pronomes possessivos do texto (<poss 1S>, <poss 1P>, <poss 2S>, <poss 2P>, <poss 3S>, <poss 3P>)

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras na função de POS tagger

        **Limitações da métrica**:

        **Crítica**: os pronomes pessoais e possessivos nas primeiras pessoas, assim como os verbos flexionados nas
        primeiras pessoas constituem marcas de textos em que o autor revela sua opinião. A métrica deveria juntar as
        três ocorrências e não apenas os pronomes possessivos.

        **Projeto**: AIC

        **Teste**: Minha primeira tentativa fracassou, mas agora eu atingi meu objetivo. Seu apoio foi muito importante
        para mim. Obrigada por sua dedicação, prova do quanto é forte nossa amizade.

        **Contagens**: 3 pronomes possessivos na primeira pessoa (minha, meu, nossa), 1 pronome pessoal na primeira
        pessoa (eu), 1 pronome oblíquo na primeira pessoa (mim), 2 pronomes possessivos na terceira pessoa (seu, sua).

        **Resultado Esperado**: 3/5 = 0,60

        **Resultado Obtido**: 0,60

        **Status**: correto
    """

    name = 'Ratio of First Person Possessive Pronouns'
    column_name = 'first_person_possessive_pronouns'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            return sum(
                [
                    flat.count('<poss 1S>'),
                    flat.count('<poss 1P>'),
                ]
            ) / sum(
                [
                    flat.count('<poss 1S>'),
                    flat.count('<poss 1P>'),
                    flat.count('<poss 2S>'),
                    flat.count('<poss 2P>'),
                    flat.count('<poss 3S>'),
                    flat.count('<poss 3P>'),
                ])
        except ZeroDivisionError:
            return 0


class SecondPersonPossessivePronouns(base.Metric):
    """
        **Nome da Métrica**: second_person_possessive_pronouns

        **Interpretação**: não está clara a relação da métrica com a complexidade textual

        **Descrição da métrica**: proporção de pronomes possessivos de segunda pessoa em relação à quantidade total de
        pronomes possessivos no texto

        **Definição dos termos que aparecem na descrição da métrica**: pronomes possessivos indicam posse. Os das
        segundas pessoas são: teu, teus, tua, tuas, vosso, vossos, vossa, vossas.

        **Forma de cálculo da métrica**: contam-se as ocorrências das etiquetas <poss 2S> e <poss 2P>, pronomes
        possessivos de segunda pessoa no singular e no plural, respectivamente. Divide-se o resultado pela soma de
        todos os pronomes possessivos do texto (<poss 1S>, <poss 1P>, <poss 2S>, <poss 2P>, <poss 3S>, <poss 3P>)

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras na função de POS tagger

        **Limitações da métrica**: no português brasileiro as formas de segunda pessoa foram suplantadas pelo uso de
        “você” e “vocês”, que usam a flexão e os pronomes possessivos de terceira pessoas. Portanto, não é possível
        distinguir os possessivos que se referem a “você” e “vocês” dos possessivos que se referem a “ele, ela, eles,
        elas”.

        **Crítica**:

        **Projeto**: AIC

        **Teste**: Minha primeira tentativa fracassou, mas agora eu atingi meu objetivo. Teu apoio foi muito importante
        para mim. Obrigada por tua dedicação, prova do quanto é forte nossa amizade.

        **Contagens**: 2 pronomes possessivos na segunda pessoa (teu, tua) e 3 pronomes possessivos na primeira pessoa
        (minha, meu, nossa).

        **Resultado Esperado**: 2/5 = 0,40

        **Resultado Obtido**: 0,40

        **Status**: correto

    """

    name = 'Ratio of Second Person Possessive Pronouns'
    column_name = 'second_person_possessive_pronouns'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            return sum(
                [
                    flat.count('<poss 2S>'),
                    flat.count('<poss 2P>'),
                ]
            ) / sum(
                [
                    flat.count('<poss 1S>'),
                    flat.count('<poss 1P>'),
                    flat.count('<poss 2S>'),
                    flat.count('<poss 2P>'),
                    flat.count('<poss 3S>'),
                    flat.count('<poss 3P>'),
                ])
        except ZeroDivisionError:
            return 0


class ThirdPersonPossessivePronouns(base.Metric):
    """
        **Nome da Métrica**: third_person_possessive_pronouns

        **Interpretação**: não está clara a relação da métrica com a complexidade textual

        **Descrição da métrica**: proporção de pronomes possessivos de terceira pessoa em relação à quantidade total de
        pronomes possessivos no texto

        **Definição dos termos que aparecem na descrição da métrica**: pronomes possessivos indicam posse. Os das
        terceiras pessoas são: seu, seus, sua, suas, dele, deles, dela, delas.

        **Forma de cálculo da métrica**: contam-se as ocorrências das etiquetas <poss 3S> e <poss 3P>, pronomes
        possessivos de segunda pessoa no singular e no plural, respectivamente. Divide-se o resultado pela soma de todos
        os pronomes possessivos do texto (<poss 1S>, <poss 1P>, <poss 2S>, <poss 2P>, <poss 3S>, <poss 3P>)

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras na função de POS tagger

        **Limitações da métrica**: no português brasileiro as formas de segunda pessoa foram suplantadas pelo uso de
        “você” e “vocês”, que usam a flexão e os pronomes possessivos de terceira pessoa. Portanto, não é possível
        distinguir os possessivos que se referem a “você” e “vocês” dos possessivos que se referem a “ele, ela, eles,
        elas”, o que gera ambiguidade. Para resolver a ambiguidade, usam-se as formas “dele, deles, dela, delas”, mas o
        parser não reconhece essas formas como pronomes possessivos e sim com contrações da preposição “de” com os
        pronomes pessoais de terceira pessoa.

        **Crítica**:

        **Projeto**: AIC

        **Teste**: Seus olhos são mais escuros do que os dela, mas ainda são claros se comparados aos meus.

        **Contagens**: 2 pronomes possessivos na terceira pessoa (seus, dela) e 1 pronome possessivo na primeira pessoa
        (meus).

        **Resultado Esperado**: 2/3 = 0,66

        **Resultado Obtido**: 0,50 (porque o parser não reconhece o pronome “dela”)

        **Status**: correto (considerando a limitação do parser)

    """

    name = 'Ratio of Third Person Possessive Pronouns'
    column_name = 'third_person_possessive_pronouns'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            return sum(
                [
                    flat.count('<poss 3S>'),
                    flat.count('<poss 3P>'),
                ]
            ) / sum(
                [
                    flat.count('<poss 1S>'),
                    flat.count('<poss 1P>'),
                    flat.count('<poss 2S>'),
                    flat.count('<poss 2P>'),
                    flat.count('<poss 3S>'),
                    flat.count('<poss 3P>'),
                ])
        except ZeroDivisionError:
            return 0


class AuxiliaryParticipleSentences(base.Metric):
    """
        **Nome da Métrica**: aux_plus_PCP_per_sentence

        **Interpretação**: quanto maior a proporção de sintagmas verbais complexos (formados por mais de um verbo),
        maior a complexidade textual.

        **Descrição da métrica**: Proporção de verbos auxiliares seguidos de particípio em relação à quantidade de
        sentenças do texto.

        **Definição dos termos que aparecem na descrição da métrica**: o particípio é uma das formas nominais do verbo,
        ao lado do infinitivo e do gerúndio. Há formas regulares e irregulares de formação do particípio e alguns verbos
        apresentam as duas. As formas regulares terminam em –ado e -ido. Exemplos: falado, colhido, sentido,
        morrido/morto, suspendido/suspenso, aceitado/aceito. Quando um verbo apresenta as duas formas, a forma regular
        é usada nos tempos compostos com os auxiliares “ter” e “haver; a forma irregular é usada com o auxiliar “ser”
        (na passiva).

        **Forma de cálculo da métrica**: localizam-se todos os verbos auxiliares (<aux>) seguidos de verbo no particípio
        (V PCP). Esses casos são contados e constituem o numerador do cálculo. O denominador é a quantidade de sentenças
        do texto. Obs: Essa métrica veio do AIC como proporção de orações passivas em relação à quantidade de sentenças,
        porém identificamos um erro: apenas o verbo SER seguido de particípio deveria ser considerado voz passiva.
        Fizemos outra métrica de voz passiva e mantivemos essa métrica do AIC, apenas mudando sua nomenclatura.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**: a precisão da métrica depende do desempenho do parser Palavras.

        **Críticas**:

        **Projeto**: AIC

        **Teste**: Os homens que tinham feito a manutenção não haviam sido remunerados. Eles serão recompensados
        futuramente com dias de descanso.

        **Contagens**: 2 sentenças, 3 auxiliares seguidos de de particípio (tinham feito, haviam sido, serão
        recompensados)

        **Resultado Esperado**: 1,5

        **Resultado Obtido**: 1,5

        **Status**: correto
    """

    name = 'Ratio of Auxiliary Verbs Followed by Participles'
    column_name = 'aux_plus_PCP_per_sentence'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t).split('\n')
        occurences = 0
        for i in range(len(flat) - 1):
            if flat[i].count('<aux>'):
                if flat[i + 1].count('V PCP'):
                    occurences += 1
        try:
            return occurences / (flat.count('</s>') - 1)
        except ZeroDivisionError:
            return 0




class PassiveClauses(base.Metric):
    """
        **Nome da Métrica**: passive_ratio

        **Interpretação**: quanto maior a Proporção de passivas no texto, mais complexo ele é. Justificativa: a voz
        passiva é uma estrutura que as crianças adquirem tardiamente

        **Descrição da métrica**: Proporção de orações na voz passiva analítica em relação à quantidade de orações do
        texto.

        **Definição dos termos que aparecem na descrição da métrica**: há duas formas de voz passiva: a sintética e a
        analítica (desenvolvida). Esta métrica trata apenas da voz passiva analítica, constituída do verbo SER seguido
        de um verbo principal na forma do particípio passado. A voz passiva sintética é muito difícil de ser capturada,
        uma vez que é construída com a partícula “se”, muito ambígua no português e, por isso mesmo, sujeita a muitas
        imprecisões de anotação do parser.

        **Forma de cálculo da métrica**: localizam-se todos os verbos auxiliares (<aux>) seguidos de verbo SER no
        particípio passado (PCP). Esses casos são contados e constituem o numerador do cálculo. O denominador é a
        quantidade de orações do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**: o parser Palavras não distingue particípio passado usado como verbo do usado como
        adjetivo ou substantivo, o que pode prejudicar essa métrica. Isso pode levar a uma superestimativa das passivas
        no texto. Exemplos de construções que atendem à heurística mas não são passivas: Ele é entendido no assunto.
        Ele é aposentado. Eu sou agradecido por tudo que você fez.

        **Críticas**:

        Textos simples contêm menos passivas, mas o inverso não é verdadeiro, pois há textos complexos que contêm poucas
        passivas também. Portanto, a ausência ou baixa proporção de passivas não é evidência de baixa complexidade. A
        alta proporção de passivas, pelo contrário, pode ser evidência de maior complexidade.

        Embora no inglês a passiva seja reconhecidamente um complexificador, no português ela é muito mais frequente.
        Temos que ter o cuidado de não dar à passiva o mesmo peso que os classificadores de complexidade textual do
        inglês dão.

        **Projeto**: AIC

        **Teste**: A campanha para designar as sete maravilhas do mundo moderno foi organizada pelo empresário e
        cineasta suíço Bernard Weber, que afirma ter sido motivado a defender a preservação do patrimônio histórico após
        a destruição dos budas gigantes de Bamiyan, no Afeganistão, pelos talibãs, em 2001.

        **Contagens**: 5 orações, 2 passivas (designar, foi organizada, afirma, sido motivado, defender)

        **Resultado Esperado**: 0,4

        **Resultado Obtido**: 0,4

        **Status**: correto
    """

    name = 'Ratio of Passive Clauses'
    column_name = 'passive_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        flat = flat.split('\n')
        occurences = 0
        for i in range(len(flat) - 1):
            if flat[i].count('<aux>') and flat[i].count('[ser]'):
                if flat[i + 1].count('V PCP'):
                    occurences += 1
        try:
            return occurences / rp.num_clauses(t)
        except ZeroDivisionError:
            return 0



class NonInflectedVerbs(base.Metric):
    """
        **Nome da Métrica**: non-inflected_verbs

        **Interpretação**: não é clara a relação entre verbos não flexionados e complexidade textual

        **Descrição da métrica**: Proporção de verbos não flexionados em relação a todos os verbos do texto

        **Definição dos termos que aparecem na descrição da métrica**: verbos não flexionados são aqueles que estão na
        forma nominal (infinitivo, gerúndio e particípio) e podem tanto constituir orações subordinadas reduzidas quanto
        sintagmas verbais com cadeias de verbos auxiliares (nessas cadeias, apenas o primeiro verbo é flexionado).

        **Forma de cálculo da métrica**: contam-se todas as ocorrências de V INF + V GER + V PCP) e divide-se pela
        quantidade de V (todos os verbos do texto).

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras na função de POS tagger.

        **Limitações da métrica**: o parser Palavras não distingue a função do particípio, considerando-o sempre verbo
        (pode ser adjetivo ou substantivo), o que pode superestimar o valor dessa métrica.

        **Crítica**:

        **Projeto**: AIC

        **Teste**: É importante observar os testes que têm sido feitos após a retirada do país da Comunidade Europeia.

        **Contagens**: 5 verbos, 2 flexionados (é, têm) e 3 não flexionados (observar, sido, feitos)

        **Resultado esperado**: 3/5 = 0,6

        **Resultado Obtido**: 0,60

        **Status**: correto
    """

    name = 'Ratio of Non Inflected Verbs'
    column_name = 'non-inflected_verbs'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            return sum(
                [
                    flat.count('V GER'),
                    flat.count('V PCP'),
                    flat.count('V INF')
                ]
            ) / flat.count(' V ')
        except ZeroDivisionError:
            return 0



class AIC_Palavras(base.Category):
    name = 'AIC (Palavras)'
    table_name = 'aic_palavras'

    def __init__(self):
        super(AIC_Palavras, self).__init__()
        self._set_metrics_from_module(__name__)
