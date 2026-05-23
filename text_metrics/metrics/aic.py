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


class SimpleWords(base.Metric):
    """
        **Nome da Métrica**: simple_word_ratio

        **Interpretação**: quanto maior a Proporção, menor a complexidade do texto

        **Descrição da métrica**: proporção de palavras de conteúdo simples, sobre o total de palavras de conteúdo do
        texto

        Definição dos termos que aparecem na descrição da métrica

        **Palavras Simples**: são palavras de baixa complexidade. Atualmente é utilizada a lista de palavras do
        Dicionário de palavras simples de Maria Tereza Biderman.

        **Palavras de Conteúdo**: são as palavras que pertencem às classes gramaticais abertas (infinitas) da língua, ou
        seja, substantivos, adjetivos, verbos e advérbios.

        **Forma de cálculo da métrica**: Calcula-se a Proporção de palavras de conteúdo simples dividindo-se a
        quantidade de palavras de conteúdo simples do texto sobre o total de palavras de conteúdo do texto. Há dois
        contadores: um de palavras de conteúdo e um de palavras de conteúdo que estão na lista de palavras simples.

        **Recursos de PLN utilizados durante o cálculo**: POS tagger nlpnet

        **Limitações da métrica**: a precisão do cálculo está condicionada à precisão dos recursos utilizados
        (lematizador, tagger e lista de palavras simples).

        **Crítica**: usando o lema, todas as formas de um mesmo verbo que está na lista de palavras simples são
        automaticamente classificadas como simples. Isso não é o ideal, pois as diversas formas de um verbo têm
        diferentes níveis de complexidade.

        **Projeto**: AIC

        **Teste**: Mesmo assim, o pão francês chegou cerca de uma hora mais tarde para os fregueses, que entenderam o
        atraso.

        **Contagens**: 19 palavras, 11 palavras de conteúdo (mesmo, assim, pão, francês, chegou, hora, mais, tarde,
        fregueses, entenderam, atraso), 10 palavras de conteúdo na lista de palavras simples (mesmo, assim, pão,
        francês, chegar, hora, mais, tarde, entender, atraso)

        **Resultado Esperado**: 0,909 (10/11)

        **Resultado Obtido**: 0,909

        **Status**: correto
    """

    name = 'Ratio of Simple Words'
    column_name = 'simple_word_ratio'

    def value_for_text(self, t, rp=default_rp):
        sw = rp.simple_words()
        content_tokens = filter(pos_tagger.tagset.is_content_word, rp.tagged_words(t))
        content_words = list(map(lambda t: t[0], content_tokens))
        word_lemmas = [rp.stemmer().get_lemma(word.lower()) for word in content_words]
        count = sum(1 for word in word_lemmas if word in sw)
        return count / len(content_words)


class FirstPersonPronouns(base.Metric):
    """
        **Nome da Métrica**: first_person_pronouns

        **Interpretação**: a primeira pessoa é marca de texto pessoal, mas não está clara sua relação com complexidade
        textual

        **Descrição da métrica**: Proporção de pronomes pessoais nas primeiras pessoas em relação a todos os pronomes
        pessoais do texto.

        **Definição dos termos que aparecem na descrição da métrica**: pronomes pessoais são: eu, tu, você, ele, ela,
        nós, vós, vocês, eles, elas. Pronomes pessoais de primeiras pessoas são: eu e nós.

        **Forma de cálculo da métrica**: contam-se as ocorrências de pronomes pessoais de primeiras pessoas e divide-se
        pelo total de pronomes pessoais.

        **Recursos de PLN utilizados durante o cálculo**:

        **Limitações da métrica**:

        **Críticas**:

            1) Para identificar textos na primeira pessoa, seria necessário somar pronomes pessoais e possessivos nas
            primeiras pessoas e verbos conjugados nas primeiras pessoas. É muito comum a supressão dos pronomes pessoais
            de primeira pessoas, pois a flexão verbal já é marca suficiente de pessoa.

            2) Seria importante computar o pronome de primeira pessoa “a gente”, que no português brasileiro é comum.
            Esse pronome é de primeira pessoa do plural (corresponde a “nós”), mas usa flexão de terceira pessoa do
            singular.

        **Projeto**: AIC

        **Teste 1**:

        Após muitas viagens, esse é meu primeiro relato.  Estou praticamente sendo obrigado a relatar, impelido por
        gratidão a todos e por achar que as informações pra esse destino estão um pouco confusas. Minha tentativa é
        ajudar um pouco mais aqueles que buscam informações sobre essa área, El Chalten e El Calafate, e quem sabe
        encorajar outros viajantes! Somos um casal de mochileiros e viajamos com a economia sempre sendo uma premissa
        de viagem. Não temos metas restritas de gastar 1 dólar por dia nem nada muito radical, mas evitamos gastar
        dinheiro desnecessariamente. O orçamento é curto e exige sacrifícios, o nosso dinheiro não dá pra viajar sem
        nos preocuparmos com despesas. Portanto, sacrificamos alguns dedos para salvar a mão. Não fazemos questão de
        luxo em hospedagens nem em restaurantes e sempre que possível fazemos tudo por nossa conta.

        **Contagens**: nenhum pronome pessoal, 3 pronomes possessivos na terceira pessoa e 9 verbos flexionados nas
        primeiras pessoas.

        **Resultado Esperado**: 0,0

        **Resultado Obtido**: 0,0

        **Status**: correto

        **Teste 2**: Eu nunca mais deixei ninguém lembrar nada de nós. Mas eles insistem em dizer que eu deveria
        conversar sobre nossa relação.

        **Contagens**: 4 pronomes pessoais, sendo 3 de primeiras pessoas

        **Resultado Esperado**: 0,75

        **Resultado Obtido**: 0,75

        **Status**: correto

    """

    name = 'Ratio of First Person Pronouns'
    column_name = 'first_person_pronouns'

    def value_for_text(self, t, rp=default_rp):
        words = [i.lower() for i in rp.all_words(t)]
        try:
            return sum(
                [
                    words.count('eu'),
                    words.count('nós'),
                ]
            ) / sum(
                [
                    words.count('eu'),
                    words.count('tu'),
                    words.count('você'),
                    words.count('vocês'),
                    words.count('ele'),
                    words.count('ela'),
                    words.count('nós'),
                    words.count('vós'),
                    words.count('eles'),
                    words.count('elas'),
                ])
        except ZeroDivisionError:
            return 0


class SecondPersonPronouns(base.Metric):
    """
        **Nome da Métrica**: second_person_pronouns

        **Interpretação**: a segunda pessoa é marca de “diálogo” com o leitor em textos jornalísticos, o que está
        associado com menor complexidade textual

        **Descrição da métrica**: Proporção de pronomes pessoais nas segundas pessoas em relação a todos os pronomes
        pessoais do texto.

        **Definição dos termos que aparecem na descrição da métrica**: pronomes pessoais são: eu, tu, você, ele, ela,
        nós, vós, vocês, eles, elas. Pronomes pessoais de segunda pessoa são: tu e vós.

        **Forma de cálculo da métrica**: contam-se as ocorrências de pronomes pessoais de segunda pessoa e divide-se
        pelo total de pronomes pessoais.

        **Recursos de PLN utilizados durante o cálculo**:

        **Limitações da métrica**:

        **Crítica**:

            1) Para identificar a segunda pessoa, é preciso incluir os pronomes pessoais “você” e “vocês”, que usam a
            flexão verbal de terceira pessoa, mas são efetivamente segundas pessoas no discurso.

            2) A métrica não está contando os pronomes pessoais “você” e “vocês” em nenhuma das pessoas, mas está
            computando-os no total de pronomes.

        **Projeto**: AIC

        **Teste 1**: Você já percebeu como é difícil decorar todos aqueles nomes de compostos orgânicos? Aqui nós
        propomos uma série de dicas para você não esquecer e nem confundir os nomes.

        **Contagens**: 2 pronomes pessoais na segunda pessoa (você, você) e 1 na primeira pessoa do plural (nós).

        **Resultado Esperado**: 2/3 = 0,67

        **Resultado Obtido**: 0,67

        **Status**: correto
    """

    name = 'Ratio of Second Person Pronouns'
    column_name = 'second_person_pronouns'

    def value_for_text(self, t, rp=default_rp):
        words = [i.lower() for i in rp.all_words(t)]
        try:
            return sum(
                [
                    words.count('tu'),
                    words.count('você'),
                    words.count('vós'),
                ]
            ) / sum(
                [
                    words.count('eu'),
                    words.count('tu'),
                    words.count('você'),
                    words.count('vocês'),
                    words.count('ele'),
                    words.count('ela'),
                    words.count('nós'),
                    words.count('vós'),
                    words.count('eles'),
                    words.count('elas'),
                ])
        except ZeroDivisionError:
            return 0


class ThirdPersonPronouns(base.Metric):
    """
        **Nome da Métrica**: third_person_pronouns

        **Interpretação**: não está clara a relação da métrica com a complexidade textual

        **Descrição da métrica**: Proporção de pronomes pessoais nas terceiras pessoas em relação a todos os pronomes
        pessoais do texto.

        **Definição dos termos que aparecem na descrição da métrica**: pronomes pessoais são: eu, tu, você, ele, ela,
        nós, vós, vocês, eles, elas. Pronomes pessoais de terceira pessoa são: ele, ela, eles, elas.

        **Forma de cálculo da métrica**: contam-se as ocorrências de pronomes pessoais de terceira pessoa e divide-se
        pelo total de pronomes pessoais.

        **Recursos de PLN utilizados durante o cálculo**: POS tagger do nlpnet

        **Limitações da métrica**: a forma “a gente” funciona como um pronome de primeira pessoa do plural (que conjuga
        o verbo na terceira pessoa do singular), mas não é reconhecida pelo POS tagger.

        **Crítica**:

        **Projeto**: AIC

        **Teste 1**: Você já viu um fantasma? Eu nunca vi, mas eles são tão comuns nos filmes que a gente fica
        imaginando se eles não existem mesmo.

        **Contagens**: 2 pronomes pessoais na terceira pessoa, 1 na primeira pessoa e 1 na segunda pessoa.

        **Resultado Esperado**: 2/4 = 0,50

        **Resultado Obtido**: 0,50

        **Status**: correto
    """

    name = 'Ratio of Third Person Pronouns'
    column_name = 'third_person_pronouns'

    def value_for_text(self, t, rp=default_rp):
        words = [i.lower() for i in rp.all_words(t)]
        try:
            return sum(
                [
                    words.count('vocês'),
                    words.count('ele'),
                    words.count('ela'),
                    words.count('eles'),
                    words.count('elas')
                ]
            ) / sum(
                [
                    words.count('eu'),
                    words.count('tu'),
                    words.count('você'),
                    words.count('vocês'),
                    words.count('ele'),
                    words.count('ela'),
                    words.count('nós'),
                    words.count('vós'),
                    words.count('eles'),
                    words.count('elas'),
                ])
        except ZeroDivisionError:
            return 0


# class DiscourseMarkers(base.Metric):
#     """
#         **Nome da Métrica**: discourse_markers_ratio
#
#         **Interpretação**: Marcadores discursivos auxiliam a estruturação lógica do texto. Seu uso pode facilitar a
#         leitura. Quanto mais marcadores houver, mais complexa é a estrutura lógica do texto.
#
#         **Descrição da métrica**: proporção de marcadores discursivos em relação ao total de palavras do texto.
#
#         **Definição dos termos que aparecem na descrição da métrica**: marcadores discursivos são palavras ou expressões
#         que desempenham um papel de estruturadores lógicos do discurso.
#
#         **Forma de cálculo da métrica**: contam-se as ocorrências dos marcadores discursivos constantes da lista de
#         marcadores discursivos elaborada pelo projeto AIC e divide-se o resultado pelo total de palavras do texto.
#
#         **Recursos de PLN utilizados durante o cálculo**: tokenizador, para contar palavras e lista de marcadores
#         discursivos
#
#         **Limitações da métrica**: há palavras e expressões que somente precedidas ou sucedidas de vírgula é que têm
#         função de marcadores discursivos.
#
#         **Crítica**: a precisão desta métrica depende da completude do léxico de marcadores e do reconhecimento das
#         marcas para desambiguização. Existem listas de marcadores concorrentes: a do AIC e as da Guten. Nenhuma das duas
#         listas trabalham características de identificação da função de marcador, ou seja, há palavras da lista que podem
#         não estar funcionando como marcadores discursivos no texto e mesmo assim estão sendo computadas como se fossem.
#         Por esse motivo, a métrica será comentada.
#
#         **Projeto**: AIC
#
#         **Teste**: Embora tenhamos chegado cedo, fomos atendidos muito tarde, infelizmente. Sendo assim, não pudemos
#         comparecer ao outro compromisso. No entanto, nada nos impede de fazer isso amanhã. Assim, poderemos cumprir toda
#         nossa agenda até o final de semana.
#
#         **Contagens**: 5 marcadores: embora, infelizmente, sendo assim, no entanto, assim. 38 palavras, contando 1
#         contração (ao)
#
#         **Resultado Esperado**: 0,132
#
#         **Resultado Obtido**: 0,054
#
#         **Status**: incorreto
#     """
#
#     name = 'Ratio of Discourse Markers'
#     column_name = 'discourse_markers_ratio'
#
#     def value_for_text(self, t, rp=default_rp):
#         marcadores = rp.discourse_markers()
#         tokens = rp.all_tokens(t)
#         count = 0
#         for marcador in marcadores:
#             if marcador in tokens:
#                 count += 1
#         return count / len(rp.all_words(t))


# class AmbiguousDiscourseMarkers(base.Metric):
#     """
#         **Nome da Métrica**: ambiguous_discourse_markers
#
#         **Interpretação**: não está clara a relação da métrica com a complexidade textual
#
#         **Descrição da métrica**: proporção de marcadores discursivos com mais de uma função discursiva em relação ao
#         total de palavras do texto
#
#         **Definição dos termos que aparecem na descrição da métrica**: marcador discursivo ambíguo é o conectivo que
#         tem mais de uma função discursiva. Esses marcadores foram identificados e reunidos em forma de lista pelo
#         projeto AIC.
#
#         **Forma de cálculo da métrica**: contam-se as ocorrências dos marcadores discursivos ambíguos e divide-se o
#         resultado pelo total de palavras do texto.
#
#         **Recursos de PLN utilizados durante o cálculo**: tokenizador, para contar palavras e tabela de conectivos
#
#         **Limitações da métrica**: a precisão desta métrica depende da completude do léxico de marcadores e do
#         reconhecimento das marcas para desambiguização.
#
#         **Crítica**:
#
#         A ambiguidade aqui tratada não é de sentidos, mas de funções discursivas, o que não tem relação conhecida com a
#         complexidade textual. Os marcadores multi-palavras não estão sendo reconhecidos como um único marcador:
#         “ainda=assim” está sendo contado como dois marcadores.
#
#         **Projeto**: AIC
#
#         **Teste 1**: Assim que vocês chegarem nós iniciaremos a reunião. Portanto, não se atrasem, pois assim teremos
#         mais tempo para discutir os resultados.
#
#         **Contagens**: 2 marcadores discursivos ambíguos (assim, assim) e 21 palavras
#
#         **Resultado Esperado**: 0,095
#
#         **Resultado Obtido**: 0,095
#
#         **Status**: correto
#
#         **Teste 2**: O menino colou na prova. Ainda assim, ele não obteve uma boa nota e ainda foi advertido pela escola.
#
#         **Contagens**: 1 marcador discursivo ambíguo (assim) e 21 palavras, contando duas descontrações (em+a=na,
#         por+o=pelo)
#
#         **Resultado Esperado**: 0,048
#
#         **Resultado Obtido**: 0,158 9 (considerou 4 marcadores: ainda, assim, ainda, e)
#
#         **Status**: incorreto
#     """
#
#     name = 'Ratio of Ambiguous Discourse Markers'
#     column_name = 'ambiguous_discourse_markers'
#
#     def value_for_text(self, t, rp=default_rp):
#         marcadores = rp.ambiguous_discourse_markers()
#         tokens = rp.all_tokens(t)
#         count = 0
#         for marcador in marcadores:
#             if marcador in tokens:
#                 count += 1
#         return count / len(rp.all_words(t))


class AIC(base.Category):
    name = 'AIC'
    table_name = 'aic'

    def __init__(self):
        super(AIC, self).__init__()
        self._set_metrics_from_module(__name__)
