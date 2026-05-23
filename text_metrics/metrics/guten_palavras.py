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

from nltk import ngrams
import numpy as np

from text_metrics import base
from text_metrics.resource_pool import rp as default_rp
from text_metrics.utils import ilen
from text_metrics.tools import syllable_separator, pos_tagger
from text_metrics.metrics.anaphoras import AnaphoricReferencesBase
from text_metrics.metrics.ambiguity import get_meanings_count

from itertools import filterfalse
from itertools import chain
import re


def subfinder(mylist, pattern):
    pattern = list(pattern)
    matches = []
    i = 0
    while i < len(mylist):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            i += len(pattern)
        else:
            matches.append(mylist[i])
            i += 1
    return matches


# class SimpleSentences(base.Metric):
#     """
#         **Nome da Métrica**: single_clause_sentence_ratio
#
#         **Interpretação**: quanto maior o resultado da métrica, menor a complexidade textual
#
#         **Descrição da métrica**: Proporção de sentenças com um único verbo principal em relação à quantidade de
#         sentenças do texto.
#
#         **Definição dos termos que aparecem na descrição da métrica**: verbo principal é todo verbo de conteúdo do
#         texto, ou seja, todo verbo que não é auxiliar
#
#         **Forma de cálculo da métrica**: contam-se todas as ocorrências da etiqueta “V” e subtraem-se as ocorrências da
#         etiqueta “<aux>”. Cada vez que o resultado for 1, acumula-se +1 no contador de sentenças que possuem um único
#         verbo principal. Ao final, divide-se o resultado do contador pela quantidade de sentenças do texto.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras
#
#         **Limitações da métrica**:  não há
#
#         **Crítica**: essa métrica já existia sob outra nomenclatura: “Ratio of Sentences With One Clause”, vinda do AIC.
#         Portanto, há duas métricas iguais e uma deve ser excluída.
#
#         **Projeto**: GUTEN
#
#         **Teste**: Apesar de eu não saber se estou entendendo do assunto, eu me esforço ao máximo. No final, vou
#         conseguir.
#
#         **Contagens**: 2 sentenças: a primeira com 4 verbos, um dos quais é auxiliar (estou); a segunda com 2 verbos,
#         um dos quais é auxiliar (vou). Apenas a segunda sentença tem um único verbo principal.
#
#         **Resultado Esperado**: 1 / 2= 0,50
#
#         **Resultado Obtido**: 0,50
#
#         **Status**:correto
#     """
#
#     name = 'Simple Sentences'
#     column_name = 'single_clause_sentence_ratio'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         sentences = flat.split('</s>')[:-2]
#         simple_sentences = 0
#         for sentence in sentences:
#             verbs = len(re.findall(' V ', sentence))
#             aux = len(re.findall('<aux>', sentence))
#             # a diferença é 1 se há 1 verbo principal e 0 auxiliar.
#             # a diferença é 0 se há 1 verbo principal e 1 auxiliar.
#             if verbs - aux == 0 or verbs - aux == 1:
#                 simple_sentences += 1
#         try:
#             return simple_sentences / len(sentences)
#         except ZeroDivisionError:
#             return 0


# class AdverbialsLevel3(base.Metric):
#     """
#         **Nome da Métrica**: adverbials_03_ratio
#
#         **Interpretação**: orações subordinadas adverbiais estão relacionadas a maior complexidade
#
#         **Descrição da métrica**: Proporção de orações subordinadas adverbiais de nível 3 em relação à quantidade de
#         orações do texto
#
#         **Definição dos termos que aparecem na descrição da métrica**: oração adverbial é um tipo de oração subordinada.
#         Ela se encaixa na oração principal, funcionando como adjunto adverbial. Há 9 tipos de orações adverbiais:
#         causais, comparativas, concessivas, condicionais, conformativas, consecutivas, finais, proporcionais e temporais. 
#         As orações subordinadas adverbiais podem ser introduzidas por conjunções subordinativas (KS), mas nem toda
#         oração subordinada introduzida por conjunção subordinativa é adverbial. Há orações subordinadas adverbiais
#         reduzidas de gerúndio, particípio e infinitivo, as quais não estão sendo capturadas pela métrica.
#
#         **Forma de cálculo da métrica**: identificam-se as sequências lexicais definidas na lista e contam-se as
#         ocorrências. Depois divide-se o resultado pela soma de ocorrências de KS, KC e <rel>.
#
#         **Recursos de PLN utilizados durante o cálculo**: POS tagger nlpnet
#
#         **Limitações da métrica**:
#
#         **Crítica**: vários motivos justificam a recomendação de comentar esta métrica:
#
#             1) A forma de cálculo da quantidade de orações (denominador do cálculo) está errada
#
#             2) a lista de palavras que introduzem orações adverbiais precisa ser revista;
#
#             3) das 9 classes de orações adverbiais existentes, o léxico só apresenta 5
#
#             4) há uma classe definida que não existe: explicativas (que associa-se à etiqueta <rel>) Somente orações
#             subordinadas adjetivas têm uma subcategoria chamada “explicativas”;
#
#             5) as listas de nível 1, 2 e 3 não são mutuamente excludentes;
#
#             6) há palavras que possuem função ambígua e  não há pistas para desambiguização
#
#             7) é necessário associar etiquetas que identifiquem conjunções subordinativas (KS) às palavras do léxico,
#             para melhorar sua precisão;
#
#             8) a oração introduzida por uma conjunção subordinativa nem sempre é adverbial. Por exemplo: “Não sei se
#             vou viajar” (a oração introduzida por “se” é subordinada substantiva objetiva direta e não adverbial
#             condicional, como a métrica consideraria):
#
#                 não [não] ADV @ADVL>
#                 sei [saber] <fmc> <vt> V PR 1S IND VFIN @FMV
#                 se [se] KS @SUB @#FS-<ACC
#                 vou [ir] V PR 1S IND VFIN @FAUX
#                 viajar [viajar] <vi> V INF @IMV @#ICL-AUX<
#
#             9) A precisão aumentaria se a oração fosse capturada pela etiqueta do parser Palavras que identifica orações
#             adverbiais. Ex:
#
#                 se [se] KS @SUB @#FS-ADVL>
#                 você [você] PERS M/F 3S NOM @SUBJ>
#                 vier [vir] <vi> V FUT 3S SUBJ VFIN @FMV
#                 eu [eu] PERS M/F 1S NOM @SUBJ>
#                 prometo [prometer] <fmc> <vt> V PR 1S IND VFIN @FMV
#                 que [que] KS @SUB @#FS-<ACC
#                 lhe [ele] PERS M/F 3S DAT @DAT>
#                 darei [dar] <vdt> V FUT 1S IND VFIN @FMV
#                 atenção [atenção] <am> N F S @<ACC
#
#         **Projeto**: GUTEN
#
#     """
#
#     name = 'Ratio of Adverbials Level 3 to Coordinate and Subordinate Clauses'
#     column_name = 'adverbials_03_ratio'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         lower = t.raw_content.lower()
#         expression = re.compile('''
#                                 \[a=fim=de\]|                   # final
#                                 \[a=fim=de=que\]|               # final
#                                 \[porque\]|                     # final
#                                 \[já=que\]|        # causal
#                                 \[porquanto\]|                  # causal
#                                 \[uma=vez=que\]|                # causal
#                                 \[visto=que\]|                  # causal
#                                 \[como\]|                       # causal
#                                 \[que\].*<rel>|                 # explicativa
#                                 \[onde\]|                       # explicativa
#                                 \[quando\]|                     # explicativa
#                                 \[quem\]|                       # explicativa
#                                 \[quanto\]|                     # explicativa
#                                 \[assim=que\]|                  # temporal
#                                 \[logo=que\]|                   # temporal
#                                 \[contanto=que\]|               # condicional
#                                 \[se\].*KS|                     # condicional
#                                 \[caso\].*KS|                   # condicional
#                                 \[a=menos=que\]                 # condicional
#                                 \[a=não=ser=que]|  # condicional
#                                 \[exceto=se\]|                  # condicional
#                                 \[salvo=se\]|                   # condicional
#                                 \[desde=que\]|                  # condicional
#                                 \[apesar=de=que\]|              # concessiva
#                                 \[embora\]|                     # concessiva
#                                 \[mas\]|                        # concessiva
#                                 \[conquanto\]|                  # concessiva
#                                 \[ainda=que\]|                  # concessiva
#                                 \[mesmo=que\]|                  # concessiva
#                                 \[nem=que\]|                    # concessiva
#                                 \[por=mais=que\]|               # concessiva
#                                 \[posto=que\]|                  # concessiva
#                                 \[por=muito=que\]|              # concessiva
#                                 \[de=forma=que\]|               # consecutiva
#                                 \[de=modo=que\]|                # consecutiva
#                                 \[conforme\]|                   # consecutiva
#                                 \[consoante\]|                  # consecutiva
#                                 \[segundo\]|                    # consecutiva
#                                 \[como\]                        # consecutiva
#                                 ''',
#                                 re.VERBOSE)
#         ocorrencias1 = len(re.findall(expression, flat))
#         # casos não capturados por expressão regular
#         ocorrencias2 = sum(
#             [
#                 lower.count('para que'),      # final
#                 lower.count('depois de'),     # temporal
#                 lower.count('depois que'),    # temporal
#                 lower.count('antes que'),     # condicional
#                 lower.count('sem que'),       # condicional
#                 lower.count('apesar que'),    # concessiva
#                 lower.count('tamanho que'),   # consecutiva
#                 lower.count('tal que'),       # consecutiva
#                 lower.count('tanto que'),     # consecutiva
#
#             ]
#         )
#         try:
#             return (ocorrencias1 + ocorrencias2) / rp.num_clauses(t)
#         except ZeroDivisionError:
#             return 0


# class AdverbialsLevel2(base.Metric):
#     """
#         **Nome da Métrica**: adverbials_02_ratio
#
#         **Interpretação**: orações subordinadas adverbiais estão relacionadas a maior complexidade
#
#         **Descrição da métrica**: Proporção de orações subordinadas adverbiais de nível 2 em relação à quantidade de
#         orações do texto
#
#         **Definição dos termos que aparecem na descrição da métrica**: oração adverbial é um tipo de oração subordinada.
#         Ela se encaixa na oração principal, funcionando como adjunto adverbial. Há 9 tipos de orações adverbiais:
#         causais, comparativas, concessivas, condicionais, conformativas, consecutivas, finais, proporcionais e temporais. 
#         As orações subordinadas adverbiais podem ser introduzidas por conjunções subordinativas (KS), mas nem toda
#         oração subordinada introduzida por conjunção subordinativa é adverbial. Há orações subordinadas adverbiais
#         reduzidas de gerúndio, particípio e infinitivo, as quais não estão sendo capturadas pela métrica.
#
#         **Forma de cálculo da métrica**: identificam-se as sequências lexicais definidas na lista e contam-se as
#         ocorrências. Depois divide-se o resultado pela soma de ocorrências de KS, KC e <rel>.
#
#         **Recursos de PLN utilizados durante o cálculo**: POS tagger nlpnet
#
#         **Limitações da métrica**:
#
#         **Crítica**: vários motivos justificam a recomendação de comentar esta métrica:
#
#             1) A forma de cálculo da quantidade de orações (denominador do cálculo) está errada
#
#             2) a lista de palavras que introduzem orações adverbiais precisa ser revista;
#
#             3) das 9 classes existentes, o léxico só apresenta 5
#
#             4) há uma classe definida que não existe: explicativas (que associa-se à etiqueta <rel>) Somente orações
#             subordinadas adjetivas têm uma subcategoria chamada “explicativas”;
#
#             5) as listas de nível 1, 2 e 3 não são mutuamente excludentes;
#
#             6) há palavras que possuem função ambígua e  não há pistas para desambiguização
#
#             7) é necessário associar etiquetas que identifiquem conjunções subordinativas (KS) às palavras do léxico,
#             para melhorar sua precisão;
#
#             8) a oração introduzida por uma conjunção subordinativa nem sempre é adverbial. Por exemplo: “Não sei se
#             vou viajar” (a oração introduzida por “se” é subordinada substantiva objetiva direta e não adverbial
#             condicional, como a métrica consideraria):
#
#                 não [não] ADV @ADVL>
#                 sei [saber] <fmc> <vt> V PR 1S IND VFIN @FMV
#                 se [se] KS @SUB @#FS-<ACC
#                 vou [ir] V PR 1S IND VFIN @FAUX
#                 viajar [viajar] <vi> V INF @IMV @#ICL-AUX<
#
#             9) A precisão aumentaria se a oração fosse capturada pela etiqueta do parser Palavras que identifica orações
#             adverbiais. Ex:
#
#                 se [se] KS @SUB @#FS-ADVL>
#                 você [você] PERS M/F 3S NOM @SUBJ>
#                 vier [vir] <vi> V FUT 3S SUBJ VFIN @FMV
#                 eu [eu] PERS M/F 1S NOM @SUBJ>
#                 prometo [prometer] <fmc> <vt> V PR 1S IND VFIN @FMV
#                 que [que] KS @SUB @#FS-<ACC
#                 lhe [ele] PERS M/F 3S DAT @DAT>
#                 darei [dar] <vdt> V FUT 1S IND VFIN @FMV
#                 atenção [atenção] <am> N F S @<ACC
#
#         **Projeto**: GUTEN
#     """
#
#     name = 'Ratio of Adverbials Level 2 to Coordinate and Subordinate Clauses'
#     column_name = 'adverbials_02_ratio'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         lower = t.raw_content.lower()
#         expression = re.compile('''
#                                 \[a=fim=de\]|                   # final
#                                 \[a=fim=de=que\]|               # final
#                                 \[porque\]|                     # final
#                                 \[já=que\]|                     # causal
#                                 \[porquanto\]|                  # causal
#                                 \[uma=vez=que\]|                # causal
#                                 \[visto=que\]|                  # causal
#                                 \[como\]|                       # causal
#                                 \[que\].*<rel>|                 # explicativa
#                                 \[onde\]|                       # explicativa
#                                 \[quando\]|                     # explicativa
#                                 \[quem\]|                       # explicativa
#                                 \[quanto\]|                     # explicativa
#                                 \[assim=que\]|                  # temporal
#                                 \[logo=que\]|                   # temporal
#                                 \[contanto=que\]|               # condicional
#                                 \[se\].*KS|                     # condicional
#                                 \[caso\].*KS|                   # condicional
#                                 \[a=menos=que\]                 # condicional
#                                 \[a=não=ser=que]|               # condicional
#                                 \[exceto=se\]|                  # condicional
#                                 \[salvo=se\]|                   # condicional
#                                 \[desde=que\]|                  # condicional
#                                 \[apesar=de=que\]|              # concessiva
#                                 \[embora\]|                     # concessiva
#                                 \[mas\]|                        # concessiva
#                                 \[conquanto\]|                  # concessiva
#                                 \[ainda=que\]|                  # concessiva
#                                 \[mesmo=que\]|                  # concessiva
#                                 \[nem=que\]|                    # concessiva
#                                 \[por=mais=que\]|               # concessiva
#                                 \[posto=que\]|                  # concessiva
#                                 \[por=muito=que\]               # concessiva
#                                 ''',
#                                 re.VERBOSE)
#         ocorrencias1 = len(re.findall(expression, flat))
#         # casos não capturados por expressão regular
#         ocorrencias2 = sum(
#             [
#                 lower.count('para que'),      # final
#                 lower.count('depois de'),     # temporal
#                 lower.count('depois que'),    # temporal
#                 lower.count('antes que'),     # condicional
#                 lower.count('sem que'),       # condicional
#                 lower.count('apesar que'),    # concessiva
#
#             ]
#         )
#         try:
#             return (ocorrencias1 + ocorrencias2) / rp.num_clauses(t)
#         except ZeroDivisionError:
#             return 0


# class AdverbialsLevel1(base.Metric):
#     """
#         **Nome da Métrica**: adverbials_01_ratio
#
#         **Interpretação**: orações subordinadas adverbiais estão relacionadas a maior complexidade
#
#         **Descrição da métrica**: Proporção de orações subordinadas adverbiais de nível 1 em relação à quantidade de
#         orações do texto
#
#         **Definição dos termos que aparecem na descrição da métrica**: oração adverbial é um tipo de oração subordinada.
#         Ela se encaixa na oração principal, funcionando como adjunto adverbial. Há 9 tipos de orações adverbiais:
#         causais, comparativas, concessivas, condicionais, conformativas, consecutivas, finais, proporcionais e temporais.
#         As orações subordinadas adverbiais podem ser introduzidas por conjunções subordinativas (KS), mas nem toda
#         oração subordinada introduzida por conjunção subordinativa é adverbial. Há orações subordinadas adverbiais
#         reduzidas de gerúndio, particípio e infinitivo, as quais não estão sendo capturadas pela métrica.
#
#         **Forma de cálculo da métrica**: identificam-se as sequências lexicais definidas na lista e contam-se as
#         ocorrências. Depois divide-se o resultado pela soma de ocorrências de KS, KC e <rel>.
#
#         **Recursos de PLN utilizados durante o cálculo**: POS tagger nlpnet
#
#         **Limitações da métrica**:
#
#         **Crítica**: vários motivos justificam a recomendação de comentar esta métrica:
#
#             1) A forma de cálculo da quantidade de orações (denominador do cálculo) está errada
#
#             2) a lista de palavras que introduzem orações adverbiais precisa ser revista;
#
#             3) das 9 classes existentes, o léxico só apresenta 5
#
#             4) há uma classe definida que não existe: explicativas (que associa-se à etiqueta <rel>) Somente orações
#             subordinadas adjetivas têm uma subcategoria chamada “explicativas”;
#
#             5) as listas de nível 1, 2 e 3 não são mutuamente excludentes;
#
#             6) há palavras que possuem função ambígua e  não há pistas para desambiguização
#
#             7) é necessário associar etiquetas que identifiquem conjunções subordinativas (KS) às palavras do léxico,
#             para melhorar sua precisão;
#
#             8) a oração introduzida por uma conjunção subordinativa nem sempre é adverbial. Por exemplo: “Não sei se vou
#             viajar” (a oração introduzida por “se” é subordinada substantiva objetiva direta e não adverbial
#             condicional, como a métrica consideraria):
#
#                 não [não] ADV @ADVL>
#                 sei [saber] <fmc> <vt> V PR 1S IND VFIN @FMV
#                 se [se] KS @SUB @#FS-<ACC
#                 vou [ir] V PR 1S IND VFIN @FAUX
#                 viajar [viajar] <vi> V INF @IMV @#ICL-AUX<
#
#             9) A precisão aumentaria se a oração fosse capturada pela etiqueta do parser Palavras que identifica orações
#             adverbiais. Ex:
#
#         se [se] KS @SUB @#FS-ADVL>
#         você [você] PERS M/F 3S NOM @SUBJ>
#         vier [vir] <vi> V FUT 3S SUBJ VFIN @FMV
#         eu [eu] PERS M/F 1S NOM @SUBJ>
#         prometo [prometer] <fmc> <vt> V PR 1S IND VFIN @FMV
#         que [que] KS @SUB @#FS-<ACC
#         lhe [ele] PERS M/F 3S DAT @DAT>
#         darei [dar] <vdt> V FUT 1S IND VFIN @FMV
#         atenção [atenção] <am> N F S @<ACC
#
#         **Projeto**: GUTEN
#     """
#
#     name = 'Ratio of Adverbials Level 1 to All Clauses'
#     column_name = 'adverbials_01_ratio'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         lower = t.raw_content.lower()
#         expression = re.compile('''
#                                 \[a=fim=de\]|                   # final
#                                 \[a=fim=de=que\]|               # final
#                                 \[porque\]|                     # final
#                                 \[já=que\]|                     # causal
#                                 \[porquanto\]|                  # causal
#                                 \[uma=vez=que\]|                # causal
#                                 \[visto=que\]|                  # causal
#                                 \[como\]|                       # causal
#                                 \[que\].*<rel>|                 # explicativa
#                                 \[onde\]|                       # explicativa
#                                 \[quando\]|                     # explicativa
#                                 \[quem\]|                       # explicativa
#                                 \[quanto\]|                     # explicativa
#                                 \[assim=que\]|                  # temporal
#                                 \[logo=que\]|                   # temporal
#                                 \[contanto=que\]|               # condicional
#                                 \[se\].*KS|                     # condicional
#                                 \[caso\].*KS|                   # condicional
#                                 \[a=menos=que\]                 # condicional
#                                 \[a=não=ser=que]                # condicional
#                                 \[exceto=se\]                   # condicional
#                                 \[salvo=se\]                    # condicional
#                                 \[desde=que\]                   # condicional
#                                 \[se=bem=que\]                  # condicional
#                                 ''',
#                                 re.VERBOSE)
#         ocorrencias1 = len(re.findall(expression, flat))
#         # casos não capturados por expressão regular
#         ocorrencias2 = sum(
#             [
#                 lower.count('para que'),      # final
#                 lower.count('depois de'),     # temporal
#                 lower.count('depois que'),    # temporal
#                 lower.count('antes que'),     # condicional
#                 lower.count('sem que'),       # condicional
#             ]
#         )
#         try:
#             return (ocorrencias1 + ocorrencias2) / rp.num_clauses(t)
#         except ZeroDivisionError:
#             return 0


# class DiscourseMarkersEasy(base.Metric):
#     """
#         **Nome da Métrica**: discourse_markers_easy
#
#         **Interpretação**: não está clara a relação da métrica com a complexidade textual. Se a proporção de marcadores
#         discursivos fáceis fosse calculada em relação a todos os marcadores discursivos do texto, esta métrica seria
#         inversamente proporcional à complexidade textual (quanto maior a métrica, menor a complexidade).
#
#         **Descrição da métrica**: Proporção de marcadores discursivos fáceis em relação à quantidade de orações do texto
#
#         **Definição dos termos que aparecem na descrição da métrica**: Quantidade de orações é um número obtido por meio
#         da contagem das etiquetas “V” menos as etiquetas <aux> do parser Palavras ou seja, cada verbo principal
#         representa uma oração. Marcadores discursivos fáceis são os seguintes (informados diretamente no código de
#         extração da métrica):  ainda, antes, como, e, mal, mas, nem, ora, ou, para, porque, primeiro, quando, que, se,
#         tal, também, tanto, tão
#
#         **Forma de cálculo da métrica**: contam-se os marcadores discursivos fáceis, usando a lista do próprio código
#         para identificá-los; divide-se o resultado pela quantidade de orações do texto. Recursos de PLN utilizados
#         durante o cálculo: parser Palavras para contar as orações
#
#         **Limitações da métrica**:  não prevê a ambiguidade funcional das palavras
#
#         **Crítica**: vários motivos tornam recomendável inibir (comentar) esta métrica
#
#             1) a origem da lista de marcadores discursivos não está descrita
#
#             2) a lista de marcadores discursivos contém palavras que não são marcadores discursivos ou que podem ter
#             outras funções além de marcador discursivo
#
#             3) marcas para desambiguização de marcadores discursivos, como a vírgula, não são usadas,
#
#             4) os critérios para dividir os marcadores discursivos em 4 níveis não estão explícitos;
#
#             5) palavras ambíguas funcionalmente deveriam ser suprimidas do léxico de marcadores;
#
#             6) a proporção deveria ser em relação ao total de marcadores discursivos.
#
#         **Projeto**: GUTEN
#
#         **Teste**: O médico ainda não chegou e o paciente está passando mal. Por isso, o primeiro que chegar será
#         chamado para dar assistência urgente.
#
#         **Contagens**: 2 sentenças: na primeira 1 marcador discursivo fácil (e) e 2 orações; na segunda sentença, 1
#         marcador discursivo não fácil (por isso) e 3 orações
#
#         **Resultado Esperado**: 1/5 = 0,20
#
#         **Resultado Obtido**: 1,20 (o programa identificou as palavras “ainda”, “mal” e “primeiro” “que” e “para” como
#         marcadores discursivos fáceis) e calculou 6/5.
#
#         **Status**: incorreto
#
#     """
#
#     name = '''Ratio of Easy Discourse Markers to All Clauses'''
#     column_name = 'discourse_markers_easy'
#
#     def value_for_text(self, t, rp=default_rp):
#         lower = rp.all_words(t)
#         ocorrencias = sum(
#             [
#                 lower.count('e'),
#                 lower.count('se'),
#                 lower.count('que'),
#                 lower.count('ou'),
#                 lower.count('para'),
#                 lower.count('como'),
#                 lower.count('ora'),
#                 lower.count('mas'),
#                 lower.count('tão'),
#                 lower.count('também'),
#                 lower.count('tal'),
#                 lower.count('antes'),
#                 lower.count('quando'),
#                 lower.count('ainda'),
#                 lower.count('mal'),
#                 lower.count('porque'),
#                 lower.count('nem'),
#                 lower.count('tanto'),
#                 lower.count('primeiro')
#             ]
#         )
#         ngram = list(ngrams(lower, 2))
#         ocorrencias += sum(
#             [
#                 ngram.count(('por', 'exemplo'))
#             ]
#         )
#         try:
#             return ocorrencias / rp.num_clauses(t)
#         except ZeroDivisionError:
#             return 0


# class DiscourseMarkersMedium(base.Metric):
#     """
#         **Nome da Métrica**: discourse_markers_medium
#
#         **Interpretação**: não está clara a relação da métrica com a complexidade textual.
#
#         **Descrição da métrica**: Proporção de marcadores discursivos medianos em relação à quantidade de orações do
#         texto
#
#         **Definição dos termos que aparecem na descrição da métrica**: Quantidade de orações é um número obtido por meio
#         da contagem das etiquetas “V” menos as etiquetas <aux> do parser Palavras ou seja, cada verbo principal
#         representa uma oração. Marcadores discursivos medianos são os seguintes (informados diretamente no código de
#         extração da métrica): logo, caso, assim, então, após, porém, enquanto, ou seja, por isso, para que, antes de,
#         já que, quer dizer;
#
#         **Forma de cálculo da métrica**: contam-se os marcadores discursivos medianos, usando a lista do próprio código
#         para identificá-los; divide-se o resultado pela quantidade de orações do texto.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras para contar as orações
#
#         **Limitações da métrica**:  não prevê a ambiguidade funcional das palavras
#
#         **Crítica**: vários motivos tornam recomendável inibir (comentar) esta métrica:
#
#             1) a origem da lista de marcadores discursivos não está descrita;
#
#             2) a lista de marcadores discursivos contém palavras que não são marcadores discursivos ou que podem ter
#             outras funções além de marcador discursivo;
#
#             3) marcas para desambiguização de marcadores discursivos, como a vírgula, não são usadas;
#
#             4) os critérios para dividir os marcadores discursivos em 4 níveis não estão explícitos;
#
#             5) palavras ambíguas funcionalmente deveriam ser suprimidas do léxico de marcadores;
#
#             6) a proporção deveria ser em relação ao total de marcadores discursivos.
#
#         **Projeto**: GUTEN
#
#         **Teste**: O caso é sério e é preciso operar logo após o meio-dia, assim que o médico chegar.
#
#         **Contagens**: 4 orações, 1 marcador discursivo fácil (e), nenhum mediano
#
#         **Resultado Esperado**: zero
#
#         **Resultado Obtido**: 1 (o programa identificou “caso”, “logo”, “após”, “assim” como marcadores discursivos,
#         mas essas palavras não têm essa função nessa sentença)
#
#         **Status**: incorreto
#     """
#
#     name = 'Ratio of Medium Discourse Markers to All Clauses'
#     column_name = 'discourse_markers_medium'
#
#     def value_for_text(self, t, rp=default_rp):
#         lower = t.raw_content.lower()
#         ocorrencias = sum(
#             [
#                 lower.count('logo'),
#                 lower.count('caso'),
#                 lower.count('assim'),
#                 lower.count('então'),
#                 lower.count('após'),
#                 lower.count('porém'),
#                 lower.count('enquanto')
#             ]
#         )
#         ngram = list(ngrams(lower, 2))
#         ocorrencias += sum(
#             [
#                 ngram.count(('ou', 'seja')),
#                 ngram.count(('por', 'isso')),
#                 ngram.count(('para', 'que')),
#                 ngram.count(('antes', 'de')),
#                 ngram.count(('já', 'que')),
#                 ngram.count(('quer', 'dizer'))
#             ]
#         )
#         try:
#             return ocorrencias / rp.num_clauses(t)
#         except ZeroDivisionError:
#             return 0


# class DiscourseMarkersHard(base.Metric):
#     """
#         **Nome da Métrica**: discourse_markers_hard
#
#         **Interpretação**: não está clara a relação da métrica com a complexidade textual.
#
#         **Descrição da métrica**: Proporção de marcadores discursivos de difíceis em relação à quantidade de orações do
#         texto
#
#         **Definição dos termos que aparecem na descrição da métrica**: quantidade de orações é um número obtido por meio
#         da contagem das etiquetas “V” menos as etiquetas <aux> do parser Palavras ou seja, cada verbo principal
#         representa uma oração. Marcadores discursivos difíceis são os seguintes (informados diretamente no código de
#         extração da métrica): primeiramente, a ilustrar, por esse motivo, como vimos, de tal forma, efetivamente, na
#         realidade, em consequência, tampouco, por um lado, em segundo lugar, a não ser que, por último, embora, mas
#         também, até que, na verdade, assim como, acima, daí, portanto, no entanto, não só, apesar de, mesmo que, em
#         seguida, ao mesmo tempo, isto é, de fato, a seguir, assim que, mais tarde, em relação ao, bem como, depois que,
#         naturalmente, ainda que, desde que, de modo, certamente, sobretudo, com isso, uma vez que, mesmo assim, ou
#         melhor, repare-se, a fim de, igualmente, menos do que, entretanto, para começar, de forma a, contudo, com o
#         objetivo de, ainda assim, por outro lado, dado que, mais ainda, além disso, tal como, para concluir, logo que,
#         pelo contrário, é o caso de, em primeiro lugar.
#
#         **Forma de cálculo da métrica**: contam-se os marcadores discursivos difíceis, usando a lista do próprio código
#         para identificá-los; divide-se o resultado pela quantidade de orações do texto.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras para contar as orações
#
#         **Limitações da métrica**:  não prevê a ambiguidade funcional das palavras
#
#         **Crítica**: vários motivos tornam recomendável inibir (comentar) esta métrica
#
#             1) a origem da lista de marcadores discursivos não está descrita
#
#             2) a lista de marcadores discursivos contém palavras que não são marcadores discursivos ou que podem ter
#             outras funções além de marcador discursivo
#
#             3) marcas para desambiguização de marcadores discursivos, como a vírgula, não são usadas,
#
#             4) os critérios para dividir os marcadores discursivos em 4 níveis não estão explícitos
#
#             5) marcadores discursivos muito ambíguos e sem marcas de desambiguização deveriam ser suprimidos;
#
#             6) a proporção deveria ser em relação ao total de marcadores discursivos e não por número de orações.
#
#         **Projeto**: GUTEN
#
#         **Teste**: Vamos embora tão logo tenhamos resolvido essas pendências, apesar de estarmos adorando a estadia no
#         hotel.
#
#         **Contagens**: 3 orações e 1 marcador discursivo (apesar de)
#
#         **Resultado Esperado**: 1/3 = 0,33
#
#         **Resultado Obtido**: 0,667 (a palavra “embora” foi indevidamente reconhecida como marcador discursivo)
#
#         **Status**: incorreto
#     """
#
#     name = 'Ratio of Hard Discourse Markers to All Clauses'
#     column_name = 'discourse_markers_hard'
#
#     def value_for_text(self, t, rp=default_rp):
#         lower = t.raw_content.lower()
#         ocorrencias = sum(
#             [
#                 lower.count('embora'),
#                 lower.count('mas também'),
#                 lower.count('até que'),
#                 lower.count('na verdade'),
#                 lower.count('assim como'),
#                 lower.count('acima'),
#                 lower.count('daí'),
#                 lower.count('portanto'),
#                 lower.count('no entanto'),
#                 lower.count('não só'),
#                 lower.count('apesar de'),
#                 lower.count('mesmo que'),
#                 lower.count('em seguida'),
#                 lower.count('ao mesmo tempo'),
#                 lower.count('isto é'),
#                 lower.count('de fato'),
#                 lower.count('a seguir'),
#                 lower.count('assim que'),
#                 lower.count('mais tarde'),
#                 lower.count('em relação ao'),
#                 lower.count('bem como'),
#                 lower.count('depois que'),
#                 lower.count('naturalmente'),
#                 lower.count('ainda que'),
#                 lower.count('desde que'),
#                 lower.count('desde que'),
#                 lower.count('de modo'),
#                 lower.count('certamente'),
#                 lower.count('sobretudo'),
#                 lower.count('com isso'),
#                 lower.count('uma vez que'),
#                 lower.count('mesmo assim'),
#                 lower.count('ou melhor'),
#                 lower.count('repare-se'),
#                 lower.count('a fim de'),
#                 lower.count('igualmente'),
#                 lower.count('menos do que'),
#                 lower.count('entretanto'),
#                 lower.count('para começar'),
#                 lower.count('de forma a'),
#                 lower.count('contudo'),
#                 lower.count('com o objetivo de'),
#                 lower.count('ainda assim'),
#                 lower.count('por outro lado'),
#                 lower.count('dado que'),
#                 lower.count('mais ainda'),
#                 lower.count('além disso'),
#                 lower.count('tal como'),
#                 lower.count('para concluir'),
#                 lower.count('pelo contrário'),
#                 lower.count('logo que'),
#                 lower.count('é o caso de'),
#                 lower.count('em primeiro lugar'),
#                 lower.count('por último'),
#                 lower.count('a não ser que'),
#                 lower.count('em segundo lugar'),
#                 lower.count('por um lado'),
#                 lower.count('tampouco'),
#                 lower.count('em consequência'),
#                 lower.count('na realidade'),
#                 lower.count('efetivamente'),
#                 lower.count('de tal forma'),
#                 lower.count('como vimos'),
#                 lower.count('por esse motivo'),
#                 lower.count('a ilustrar'),
#                 lower.count('primeiramente')
#             ]
#         )
#         try:
#             return ocorrencias / rp.num_clauses(t)
#         except ZeroDivisionError:
#             return 0


# class DiscourseMarkersVeryHard(base.Metric):
#     """
#         **Nome da Métrica**: discourse_markers_veryhard
#
#         **Interpretação**: quanto maior a proporção, maior a complexidade
#
#         **Descrição da métrica**: Proporção de marcadores discursivos muito difíceis em relação à quantidade de orações
#         do texto
#
#         **Definição dos termos que aparecem na descrição da métrica**: quantidade de orações é um número obtido por meio
#         da contagem das etiquetas “V” menos as etiquetas <aux> do parser Palavras ou seja, cada verbo principal
#         representa uma oração. Marcadores discursivos muito difíceis são os seguintes (informados diretamente no código
#         de extração da métrica): com efeito, daí que, em terceiro lugar, concluindo, paralelamente, ou antes, nesse
#         contexto, em contrapartida, visto que, como se pode ver, seguidamente, em resumo, por ora, pretendemos,
#         evidentemente, é evidente que, contanto que, além de tudo, com o propósito de, por conseguinte, prosseguindo,
#         em alternativa, documentando, a fim de que, para encerrar, por este motivo, note-se que, com intuito de, nesse
#         cenário, todavia, mais concretamente, contrariamente, pelo que referi anteriormente, em nosso entender, dizendo
#         melhor, opcionalmente, concomitantemente, com toda a certeza, conquanto, significa isto que, quer isto dizer,
#         decerto, recapitulando, exemplificando, não se pense que, em vias de, por outras palavras, estou em crer que,
#         sintetizando, não obstante, em conclusão, a meu ver, com vistas a, alternativamente, deste modo, malgrado, com
#         isto, veja-se, em suma.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras para contar as orações
#
#         **Limitações da métrica**:  não prevê a ambiguidade funcional das palavras
#
#         **Crítica**: é recomendável comentar esta métrica por uma série de motivos:
#
#             1) a origem da lista de marcadores discursivos não está descrita
#
#             2) a lista de marcadores discursivos contém palavras e sequências de palavras que não são marcadores
#             discursivos ou que podem ter outras funções além de marcador discursivo
#
#             3) marcas para desambiguização de marcadores discursivos, como a vírgula, não são usadas,
#
#             4) os critérios para dividir os marcadores discursivos em 4 níveis não estão explícitos;
#
#             5) palavras ambíguas funcionalmente deveriam ser suprimidas do léxico de marcadores.
#
#         **Projeto**: GUTEN
#
#         **Teste**: É importante esclarecer o assunto para que não se pense que somos contrários à greve.
#
#         **Contagens**: nenhum marcador discursivo, 4 orações
#
#         **Resultado Esperado**: 0
#
#         **Resultado Obtido**: 0,25 (o extrator localizou “não se pense que” e contou como marcador)
#
#         **Status**: incorreto
#     """
#
#     name = 'Ratio of Very Hard Discourse Markers to All Clauses'
#     column_name = 'discourse_markers_veryhard'
#
#     def value_for_text(self, t, rp=default_rp):
#         lower = t.raw_content.lower()
#         ocorrencias = sum(
#             [
#                 lower.count('com efeito'),
#                 lower.count('daí que'),
#                 lower.count('em terceiro lugar'),
#                 lower.count('concluindo'),
#                 lower.count('paralelamente'),
#                 lower.count('ou antes'),
#                 lower.count('nesse contexto'),
#                 lower.count('em contrapartida'),
#                 lower.count('visto que'),
#                 lower.count('como se pode ver'),
#                 lower.count('seguidamente'),
#                 lower.count('em resumo'),
#                 lower.count('por ora'),
#                 lower.count('pretendemos'),
#                 lower.count('evidentemente'),
#                 lower.count('é evidente que'),
#                 lower.count('contanto que'),
#                 lower.count('além de tudo'),
#                 lower.count('com o propósito de'),
#                 lower.count('por conseguinte'),
#                 lower.count('prosseguindo'),
#                 lower.count('em alternativa'),
#                 lower.count('documentando'),
#                 lower.count('a fim de que'),
#                 lower.count('para encerrar'),
#                 lower.count('por este motivo'),
#                 lower.count('note-se que'),
#                 lower.count('com intuito de'),
#                 lower.count('nesse cenário'),
#                 lower.count('todavia'),
#                 lower.count('mais concretamente'),
#                 lower.count('contrariamente'),
#                 lower.count('pelo que referi anteriormente'),
#                 lower.count('em nosso entender'),
#                 lower.count('dizendo melhor'),
#                 lower.count('opcionalmente'),
#                 lower.count('concomitantemente'),
#                 lower.count('com toda a certeza'),
#                 lower.count('conquanto'),
#                 lower.count('significa isto que'),
#                 lower.count('quer isto dizer'),
#                 lower.count('decerto'),
#                 lower.count('recapitulando'),
#                 lower.count('exemplificando'),
#                 lower.count('não se pense que'),
#                 lower.count('em vias de'),
#                 lower.count('por outras palavras'),
#                 lower.count('estou em crer que'),
#                 lower.count('sintetizando'),
#                 lower.count('não obstante'),
#                 lower.count('em conclusão'),
#                 lower.count('a meu ver'),
#                 lower.count('com vistas a'),
#                 lower.count('alternativamente'),
#                 lower.count('deste modo'),
#                 lower.count('malgrado'),
#                 lower.count('com isto'),
#                 lower.count('veja-se'),
#                 lower.count('em suma')
#             ]
#         )
#         try:
#             return ocorrencias / rp.num_clauses(t)
#         except ZeroDivisionError:
#             return 0


class NotSVO(base.Metric):
    """
        **Nome da Métrica**: non_svo_ratio

        **Interpretação**: quanto maior o resultado da métrica, maior a complexidade.

        **Descrição da métrica**: Proporção de orações que não estão no formato SVO (sujeito-verbo-objeto) em relação a
        todas orações do texto

        **Definição dos termos que aparecem na descrição da métrica**: a ordem SVO é a ordem canônica ou “natural” dos
        constituintes na língua portuguesa (ex: Ela adotou um menino). Ordens não SVO são: OSV (ex: Greve eu não farei),
        OVS (ex: dinheiro só ganharão os bons profissionais), VS (ex: Acabou o prazo), VSO (ex: Ouviram do Ipiranga as
        margens plácidas, de um povo heroico o brado retumbante), VOS (ex: Fez vinte e oito pontos a melhor aluna).

        **Forma de cálculo da métrica**:

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**:

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: Ouviram as margens plácidas o brado retumbante. Acabou o prazo de análise, mas nós ainda podemos
        pedir prorrogação.

        **Contagens**: 3 orações (ouviram..., acabou..., podemos pedir...), a primeira na ordem VSO, a segunda na ordem
        VS e a terceira na ordem SVO, portanto, 2 orações em ordens não SVO.

        **Resultado Esperado**: 2/3 = 0,667

        **Resultado Obtido**: 0,667

        **Status**: correto
    """

    name = 'Ratio of not SVO clauses to all clauses'
    column_name = 'non_svo_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        lines = flat.split('\n')
        count = 0
        found = False
        for line in lines:
            if ('@ADVL>' in line or 'APP' in line or '<SUBJ' in line) and not found:
                found = True
                count += 1
            if ' V ' in line and '<aux>' not in line:
                found = False
        nverbs = flat.count(' V ') - flat.count('<aux>')
        try:
            return count / nverbs
        except ZeroDivisionError:
            return 0


# class DiscourseVoices(base.Metric):
#     """
#         **Nome da Métrica**: discourse_voices_ratio
#
#         **Interpretação**: não está clara a relação da métrica com a complexidade textual
#
#         **Descrição da métrica**: Proporção de palavras que introduzem vozes no discurso em relação à quantidade de
#         palavras do texto
#
#         **Definição dos termos que aparecem na descrição da métrica**: de acordo com a concepção da métrica, palavras
#         que introduzem vozes no discurso são: “de=acordo”, “segundo”, “diz que”, “disse que”, “disseram”, “afirma”,
#         “afirmam”
#
#         **Forma de cálculo da métrica**: contam-se as ocorrências das sequências lexicais definidas como introdutoras
#         de vozes no discurso e divide-se pelo total de palavras do texto
#
#         **Recursos de PLN utilizados durante o cálculo**: POS tagger nlpnet
#
#         **Limitações da métrica**: por ser baseada em léxico, tem muita ambiguidade e pode ser falsamente inflada
#
#         **Crítica**: o que é definido como “vozes do discurso” são formas de introduzir discurso indireto. A lista de
#         verbos de elocução é muito limitada. É preciso estudo para aumentar o léxico e para diminuir os riscos de
#         capturar as mesmas palavras com outras funções além dessa.
#
#         **Projeto**: GUTEN
#
#         **Teste**: Foi segundo colocado na competição e afirma nunca ter usado esteroides.
#
#         **Contagens**: 11 palavras, 1 voz no discurso
#
#         **Resultado Esperado**: 1/11 = 0,09
#
#         **Resultado Obtido**: 0,25
#
#         **Status**: incorreto
#
#     """
#
#     name = 'Ratio of Discourse Voices to all Words'
#     column_name = 'discourse_voices_ratio'
#
#     def value_for_text(self, t, rp=default_rp):
#         lower = t.raw_content.lower()
#         flat = rp.palavras_flat(t)
#         lines = flat.split('\n')
#         ocorrencias = 0
#         words = 0
#         expression = re.compile('''
#                                 de=acordo|
#                                 afirma|
#                                 \[segundo\]\ <com>
#                                 ''',
#                                 re.VERBOSE)
#         for line in lines:
#             if len(line) > 10:
#                 words += 1
#                 ocorrencias += len(re.findall(expression, line))
#         ocorrencias2 = sum(
#             [
#                 lower.count('diz que'),
#                 lower.count('disse que'),
#                 lower.count('disseram'),
#                 lower.count('afirma'),
#                 lower.count('afirmam'),
#
#             ]
#         )
#         try:
#             return (ocorrencias + ocorrencias2) / words
#         except ZeroDivisionError:
#             return 0


class TopicalizedClausesRatio(base.Metric):
    """
        **Nome da Métrica**: adverbs_before_main_verb_ratio

        **Interpretação**: quanto maior o resultado da métrica (o máximo é 1), maior a complexidade textual, pois
        representa carga de trabalho de leitura, ou seja, aquilo que se acumula na memória antes da chegada do verbo
        principal.

        **Descrição da métrica**: Proporção de orações com advérbio antes do verbo principal em relação à quantidade de
        orações do texto

        **Definição dos termos que aparecem na descrição da métrica**: advérbios antes de verbos principais são
        ocorrências que rompem a ordem canônica dos constituintes das sentenças (Sujeito, Verbo, Objeto, Advérbio).

        **Forma de cálculo da métrica**: verificam-se quantas orações possuem ocorrências de adjuntos adverbiais
        antepostos aos verbos (@ADVL>), não importando a quantidade de advérbios antepostos. Divide-se o resultado do
        contador pela quantidade de orações do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**:

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: Gradativamente, ele foi se acostumando às novas condições de trabalho, porém, praticamente não se
        conformou até hoje com a perda de status. Hoje é fácil perceber isso, no entanto, naquela época, dificilmente
        alguém poderia saber que ele estava sofrendo profundamente.

        **Contagens**: 6 orações, 4 com advérbios antes dos verbos principais

        **Resultado Esperado**: 4/6 = 0,667

        **Resultado Obtido**: 0,667

        **Status**: correto

    """

    name = 'Ratio of Adverbs Before Main Verb'
    column_name = 'adverbs_before_main_verb_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        lines = flat.split('\n')
        count = 0
        advl = False
        for line in lines:
            if '@ADVL>' in line and not advl:
                advl = True
                count += 1
            if ' V ' in line and '<aux>' not in line:
                advl = False
        nverbs = flat.count(' V ') - flat.count('<aux>')
        try:
            return count / nverbs
        except ZeroDivisionError:
            return 0


# class AdverbsBeforeMainVerb(base.Metric):
#     """
#         **Nome da Métrica**: adverbs_before_main_verb
#
#         **Interpretação**: quanto maior o resultado da métrica, maior a complexidade textual
#
#         **Descrição da métrica**: Proporção entre a quantidade de advérbios antes de verbos principais e a quantidade
#         de verbos do texto.
#
#         **Definição dos termos que aparecem na descrição da métrica**: advérbios antes de verbos principais são
#         ocorrências que rompem a ordem canônica dos constituintes das sentenças (Sujeito, Verbo, Objeto, Advérbio).
#
#         **Forma de cálculo da métrica**: verificam-se as ocorrências de adjuntos adverbiais antepostos ao verbo (@ADVL>).
#          Somam-se todas as ocorrências e divide-se o resultado pela quantidade de orações do texto (V – aux).
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras
#
#         **Limitações da métrica**: o parser Palavras anota conjunções adversativas como advérbios antepostos ao verbo, o
#         que pode levar a uma superestimativa da métrica.
#
#         **Crítica**: Esta métrica é muito semelhante à adverbs_before_main_verb_ratio e pode ser comentada para não
#         gerar redundância.
#
#         **Projeto**: GUTEN
#
#         **Teste**: Gradativamente, ele foi se acostumando às novas condições de trabalho, porém, praticamente não se
#         conformou até hoje com a perda de status. Hoje é fácil perceber isso, no entanto, naquela época, dificilmente
#         alguém poderia saber que ele estava sofrendo profundamente.
#
#         **Contagens**: 6 orações, 6 advérbios antes dos verbos principais (gradativamente, praticamente, não, hoje,
#         naquela época, dificilmente)
#
#         **Resultado Esperado**: 6/6 = 0,667
#
#         **Resultado Obtido**: 0,1,333 (8/6, pois o parser anotou as conjunções adversativas “porém” e “no entanto”
#         com a etiqueta @ADVL)
#
#         **Status**: correto, considerando a limitação do parser
#     """
#
#     name = 'Adverbs Before Main Verb'
#     column_name = 'adverbs_before_main_verb'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         adverbs = re.findall('@ADVL>', flat)    # advérbios antes do verbo
#         nverbs = flat.count(' V ') - flat.count('<aux>')
#         try:
#             return len(adverbs) / nverbs
#         except ZeroDivisionError:
#             return 0
#

class PostponedSubject(base.Metric):
    """
        **Nome da Métrica**: postponed_subject_ratio

        **Interpretação**: sujeito posposto (VS) é uma ordem de constituintes mais complexa que sujeito anteposto (SV)

        **Descrição da métrica**: Proporção de sujeitos pospostos em relação a todos os sujeitos do texto

        **Definição dos termos que aparecem na descrição da métrica**: sujeito posposto é o sujeito que ocorre após o
        verbo.

        **Forma de cálculo da métrica**: contam-se todas as etiquetas <SUBJ (sujeito após verbo principal) do texto e
        divide-se o resultado pela soma de todas as etiquetas SUBJ (sujeitos) do texto

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**:

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: São tomadas muitas iniciativas a fim de melhorar a situação da educação no Brasil, porém são poucas
        as que dão resultado.

        **Contagens**: 4 orações (são tomadas, melhorar, são, dão), 3 sujeitos (muitas iniciativas, as que dão
        resultado, que) 2 sujeitos pospostos

        **Resultado Esperado**: 2/3 = 0,667

        **Resultado Obtido**: 0,667

        **Status**: correto
    """

    name = 'Postponed Subject Ratio'
    column_name = 'postponed_subject_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        pospostos = re.findall('<SUBJ', flat)
        sujeitos = re.findall('SUBJ', flat)
        try:
            return len(pospostos) / len(sujeitos)
        except ZeroDivisionError:
            return 0


class TemporalAdjunctRatio(base.Metric):
    """
        **Nome da Métrica**: temporal_adjunct_ratio

        **Interpretação**: não está clara a relação entre a métrica e a complexidade

        **Descrição da métrica**: Proporção de adjuntos adverbiais de tempo em relação a todos os adjuntos adverbiais
        do texto

        **Definição dos termos que aparecem na descrição da métrica**: adjuntos adverbiais de tempo são palavras ou
        expressões que informam quando a ação do verbo aconteceu (ex: ontem, de vez em quando, frequentemente, no dia
        18 de novembro, etc.). Para identificá-las, utilizam-se as expressões regulares criadas por Baptista
        et. Al. (2008).

        **Forma de cálculo da métrica**: identificam-se todos os adjuntos adverbiais (etiqueta ADVL do nlpnet).
        Filtram-se os adjuntos adverbiais de tempo, utilizando o léxico e as expressões regulares do
        rp.temporal_expressions. Divide-se a quantidade de adjuntos adverbiais de tempo pela quantidade total de
        adjuntos adverbiais.

        **Recursos de PLN utilizados durante o cálculo**: POS tagger nlpnet

        **Limitações da métrica**: a precisão da métrica depende do desempenho do tagger nlpnet.

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: Foi durante meus experimentos que eu me machuquei. Certamente cometi um erro.

        **Contagens**: 2 adjuntos adverbiais, um dos quais é de tempo (durante meus experimentos)

        **Resultado Esperado**: 1/2 = 0,50

        **Resultado Obtido**: 0,50

        **Status**: correto
    """

    name = 'Ratio of Temporal Adjuncts to All Adjuncts'
    column_name = 'temporal_adjunct_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        adjuncts = re.findall('ADVL', flat)
        temporals = rp.temporal_expressions(t)
        try:
            return len(temporals) / len(adjuncts)
        except ZeroDivisionError:
            return 0


class RelativePronouns(base.Metric):
    """
        **Nome da Métrica**: relative_pronouns_ratio

        **Interpretação**: pronomes relativos introduzem orações subordinadas adjetivas, substantivas e adverbiais, que
        expandem o conteúdo de um sintagma nominal e aumentam a complexidade textual

        **Descrição da métrica**: Proporção de Pronomes Relativos em relação à quantidade de pronomes do texto

        **Definição dos termos que aparecem na descrição da métrica**: pronomes relativos retomam e qualificam um nome
        que os antecedem. Suas formas são: que, o que, o qual, os quais, a qual, as quais, cujo, cujos, cuja, cujas,
        quem, quando, onde, como, quanto, quantos, quanta, quantas. Muitos deles são ambíguos funcionalmente (podem
        atuar como outro tipo de pronome), por isso a etiqueta <rel> é importante para capturar apenas aqueles com
        função relativa.

        **Forma de cálculo da métrica**: contam-se todas as ocorrências da etiqueta <rel> e divide-se o resultado pela
        soma de todas as ocorrências de pronomes do texto.

        **Recursos de PLN utilizados durante o cálculo**:

        **Limitações da métrica**:

        **Crítica**:

        embora a classe de pronomes seja imediatamente superior à  de pronomes relativos, a métrica poderia ser um
        índice melhor se fosse calculada por sintagma nominal.

        O POS tagger usado não reconhece multipalavras, como “o=que”, por exemplo. Isso faz com que a métrica possa ser
        superestimada em alguns casos, mas não é um problema grave.

        **Projeto**: GUTEN

        **Teste**: Regressando de São Paulo, visitei o sítio de minha tia, o qual me deixou encantado. Era exatamente o
        que eu esperava, apesar de nunca ter imaginado que eu estaria ali.

        **Contagens**: 7 pronomes (minha, o qual, me, o que, eu, que, eu), 9 contados pelo POS tagger nlpnet, dos quais
        2 são pronomes relativos (o qual, o que)

        **Resultado Esperado**: 2/7 = 0,285 ou 2/9 (o sistema identifica dois pronomes em “o que” e em “o qual”

        **Resultado Obtido**: 0,222

        **Status**: correto

    """

    name = 'Ratio of Relative Pronouns to all Pronouns'
    column_name = 'relative_pronouns_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        casos = re.findall('\[.*\].*<rel>', flat)
        if casos:
            relativos = [re.search('\[(.*)\].*<rel>', i).group(1) for i in casos]
            pronouns = filter(pos_tagger.tagset.is_pronoun, rp.tagged_words(t))
            try:
                return len(relativos) / len(list(pronouns))
            except ZeroDivisionError:
                return 0
        else:
            return 0


class RelativePronounsDiversity(base.Metric):
    """
        **Nome da Métrica**: relative_pronouns_diversity_ratio

        **Interpretação**: pronomes relativos introduzem orações subordinadas adjetivas, substantivas e adverbiais, que
        expandem o conteúdo de um sintagma nominal e aumentam a complexidade textual. Sua forma mais simples é o “que”.
        Se houver uso de outros pronomes relativos, a complexidade textual aumenta.

        **Descrição da métrica**: Proporção de types de pronomes relativos  em relação à quantidade de tokens de
        pronomes relativos no texto

        **Definição dos termos que aparecem na descrição da métrica**: pronomes relativos retomam e qualificam um nome
        que os antecedem. Suas formas são: que, o que, o qual, os quais, a qual, as quais, cujo, cujos, cuja, cujas,
        quem, quando, onde, como, quanto, quantos, quanta, quantas. Muitos deles são ambíguos funcionalmente (podem
        atuar como outro tipo de pronome), por isso a etiqueta <rel> é importante para capturar apenas aqueles com
        função relativa.

        **Forma de cálculo da métrica**: contam-se todas as ocorrências da etiqueta <rel>, sem considerar repetições;
        divide-se o resultado pela soma de todas as ocorrências da etiqueta <rel> do texto, considerando as repetições.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**:

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: A escola na qual estudo é muito rigorosa, mas os professores que dão aula para mim são muito bons. A
        professora de gramática, a qual dá aula para mim desde o sexto ano, tem uma didática fantástica.

        **Contagens**: 3 pronomes relativos ((em) a qual, que, a qual ), dos quais 2 sem considerar repetições.

        **Resultado Esperado**: 2/3 = 0,667

        **Resultado Obtido**: 0,667

        **Status**: correto
    """

    name = 'Relative Pronouns Diversity'
    column_name = 'relative_pronouns_diversity_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        casos = re.findall('\[.*\].*<rel>', flat)
        if casos:
            relativos = [re.search('\[(.*)\].*<rel>', i).group(1) for i in casos]
            # unique = len(set(relativos))
            try:
                return rp.mattr(relativos)
                # return unique / len(relativos)
            except ZeroDivisionError:
                return 0
        else:
            return 0


# class RelativeSubordinateClauses(base.Metric):
#     """
#         **Nome da Métrica**: relative_subordinate_clauses
#
#         **Interpretação**: quanto maior a proporção, maior a complexidade
#
#         **Descrição da métrica**: Proporção de orações relativas em relação a todas as orações do texto
#
#         **Definição dos termos que aparecem na descrição da métrica**: orações relativas são orações subordinadas
#         adjetivas introduzidas por pronomes relativos: que, o qual, a qual, os quais, as quais, cujo, cuja, cujos,
#         cujas.
#
#         **Forma de cálculo da métrica**: contam-se as ocorrências da etiqueta <rel> e divide-se o resultado pela
#         quantidade de orações do texto (V – aux)
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras.
#
#         **Limitações da métrica**: a precisão dos valores depende diretamente do desempenho do parser Palavras
#
#         **Crítica**: Esta métrica foi corrigida, pois anteriormente o divisor era a quantidade de conjunções
#         coordenativas e subordinativas, tomadas como índices de orações coordenadas e subordinadas, o que foi
#         constatado ser um erro. Com a correção, a métrica passou a usar a quantidade de orações como divisor, mas isso
#         a tornou idêntica a outra métrica, relative_clauses, do AIC, motivo pelo qual deve ser comentada.
#
#         **Projeto**: GUTEN
#
#         **Teste**: A mulher que eu contratei, cujo pai eu já conhecia, é aquela que você viu no dia em que chegou aqui.
#          Não sei qual é seu interesse nela, mas afirmo que, seja ele qual for, não me interessa. Os assuntos nos quais
#          estou interessado não são de ordem pessoal.
#
#         **Contagens**: 13 orações e 6 pronomes relativos: que, cujo, que, que, qual, quais (qual=for é anotado como
#         expressão)
#
#         **Resultado Esperado**: 0,462
#
#         **Resultado Obtido**: 0,462
#
#         **Status**: correto
#     """
#
#     name = '''Ratio of Relative Subordinate Clauses per number of Clauses'''
#     column_name = 'relative_subordinate_clauses'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         adjectives = len(re.findall('<rel>', flat))
#         try:
#             return adjectives / rp.num_clauses(t)
#         except ZeroDivisionError:
#             return 0


# class CoordinateClausesPredominance(base.Metric):
#     """
#         **Nome da Métrica**: coordinate_clauses_predominance
#
#         **Interpretação**: as orações coordenadas são mais simples que as subordinadas; portanto, se houver
#         predominância de coordenadas a complexidade textual é menor do que se não houver
#
#         **Descrição da métrica**: proporção de orações coordenadas em relação a todas as orações coordenadas,
#         subordinadas e relativas.
#
#         **Definição dos termos que aparecem na descrição da métrica**: orações coordenadas são orações independentes;
#         orações subordinadas são dependentes.
#
#         **Forma de cálculo da métrica**: somam-se as ocorrências de conjunções coordenativas (tag ‘KC’) e divide-se o
#         resultado pela soma de conjunções coordenativas (tag: ‘KC’), conjunções subordinativas (tag ‘KS’) e orações
#         relativas (tag <rel>)
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras
#
#         **Limitações da métrica**: após extenso estudo, inclusive envolvendo o criador do parser, Eckhard Bick,
#         descobriu-se que não há algoritmo conhecido para capturar as orações coordenadas, portanto esta métrica não é
#         possível de ser calculada com os recursos disponíveis atualmente.
#
#         **Crítica**: a métrica deve ser comentada devido aos seguintes motivos:
#
#             1) A forma de calcular orações deveria ser como nas outras métricas (V – <aux>)
#
#             2) As etiquetas utilizadas para o cálculo são inapropriadas. As conjunções coordenativas (KC) coordenam
#             tanto sintagmas nominais quanto orações, portanto, utilizar sua quantidade como reflexo da quantidade de
#             orações coordenadas é errado. Já as conjunções subordinativas (KS) introduzem orações subordinadas, mas
#             apenas um conjunto delas. As subordinadas relativas estão sendo citadas como se não fossem um tipo de
#             subordinadas e as subordinadas reduzidas de gerúndio, particípio e infinitivo não estão sendo computadas.
#
#             3) A captura das coordenadas via conjunção coordenativa deveria multiplicar o resultado por dois, pois cada
#             conjunção coordena 2 orações. Isso não ocorre nas orações subordinadas, pois apenas uma das orações é
#             subordinada e a outra, a subordinadora, é chamada de oração principal.
#
#             4) As orações coordenadas assindéticas (separadas por vírgulas) não estão sendo computadas
#
#             5) A métrica não está retornando 0 ou 1 dependendo de haver ou não predominância.
#
#         **Projeto**: GUTEN
#
#         **Teste**: Aldo e José estudaram para a prova, assistindo vídeo-aulas.
#
#         **Contagens**: 2 orações, 1 subordinada reduzida de gerúndio, nenhuma coordenada.
#
#         **Resultado Esperado**: 0
#
#         **Resultado Obtido**: 0,50 (o classificador contou o “e” como evidência de oração coordenada e não identificou a
#         oração subordinada porque ela não é introduzida por conjunção subordinativa)
#
#         **Status**: incorreto
#     """
#
#     name = 'Ratio of Coordinate Clauses to Coordinate and Subordinate Ones'
#     column_name = 'coordinate_clauses_predominance'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         coord_clauses = re.findall(r"<co-(vfin|ger|inf|pcv)>.*KC", flat)
#         try:
#             return len(coord_clauses) / rp.num_clauses(t)
#         except ZeroDivisionError:
#             return 0


# class Verbs3DegreeRatio(base.Metric):
#     """
#         **Nome da Métrica**: most_frequent_verbs_of_3-degree
#
#         **Interpretação**: em sua concepção, quanto maior a proporção, menor a complexidade
#
#         **Descrição da métrica**: Proporção de verbos simples em relação a todos os verbos do texto
#
#         **Definição dos termos que aparecem na descrição da métrica**: verbos simples ou verbos mais frequentes do nível
#         1 ou verbos mais frequentes do 3º ano são: ser, ir, ter, poder, estar, ficar, dever, precisar, haver, viver,
#         passar. Não há comentário sobre como essa lista de verbos foi levantada.
#
#         **Forma de cálculo da métrica**: contam-se as ocorrências dos verbos da lista e divide-se o resultado pela
#         quantidade de verbos do texto.
#
#         **Recursos de PLN utilizados durante o cálculo**:
#
#         **Limitações da métrica**:
#
#         **Crítica**:
#
#         A métrica só reconhece formas no infinitivo e não usa lematizador.
#
#         Verbos não podem ser classificados em “simples” ou “complexos” pelos seus lemas. Há formas verbais simples e
#         formas verbais complexas que possuem um mesmo lema. Por exemplo, a forma “há” do verbo haver é altamente
#         frequente, mas a forma “houvéssemos”, do mesmo verbo, é mais rara e complexa.
#
#         A métrica não distingue verbos principais de verbos auxiliares. Verbos auxiliares ocorrem mais, devido a sua
#         função gramatical de atribuir tempo, modo, aspecto e voz passiva. Portanto, verbos com função de auxiliares
#         deveriam ser desprezados.
#
#         **Projeto**: GUTEN
#
#         **Teste**: O reaproveitamento das águas residuais está apenas no início e tem todas as condições para ser uma
#         importante fonte não apenas de água doce, mas também de itens como fósforo, nitratos e biogás
#
#         **Contagens**: 3 verbos (está, tem, ser), todos da lista de verbos simples
#
#         **Resultado Esperado**: 3/3 = 1
#
#         **Resultado Obtido**: 0,333
#
#         **Status**: incorreto
#
#     """
#
#     name = 'Most Frequent Verbs of 3-degree'
#     column_name = 'most_frequent_verbs_of_3-degree'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         expression = re.compile('''
#                                 \[ser\]|
#                                 \[ir\]|
#                                 \[ter\]|
#                                 \[poder\]|
#                                 \[estar\]|
#                                 \[ficar\]|
#                                 \[dever\]|
#                                 \[precisar\]|
#                                 \[haver\]|
#                                 \[viver\]|
#                                 \[passar\]
#                                 ''',
#                                 re.VERBOSE)
#         try:
#             return len(re.findall(expression, flat)) / flat.count(' V ')
#         except ZeroDivisionError:
#             return 0


class NamedEntitiesRatioText(base.Metric):
    """
        **Nome da Métrica**: named_entity_ratio_text

        **Interpretação**: não está clara a relação da métrica com a complexidade textual

        **Descrição da métrica**: Proporção de Nomes Próprios em relação à quantidade de palavras do Texto

        **Definição dos termos que aparecem na descrição da métrica**: nomes próprios são palavras escritas em letra
        maiúscula, compostos de um ou mais tokens, capturadas por meio da etiqueta morfossintática PROP do parser
        Palavras. A etiqueta PROP junta todos os tokens de um mesmo nome próprio em uma multipalavra.

        **Forma de cálculo da métrica**: contam-se todas as ocorrências da etiqueta PROP e divide-se o resultado pela
        quantidade de palavras do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**: é importante para essa métrica usar a opção de junção de palavras que compõem os
        nomes próprios, disponível no parser Palavras; caso contrário, corre-se o risco de contar cada token do nome
        próprio como uma entidade nomeada diferente.

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: O melhor amigo do João é o Jorge Campos, que trabalha na Siemens. Eles se conheceram no Palestra
        Itália, num dia de decisão entre Palmeiras e São Paulo.

        **Contagens**: 28 palavras, 6 nomes próprios: João, Jorge Campos, Siemens, Palestra Itália, Palmeiras, São Paulo

        **Resultado Esperado**: 6/29 = 0,207

        **Resultado Obtido**: 0,207

        **Status**: correto
    """

    name = 'Named Entity Ratio on Text'
    column_name = 'named_entity_ratio_text'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            occurances = re.findall(' PROP ', flat)
            repeated = re.findall(' PROP .* PROP ', flat)  # Palavras bug
            elements = [i for i in flat.split('\n') if len(i) > 10]
            return (len(occurances) - len(repeated)) / len(elements)
        except ZeroDivisionError:
            return 0


class NamedEntitiesRatioSentence(base.Metric):
    """
        **Nome da Métrica**: named_entity_ratio_sentence

        **Interpretação**: segundo hipotetizado por Feng et al., 2010* (A Comparison of Features for Automatic
        Readability Assessment), quanto maior a quantidade de entidades mencionadas, maior a carga de memória requerida
        e, portanto, maior a complexidade textual. 

        **Descrição da métrica**: Média das proporções de nomes próprios em relação à quantidade de palavras das
        Sentenças

        **Definição dos termos que aparecem na descrição da métrica**: entidade nomeada é uma entidade do mundo real,
        como pessoas, lugares, datas, organizações, produtos, etc. que pode ser denotada por meio de um nome. O
        reconhecimento de entidades nomeadas é uma subtarefa do processamento de línguas naturais. Nessa métrica só
        estão sendo reconhecidas as entidades nomeadas que possuem nome próprio em letra maiúscula, capturadas por meio
        da etiqueta morfossintática PROP do parser Palavras.

        **Forma de cálculo da métrica**: contam-se as ocorrências da etiqueta PROP em cada sentença e divide-se o
        resultado pela quantidade de palavras da sentença (proporção de entidades nomeadas por quantidade de palavras
        das sentenças). Depois somam-se todas as proporções calculadas e divide-se o resultado pelo número de sentenças,
        para obter a média das proporções.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**: a precisão da métrica depende do desempenho do parser Palavras

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: Romero Jucá já disse que presidente deve vetar trecho sobre PIS-Cofins.

        **Contagens**: 10 palavras, 2 PROP (Romero=Jucá  e PIS-Cofins)

        **Resultado Esperado**: 0,20

        **Resultado Obtido**: 0,20

        **Status**: correto
    """

    name = 'Named Entity Ratio on Sentences'
    column_name = 'named_entity_ratio_sentence'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        sentences = flat.split('</s>')[:-2]
        results = []
        for sentence in sentences:
            occurances = re.findall(' PROP ', sentence)
            repeated = re.findall(' PROP .* PROP ', sentence)   # Palavras bug
            elements = [i for i in sentence.split('\n') if len(i) > 10]
            try:
                results.append((len(occurances) - len(repeated)) / len(elements))
            except ZeroDivisionError:
                results.append(0)
        try:
            return sum(results) / len(results)
        except ZeroDivisionError:
            return 0


# class HumanNamedEntitiesRatioText(base.Metric):
#     """
#         **Nome da Métrica**: human_named_entity_ratio_text
#
#         **Interpretação**: não está clara a relação da métrica com a complexidade textual
#
#         **Descrição da métrica**: Proporção de Entidades Nomeadas Humanas em relação à quantidade de palavras do texto
#
#         **Definição dos termos que aparecem na descrição da métrica**: entidade nomeada é uma entidade do mundo real,
#         como pessoas, lugares, datas, organizações, produtos, etc. que pode ser denotada por meio de um nome. O
#         reconhecimento de entidades nomeadas é uma subtarefa do processamento de línguas naturais. Nesta métrica só
#         estão sendo reconhecidas as entidades nomeadas que têm nome próprio em letra maiúscula (PROP) e que sejam
#         humanas, ou seja, que apresentem também a etiqueta semântica <hum> do parser Palavras.
#
#         **Forma de cálculo da métrica**: contam-se as ocorrências conjuntas das etiquetas PROP e <hum> no texto e
#         divide-se o resultado pela quantidade de palavras do texto.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras
#
#         **Limitações da métrica**:
#
#         **Crítica**: já existe uma métrica que engloba essa (named_entity_ratio_text), que tem mais correlação com
#         complexidade textual. Por esse motivo, esta métrica deve ser comentada.
#
#         **Projeto**: GUTEN
#     """
#
#     name = 'Human Named Entity Ratio on Text'
#     column_name = 'human_named_entity_ratio_text'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         try:
#             occurances = re.findall('<hum>.* PROP ', flat)
#             elements = [i for i in flat.split('\n') if len(i) > 10]
#             return len(occurances) / len(elements)
#         except ZeroDivisionError:
#             return 0


# class HumanNamedEntitiesRatioSentence(base.Metric):
#     """
#         **Nome da Métrica**: human_named_entity_ratio_sentence
#
#         **Interpretação**: não está clara a relação da métrica com a complexidade textual
#
#         **Descrição da métrica**: média das proporções de Entidades Nomeadas Humanas nas sentenças em relação à
#         quantidade de palavras das Sentenças
#
#         **Definição dos termos que aparecem na descrição da métrica**: entidade nomeada é uma entidade do mundo real,
#         como pessoas, lugares, datas, organizações, produtos, etc. que pode ser denotada por meio de um nome. O
#         reconhecimento de entidades nomeadas é uma subtarefa do processamento de línguas naturais. Nesta métrica só
#         estão sendo reconhecidas as entidades nomeadas que têm nome próprio em letra maiúscula (PROP) e que sejam
#         humanas, ou seja, que apresentem também a etiqueta semântica <hum> do parser Palavras. A métrica de entidades
#         nomeadas geral faz mais sentido para aferir complexidade textual.
#
#         **Forma de cálculo da métrica**: contam-se as ocorrências conjuntas das etiquetas PROP e <hum> em cada sentença
#         e divide-se o resultado pela quantidade de palavras da sentença (proporção de entidades nomeadas humanas por
#         quantidade de palavras nas sentenças). Depois somam-se todas as proporções calculadas e divide-se o resultado
#         pelo número de sentenças, para obter a média das proporções.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras
#
#         **Limitações da métrica**:
#
#         **Crítica**: já existe uma métrica que engloba essa (named_entity_ratio_sentence), que tem mais correlação com
#         complexidade textual. Por esse motivo, esta métrica deve ser comentada.
#
#         **Projeto**: GUTEN
#     """
#
#     name = 'Human Named Entity Ratio on Sentences'
#     column_name = 'human_named_entity_ratio_sentence'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         sentences = flat.split('</s>')[:-2]
#         results = []
#         for sentence in sentences:
#             occurances = re.findall('<hum>.* PROP ', sentence)
#             elements = [i for i in sentence.split('\n') if len(i) > 10]
#             try:
#                 results.append(len(occurances) / len(elements))
#             except ZeroDivisionError:
#                 results.append(0)
#         try:
#             return sum(results) / len(results)
#         except ZeroDivisionError:
#             return 0


# class NonHumanAnimateMovingNamedEntitiesRatioText(base.Metric):
#     """
#         **Nome da Métrica**: non-human_animate_moving_named_entity_ratio_text
#
#         **Interpretação**: não está clara a relação da métrica com a complexidade textual
#
#         **Descrição da métrica**: Proporção de Entidades Nomeadas Não Humanas Animadas que se Movem em relação à
#         quantidade de palavras do Texto
#
#         **Definição dos termos que aparecem na descrição da métrica**: entidade nomeada é uma entidade do mundo real,
#         como pessoas, lugares, datas, organizações, produtos, etc. que pode ser denotada por meio de um nome. O
#         reconhecimento de entidades nomeadas é uma subtarefa do processamento de línguas naturais. Nesta métrica estão
#         sendo reconhecidas como entidades nomeadas não humanas, animadas, que se movem, 7 etiquetas semânticas de
#         “animal prototypes” do parser Palavras ou seja, <Adom> (animal doméstico ou grande mamífero), <Amyth> (animal
#         mitológico), <Azo> (animal terrestre), <Aorn> (pássaro), <Acell> (animal celular ou célula do sangue), <Aich>
#         (animal aquático), <Aent> (inseto).
#
#         **Forma de cálculo da métrica**: contam-se as ocorrências das etiquetas de animais no texto e divide-se o
#         resultado pela quantidade de palavras do texto.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras
#
#         **Limitações da métrica**:
#
#         **Crítica**:
#
#         As palavras buscadas não se enquadram na descrição de entidades nomeadas;
#
#         O parser Palavras tem uma categoria de etiquetas semânticas exclusiva para entidades nomeadas, mas as etiquetas
#         eleitas para esta métrica não fazem parte dessa categoria;
#
#         Se a categoria semântica de onde foram extraídas as etiquetas se chama “animal prototypes”, qual o sentido de
#         renomear o conceito para “entidades nomeadas não humanas animadas com movimento? REVER conceito da métrica.
#
#         **Projeto**: GUTEN
#     """
#
#     name = 'Non-Human Animate Moving Named Entity Ratio on Text'
#     column_name = 'non-human_animate_moving_named_entity_ratio_text'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         try:
#             expression = re.compile('''
#                                     <Amyth>|    # Mitológico
#                                     <Acell>|    # Bactéria
#                                     <Adom>|     # Mamíferos
#                                     <Azo>|      # Pássaros
#                                     <Aorn>|     # Pássaros
#                                     <Aent>|     # Insetos
#                                     <Aich>      # Peixes
#                                     ''',
#                                     re.VERBOSE)
#             occurances = re.findall(expression, flat)
#             elements = [i for i in flat.split('\n') if len(i) > 10]
#             return len(occurances) / len(elements)
#         except ZeroDivisionError:
#             return 0


# class NonHumanAnimateMovingNamedEntitiesRatioSentence(base.Metric):
#     """
#         **Nome da Métrica**: non-human_animate_moving_named_entity_ratio_sentence
#
#         **Interpretação**: não está clara a relação da métrica com a complexidade textual
#
#         **Descrição da métrica**: média das proporções de Entidades Nomeadas Não Humanas Animadas que se Movem em
#         relação à quantidade de palavras nas sentenças
#
#         **Definição dos termos que aparecem na descrição da métrica**: entidade nomeada é uma entidade do mundo real,
#         como pessoas, lugares, datas, organizações, produtos, etc. que pode ser denotada por meio de um nome. O
#         reconhecimento de entidades nomeadas é uma subtarefa do processamento de línguas naturais. Nesta métrica estão
#         sendo reconhecidas como entidades nomeadas não humanas, animadas, que se movem, 7 etiquetas semânticas de
#         “animal prototypes” do parser Palavras ou seja, <Adom> (animal doméstico ou grande mamífero), <Amyth> (animal
#         mitológico), <Azo> (animal terrestre), <Aorn> (pássaro), <Acell> (animal celular ou célula do sangue), <Aich>
#         (animal aquático), <Aent> (inseto).
#
#         **Forma de cálculo da métrica**: contam-se as ocorrências das etiquetas de animais em cada sentença e divide-se
#         o resultado pela quantidade de palavras da sentença. Depois somam-se todas as proporções calculadas e divide-se
#         o resultado pelo número de sentenças, para obter a média das proporções.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras
#
#         **Limitações da métrica**:
#
#         **Crítica**:
#
#         As palavras buscadas não se enquadram na descrição de entidades nomeadas.
#
#         O parser Palavras tem uma categoria de etiquetas semânticas exclusiva para entidades nomeadas, mas as etiquetas
#         eleitas para esta métrica não fazem parte dessa categoria.
#
#         Se a categoria semântica de onde foram extraídas as etiquetas se chama “animal prototypes”, qual o sentido de
#         renomear o conceito para “entidades nomeadas não humanas animadas com movimento?
#
#         Por esses motivos, esta métrica deve ser comentada.
#
#         **Projeto**: GUTEN
#     """
#
#     name = 'Non-Human Animate Moving Named Entity Ratio on Sentences'
#     column_name = 'non-human_animate_moving_named_entity_ratio_sentence'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         sentences = flat.split('</s>')[:-2]
#         results = []
#         for sentence in sentences:
#             expression = re.compile('''
#                                     <Amyth>|    # Mitológico
#                                     <Acell>|    # Bactéria
#                                     <Adom>|     # Mamíferos
#                                     <Azo>|      # Pássaros
#                                     <Aorn>|     # Pássaros
#                                     <Aent>|     # Insetos
#                                     <Aich>      # Peixes
#                                     ''',
#                                     re.VERBOSE)
#             occurances = re.findall(expression, sentence)
#             elements = [i for i in sentence.split('\n') if len(i) > 10]
#             try:
#                 results.append(len(occurances) / len(elements))
#             except ZeroDivisionError:
#                 results.append(0)
#         try:
#             return sum(results) / len(results)
#         except ZeroDivisionError:
#             return 0


# class NonHumanAnimateNonMovingNamedEntitiesRatioText(base.Metric):
#     """
#         **Nome da Métrica**: non-human_animate_non-moving_named_entity_ratio_text
#
#         **Interpretação**: não está clara a relação da métrica com a complexidade textual
#
#         **Descrição da métrica**: Proporção de Entidades Nomeadas Não Humanas Animadas que não se Movem em relação à
#         quantidade de palavras do Texto
#
#         **Definição dos termos que aparecem na descrição da métrica**: entidade nomeada é uma entidade do mundo real,
#         como pessoas, lugares, datas, organizações, produtos, etc. que pode ser denotada por meio de um nome. O
#         reconhecimento de entidades nomeadas é uma subtarefa do processamento de línguas naturais. Nesta métrica estão
#         sendo reconhecidas como entidades nomeadas não humanas, animadas, que não se movem, apenas 1 das 5 etiquetas da
#         categoria semântica “plant prototypes” do parser Palavras ou seja, <Btree> (árvore)
#
#         **Forma de cálculo da métrica**: contam-se as ocorrências da etiquetas <Btree> no texto e divide-se o resultado
#         pela quantidade de palavras do texto.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras
#
#         **Limitações da métrica**:
#
#         **Crítica**: As palavras buscadas não se enquadram na descrição de entidades nomeadas.
#
#         O parser Palavras tem uma categoria de etiquetas semânticas exclusiva para entidades nomeadas, mas a etiqueta
#         eleita para esta métrica não faz parte dessa categoria.
#
#         Se a categoria semântica de onde foram extraídas as etiquetas se chama “plant prototypes”, qual o sentido de
#         renomear o conceito para “entidades nomeadas não humanas animadas sem movimento”?
#
#         O traço +animado é usado em semântica para descrever aquilo que tem movimento e, portanto, “animadas sem
#         movimento” é uma definição que traz o traço repetido com informações contrárias (que se move e que não se move),
#         o que é ilógico.
#
#         Por esses motivos, esta métrica deve ser comentada.
#     """
#
#     name = 'Non-Human Animate Non-Moving Named Entity Ratio on Text'
#     column_name = 'non-human_animate_non-moving_named_entity_ratio_text'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         try:
#             expression = '<Btree>'
#             # 'Árvores'
#             occurances = re.findall(expression, flat)
#             elements = [i for i in flat.split('\n') if len(i) > 10]
#             return len(occurances) / len(elements)
#         except ZeroDivisionError:
#             return 0


# class NonHumanAnimateNonMovingNamedEntitiesRatioSentence(base.Metric):
#     """
#         **Nome da Métrica**: non-human_animate_non-moving_named_entity_ratio_sentence
#
#         **Interpretação**: não está clara a relação da métrica com a complexidade textual
#
#         **Descrição da métrica**: média das proporções de Entidades Nomeadas Não Humanas Animadas que não se Movem em
#         relação à quantidade de palavras nas sentenças
#
#         **Definição dos termos que aparecem na descrição da métrica**: entidade nomeada é uma entidade do mundo real,
#         como pessoas, lugares, datas, organizações, produtos, etc. que pode ser denotada por meio de um nome. O
#         reconhecimento de entidades nomeadas é uma subtarefa do processamento de línguas naturais. Nesta métrica estão
#         sendo reconhecidas como entidades nomeadas não humanas, animadas, que não se movem, apenas 1 das 5 etiquetas da
#         categoria semântica “plant prototypes” do parser Palavras ou seja, <Btree> (árvore)
#
#         **Forma de cálculo da métrica**: contam-se as ocorrências da etiqueta <Btree> em cada sentença e divide-se o
#         resultado pela quantidade de palavras da sentença. Depois somam-se todas as proporções calculadas e divide-se
#         o resultado pelo número de sentenças, para obter a média das proporções.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras
#
#         **Limitações da métrica**:
#
#         **Crítica**: As palavras buscadas não se enquadram na descrição de entidades nomeadas. O parser Palavras tem
#         uma categoria de etiquetas semânticas exclusiva para entidades nomeadas, mas a etiqueta eleita para esta métrica
#         não faz parte dessa categoria. Se a categoria semântica de onde foram extraídas as etiquetas se chama “plant
#         prototypes”, qual o sentido de renomear o conceito para “entidades nomeadas não humanas animadas sem movimento”?
#
#         O traço +animado é usado em semântica para descrever aquilo que tem movimento e, portanto, “animadas sem
#         movimento” é uma definição que traz o traço repetido com informações contrárias (que se move e que não se move),
#         o que é ilógico.
#
#         Por esses motivos, esta métrica deve ser comentada.
#
#         **Projeto**: GUTEN
#     """
#
#     name = 'Non-Human Animate Non-Moving Named Entity Ratio on Sentences'
#     column_name = 'non-human_animate_non-moving_named_entity_ratio_sentence'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         sentences = flat.split('</s>')[:-2]
#         results = []
#         for sentence in sentences:
#             expression = '<Btree>'
#             # 'Árvores'
#             occurances = re.findall(expression, sentence)
#             elements = [i for i in sentence.split('\n') if len(i) > 10]
#             try:
#                 results.append(len(occurances) / len(elements))
#             except ZeroDivisionError:
#                 results.append(0)
#         try:
#             return sum(results) / len(results)
#         except ZeroDivisionError:
#             return 0


# class TopologicalsRatioText(base.Metric):
#     """
#         **Nome da Métrica**: topological_ratio_text
#
#         **Interpretação**: não está clara a relação da métrica com a complexidade textual
#
#         **Descrição da métrica**: Proporção de Topônimos em relação à quantidade de palavras do texto Definição dos
#         termos que aparecem na descrição da métrica: topônimo é “nome de lugar”, como por exemplo: Morro de São Paulo.
#         Nesta métrica são usadas etiquetas semânticas para capturar os mais diversos tipos de lugares: <Lwater>(lugar de
#         água: rio, lago, mar, etc); <Ltop> (lugar natural: pântano, promontório, etc); <Lpath> (caminho: rodovia, rua,
#         pista, etc); <Lciv> (local: cidade, estado, país, etc); <Ltrap> (armadilha); <L[a-z]*> (todas as etiquetas de
#         locais); <L> (local, hiperônimo de todas as etiquetas iniciadas por L) <inst> (instituição: auto-escola, teatro,
#         escola, etc); <furn>(mobília: cadeira, cama, quadro, etc.); <BB> (grupo de plantas).
#
#         **Forma de cálculo da métrica**: Contam-se as ocorrências das etiquetas eleitas e divide-se o resultado pela
#         quantidade de palavras do texto.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras
#
#         **Limitações da métrica**:
#
#         **Crítica**: Um tipo de lugar não é um topônimo. Topônimo é um lugar que tem um nome próprio.
#
#         Foram eleitas etiquetas de diferentes “prototypes” do Palavras, sem justificativa. Talvez combinando a etiqueta
#         PROP com as etiquetas de lugares seja possível capturar os lugares que têm nome. Por não ter fundamento lógico,
#         esta métrica deve ser comentada.
#
#         **Projeto**: GUTEN
#     """
#
#     name = 'Ratio of Topologicals on Text'
#     column_name = 'topological_ratio_text'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         try:
#             expression = re.compile('''
#                                     <BB>|       # Floresta
#                                     <Lwater>|   # Água
#                                     <Ltop>|     # Céu
#                                     <Lpath>|    # Caminho
#                                     <Lciv>|     # Local
#                                     <inst>|     # Instituição
#                                     <furn>|     # Mobília
#                                     <Ltrap>|    # Armadilhas
#                                     <L[a-z]*>|
#                                     <L>
#                                     ''',
#                                     re.VERBOSE)
#             occurances = re.findall(expression, flat)
#             elements = [i for i in flat.split('\n') if len(i) > 10]
#             return len(occurances) / len(elements)
#         except ZeroDivisionError:
#             return 0


# class TopologicalsRatioSentence(base.Metric):
#     """
#         **Nome da Métrica**: topological_ratio_sentence
#
#         **Interpretação**: não está clara a relação da métrica com a complexidade textual
#
#         **Descrição da métrica**: Média das proporções de Topônimos em relação à quantidade de palavras nas Sentenças
#
#         **Definição dos termos que aparecem na descrição da métrica**: topônimo é “nome de lugar”, como por exemplo:
#         Morro de São Paulo. Nesta métrica são usadas etiquetas semânticas para capturar os mais diversos tipos de
#         lugares: <Lwater>(lugar de água: rio, lago, mar, etc); <Ltop> (lugar natural: pântano, promontório, etc);
#         <Lpath> (caminho: rodovia, rua, pista, etc); <Lciv> (local: cidade, estado, país, etc); <Ltrap> (armadilha);
#         <L[a-z]*> (todas as etiquetas de locais); <L> (local, hiperônimo de todas as etiquetas iniciadas por L) <inst>
#         (instituição: auto-escola, teatro, escola, etc); <furn>(mobília: cadeira, cama, quadro, etc.); <BB> (grupo de
#         plantas).
#
#         **Forma de cálculo da métrica**: Contam-se as ocorrências das etiquetas eleitas em cada sentença e divide-se o
#         resultado pela quantidade de palavras da respectiva sentença. Depois se faz uma média entre todas proporções
#         calculadas.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras
#
#         **Limitações da métrica**:
#
#         **Crítica**: Um tipo de lugar não é um topônimo. Topônimo é um lugar que tem um nome próprio.
#
#         Foram eleitas etiquetas de diferentes “prototypes” do Palavras, sem justificativa. Talvez combinando a etiqueta
#         PROP com as etiquetas de lugares seja possível capturar os lugares que têm nome. Por não ter fundamento lógico,
#         esta métrica deve ser comentada.
#
#         **Projeto**: GUTEN
#     """
#
#     name = 'Ratio of Topologicals on Sentences'
#     column_name = 'topological_ratio_sentence'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         sentences = flat.split('</s>')[:-2]
#         results = []
#         for sentence in sentences:
#             expression = re.compile('''
#                                     <BB>|       # Floresta
#                                     <Lwater>|   # Água
#                                     <Ltop>|     # Céu
#                                     <Lpath>|    # Caminho
#                                     <Lciv>|     # Local
#                                     <inst>|     # Instituição
#                                     <furn>|     # Mobília
#                                     <Ltrap>|    # Armadilhas
#                                     <L[a-z]*>|
#                                     <L>
#                                     ''',
#                                     re.VERBOSE)
#             occurances = re.findall(expression, sentence)
#             elements = [i for i in sentence.split('\n') if len(i) > 10]
#             try:
#                 results.append(len(occurances) / len(elements))
#             except ZeroDivisionError:
#                 results.append(0)
#         try:
#             return sum(results) / len(results)
#         except ZeroDivisionError:
#             return 0


# class ConcreteMovingRatioText(base.Metric):
#     """
#         **Nome da Métrica**: concrete_moving_ratio_text
#
#         **Interpretação**: não está clara a relação entre esta métrica e a complexidade textual
#
#         **Descrição da métrica**: média das proporções de palavras concretas com movimento em relação ao número total de
#         palavras em cada sentença.
#
#         **Definição dos termos que aparecem na descrição da métrica**: palavras concretas com movimento correspondem à
#         categoria “Veículos” do parser Palavras, cujas etiquetas são são <Vwater>, <Vair>, <V[a-z]*>
#
#         **Forma de cálculo da métrica**: contam-se as ocorrências das etiquetas <Vwater>, <Vair>, <V[a-z]*> e divide-se
#         o resultado pela quantidade de palavras do texto.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras na função de anotação semântica
#
#         **Limitações da métrica**: o parser Palavras tem anotação semântica, mas ela não dá conta de todo o léxico
#
#         **Crítica**:
#
#             1) não está clara a motivação para distinguir objetos concretos moventes de concretos não moventes nas
#             métricas.
#
#             2) As etiquetas que representam a categoria criada com o nome de “objetos concretos com movimento” são
#             categorizadas como Veículos no parser Palavras.
#
#         **Projeto**: GUTEN
#
#         **Teste**: Ele foi de avião, porém sua mãe, que tem medo de voar, preferiu ir de trem. Além disso, as poltronas
#         do trem são bem mais espaçosas que as do avião.
#
#         **Contagens**: dois objetos concretos com movimento e 16 palavras na primeira sentença; 2 objetos concretos com
#         movimento e 17 palavras na segunda sentença, contando descontrações ou 14, sem contá-las. Total: 4 objetos,
#         30 palavras, 2 sentenças.
#
#         **Resultado Esperado**: (4/30) = 0,133 (a métrica está utilizando tokenização sem descontração)
#
#         **Resultado Obtido**: 0,10
#
#         **Status**: incorreto
#
#     """
#
#     name = 'Ratio of Concrete Moving Entitites on Text'
#     column_name = 'concrete_moving_ratio_text'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         try:
#             expression = re.compile('''
#                                     <Vwater>|   # Navios
#                                     <Vair>|     # Aviões
#                                     <V[a-z]*>
#                                     ''',
#                                     re.VERBOSE)
#             occurances = re.findall(expression, flat)
#             elements = [i for i in flat.split('\n') if len(i) > 10]
#             return len(occurances) / len(elements)
#         except ZeroDivisionError:
#             return 0


# class ConcreteMovingRatioSentence(base.Metric):
#     """
#         **Nome da Métrica**: concrete_moving_ratio_sentence
#
#         **Interpretação**: não está clara a relação entre esta métrica e a complexidade textual
#
#         **Descrição da métrica**: média por sentença da proporção de palavras concretas com movimento em relação ao
#         total de palavras
#
#         **Definição dos termos que aparecem na descrição da métrica**: palavras concretas com movimento correspondem à
#         categoria “Veículos” do parser Palavras, cujas etiquetas são são <Vwater>, <Vair>, <V[a-z]*>
#
#         **Forma de cálculo da métrica**: contam-se as ocorrências das etiquetas <Vwater>, <Vair>, <V[a-z]*> e divide-se
#         o resultado pela quantidade de palavras da sentença. Depois somam-se as proporções obtidas em cada sentença e
#         divide-se o resultado pelo número de sentenças do texto.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras na função de anotação semântica
#
#         **Limitações da métrica**: o parser Palavras tem anotação semântica, mas ela não dá conta de todo o léxico
#
#         **Crítica**:
#
#             1) não está clara a motivação para distinguir objetos concretos com movimento de concretos sem movimento nas
#             métricas.
#
#             2) As etiquetas que representam a categoria criada com o nome de “objetos concretos com movimento” são
#             categorizadas como Veículos no parser Palavras.
#
#             3) ERRO: aparentemente a métrica não está dividindo pelo número de sentenças
#
#         **Projeto**: GUTEN
#
#         **Teste**: Ele foi de avião, porém sua mãe, que tem medo de voar, preferiu ir de trem. Além disso, as poltronas
#         do trem são bem mais espaçosas que as do avião.
#
#         **Contagens**: dois objetos concretos com movimento e 16 palavras na primeira sentença; 2 objetos concretos com
#         movimento e 17 palavras na segunda sentença, contando descontrações ou 14, sem contá-las.
#
#         **Resultado Esperado**: (4/30)/2 = 0,05
#
#         **Resultado Obtido**: 0,10
#
#         **Status**: incorreto
#     """
#
#     name = 'Ratio of Concrete Moving Entitites on Sentences'
#     column_name = 'concrete_moving_ratio_sentence'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         sentences = flat.split('</s>')[:-2]
#         results = []
#         for sentence in sentences:
#             expression = re.compile('''
#                                     <Vwater>|   # Navios
#                                     <Vair>|     # Aviões
#                                     <V[a-z]*>
#                                     ''',
#                                     re.VERBOSE)
#             occurances = re.findall(expression, sentence)
#             elements = [i for i in sentence.split('\n') if len(i) > 10]
#             try:
#                 results.append(len(occurances) / len(elements))
#             except ZeroDivisionError:
#                 results.append(0)
#         try:
#             return sum(results) / len(results)
#         except ZeroDivisionError:
#             return 0


# class ConcreteNonMovingRatioText(base.Metric):
#     """
#         **Nome da Métrica**: concrete_non-moving_ratio_text
#
#         **Interpretação**: não está clara a relação entre esta métrica e a complexidade textual
#
#         **Descrição da métrica**: proporção de palavras concretas sem movimento em relação à quantidade de palavras do
#         texto.
#
#         **Definição dos termos que aparecem na descrição da métrica**: objetos concretos sem movimento correspondem à
#         categoria “substância” do parser Palavras, cujas etiquetas são <cm-liq>, <mat>,<cm-rem>, <cm-[a-z]*> e <cm>
#
#         **Forma de cálculo da métrica**: contam-se as ocorrências das etiquetas <cm-liq>, <mat>, <cm-rem>, <cm-[a-z]*>
#         e <cm> no texto e divide-se o resultado pela quantidade de palavras do texto.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras na função de anotação semântica
#
#         **Limitações da métrica**: o parser Palavras tem anotação semântica, mas ela não dá conta de todo o léxico
#
#         **Crítica**:
#
#             1) não está clara a motivação para distinguir concretos com movimento de concretos sem movimento nas
#             métricas.
#
#             2) As etiquetas que representam a categoria criada com o nome de “concretos sem movimento” são categorizadas
#             como “substâncias” no parser Palavras.
#
#             3) Proposta: EXCLUIR
#
#         **Projeto**: GUTEN
#
#         **Teste**: Ele foi de avião, porém sua mãe, que tem medo de voar, preferiu ir de trem. Além disso, as poltronas
#         do trem são bem mais espaçosas que as do avião.
#
#         **Contagens**: nenhum objeto concreto sem movimento e 16 palavras na primeira sentença; 1 objeto concreto sem
#         movimento e 17 palavras na segunda sentença, contando descontrações ou 14, sem contá-las. Total: 1 objeto e 30
#         palavras no texto.
#
#         **Resultado Esperado**: 1/30 = 0,033
#
#         **Resultado Obtido**: 0,033
#
#         **Status**: correto
#     """
#
#     name = 'Ratio of Concrete Non-Moving Entities on Text'
#     column_name = 'concrete_non-moving_ratio_text'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         try:
#             expression = re.compile('''
#                                     <cm-liq>|       # Líquidos
#                                     <mat>|          # Materiais
#                                     <cm-rem>|       # Remédio
#                                     <cm-[a-z]*>|
#                                     <cm>
#                                     ''',
#                                     re.VERBOSE)
#             occurances = re.findall(expression, flat)
#             elements = [i for i in flat.split('\n') if len(i) > 10]
#             return len(occurances) / len(elements)
#         except ZeroDivisionError:
#             return 0


# class ConcreteNonMovingRatioSentence(base.Metric):
#     """
#         **Nome da Métrica**: concrete_non-moving_ratio_sentence
#
#         **Interpretação**: não está clara a relação entre esta métrica e a complexidade textual
#
#         **Descrição da métrica**: média das proporções de palavras concretas sem movimento pela quantidade total de
#         palavras de cada sentença.
#
#         **Definição dos termos que aparecem na descrição da métrica**: objetos concretos sem movimento correspondem à
#         categoria “substância” do parser Palavras, cujas etiquetas são <cm-liq>, <mat>,<cm-rem>, <cm-[a-z]*> e <cm>
#
#         **Forma de cálculo da métrica**: contam-se as ocorrências das etiquetas <cm-liq>, <mat>,<cm-rem>, <cm-[a-z]*> e
#         <cm> em cada sentença e divide-se o resultado pela quantidade de palavras da sentença. Depois somam-se as
#         proporções obtidas e divide-se o resultado pelo número de sentenças do texto.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras na função de anotação semântica
#
#         **Limitações da métrica**: o parser Palavras tem anotação semântica, mas ela não dá conta de todo o léxico
#
#         **Crítica**:
#
#             1) não está clara a motivação para distinguir concretos moventes de concretos não moventes nas métricas.
#
#             2) As etiquetas que representam a categoria criada com o nome de “concretos sem movimento” são categorizadas
#             como “Substâncias” no parser Palavras.
#
#             3) ERRO: aparentemente a métrica não está dividindo pelo número de sentenças
#
#         **Projeto**: GUTEN
#
#         **Teste**: Ele foi de avião, porém sua mãe, que tem medo de voar, preferiu ir de trem. Além disso, as poltronas
#         do trem são bem mais espaçosas que as do avião.
#
#         **Contagens**: nenhum objeto concreto sem movimento e 16 palavras na primeira sentença; 1 objeto concreto sem
#         movimento e 17 palavras na segunda sentença, contando descontrações ou 14, sem contá-las.
#
#         **Resultado Esperado**: (1/30)/2 = 0,0165 (a métrica está utilizando tokenização sem descontração)
#
#         **Resultado Obtido**: 0,033
#
#         **Status**: incorreto
#     """
#
#     name = 'Ratio of Concrete Non-Moving Entities on Sentences'
#     column_name = 'concrete_non-moving_ratio_sentence'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         sentences = flat.split('</s>')[:-2]
#         results = []
#         for sentence in sentences:
#             expression = re.compile('''
#                                     <cm-liq>|       # Líquidos
#                                     <mat>|          # Materiais
#                                     <cm-rem>|       # Remédio
#                                     <cm-[a-z]*>|
#                                     <cm>
#                                     ''',
#                                     re.VERBOSE)
#             occurances = re.findall(expression, sentence)
#             elements = [i for i in sentence.split('\n') if len(i) > 10]
#             try:
#                 results.append(len(occurances) / len(elements))
#             except ZeroDivisionError:
#                 results.append(0)
#         try:
#             return sum(results) / len(results)
#         except ZeroDivisionError:
#             return 0


class IndicativePresentMoodRatio(base.Metric):
    """
        **Nome da Métrica**: indicative_present_ratio

        **Interpretação**: o presente do indicativo é o tempo verbal de mais frequente de todas as formas flexionadas.
        Pode ocorrer em textos de diferentes níveis de complexidade.

        **Descrição da métrica**: proporção de verbos no presente do modo indicativo, em relação ao total de verbos
        flexionados do texto

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: contam-se todas as ocorrências da etiqueta V.*PR.*IND e divide-se o resultado
        pelo total de verbos flexionados do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**:

        **Crítica**:

        **Projeto**: GUTEN

        **Teste 1**:

        O secretário da Segurança Pública, Enio Bacci, disse que o aumento está ligado à legislação branda contra
        desmanches, ao aumento da frota e ao chamado golpe do seguro -- quando o dono vende o carro a bandidos e recebe
        um novo da seguradora, mas reconheceu que precisa ajustar a repressão.

        **Contagens**:

        6 verbos, 4 no presente do indicativo (está, vende, recebe, precisa), 2 no pretérito perfeito (disse,
        reconheceu)

        O parser, contudo, reconheceu indevidamente o substantivo “desmanches” como mais um verbo (no presente do
        subjuntivo).

        **Resultado Esperado**: 0,667 (4/6) ou 0,571 (4/7) se considerarmos o erro do parser

        **Resultado Obtido**: 0,571

        **Status**: correto
    """

    name = 'Ratio of Indicative Present Mood'
    column_name = 'indicative_present_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            present_ind = len(re.findall('V.*PR.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            present_ind = 0
        try:
            imperfect_ind = len(re.findall('V.*IMPF.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            imperfect_ind = 0
        try:
            perfect_ind = len(re.findall('V.*PS.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            perfect_ind = 0
        try:
            pluperfect_ind = len(re.findall('V.*MQP.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            pluperfect_ind = 0
        try:
            future_ind = len(re.findall('V.*FUT.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            future_ind = 0
        try:
            conditional_ind = len(re.findall('V.*COND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            conditional_ind = 0
        try:
            present_subj = len(re.findall('V.*PR.*SUBJ', flat)) / flat.count(' V ')
        except:
            present_subj = 0
        try:
            imperfect_subj = len(re.findall('V.*IMPF.*SUBJ', flat)) / flat.count(' V ')
        except:
            imperfect_subj = 0
        try:
            future_subj = len(re.findall('V.*FUT.*SUBJ', flat)) / flat.count(' V ')
        except:
            future_subj = 0
        try:
            imperative = len(re.findall('V.*IMP ', flat)) / flat.count(' V ')
        except:
            imperative = 0
        tenses = [present_ind, imperfect_ind, perfect_ind, pluperfect_ind,
                  future_ind, conditional_ind, present_subj, imperfect_subj,
                  future_subj, imperative]
        return tenses[0] / sum(tenses)


class IndicativePerfectMoodRatio(base.Metric):
    """
        **Nome da Métrica**: indicative_preterite_perfect_ratio

        **Interpretação**: o pretérito perfeito simples do indicativo é um tempo frequente. Pode ocorrer em textos de
        diferentes níveis de complexidade. Não está claro qual sua relação com o nível de complexidade do texto.

        **Descrição da métrica**: proporção de verbos no pretérito perfeito simples do modo indicativo, em relação ao
        total de verbos do texto

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: contam-se todas as ocorrências da etiqueta V.*PS.*IND e divide-se o resultado
        pelo total de verbos do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**:

        **Crítica**:

        **Projeto**: GUTEN

        **Teste 1**:

        Robert Lustig trabalha como endocrinologista pediátrico na Universidade da Califórnia, especializado no
        tratamento da obesidade infantil. Em 2009, ele proferiu a palestra “Açúcar: a amarga verdade”, que teve mais
        de 6 milhões de visualizações no YouTube. No decorrer de uma hora e meia, Lustig defende com veemência que a
        frutose, um açúcar onipresente na alimentação moderna, é o “veneno” responsável pela epidemia de obesidade nos
        Estados Unidos.

        **Contagens**: 6 verbos, 3 no presente, 2 no pretérito perfeito

        **Resultado Esperado**: 2/5 = 0,40

        **Resultado Obtido**: 0,40

        **Status**: correto
    """

    name = 'Ratio of Indicative Preterite Perfect Mood'
    column_name = 'indicative_preterite_perfect_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            present_ind = len(re.findall('V.*PR.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            present_ind = 0
        try:
            imperfect_ind = len(re.findall('V.*IMPF.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            imperfect_ind = 0
        try:
            perfect_ind = len(re.findall('V.*PS.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            perfect_ind = 0
        try:
            pluperfect_ind = len(re.findall('V.*MQP.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            pluperfect_ind = 0
        try:
            future_ind = len(re.findall('V.*FUT.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            future_ind = 0
        try:
            conditional_ind = len(re.findall('V.*COND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            conditional_ind = 0
        try:
            present_subj = len(re.findall('V.*PR.*SUBJ', flat)) / flat.count(' V ')
        except:
            present_subj = 0
        try:
            imperfect_subj = len(re.findall('V.*IMPF.*SUBJ', flat)) / flat.count(' V ')
        except:
            imperfect_subj = 0
        try:
            future_subj = len(re.findall('V.*FUT.*SUBJ', flat)) / flat.count(' V ')
        except:
            future_subj = 0
        try:
            imperative = len(re.findall('V.*IMP ', flat)) / flat.count(' V ')
        except:
            imperative = 0
        tenses = [present_ind, imperfect_ind, perfect_ind, pluperfect_ind,
                  future_ind, conditional_ind, present_subj, imperfect_subj,
                  future_subj, imperative]
        return tenses[2] / sum(tenses)


class IndicativeImperfectMoodRatio(base.Metric):
    """
        **Nome da Métrica**: indicative_imperfect_ratio

        **Interpretação**: o pretérito imperfeito do indicativo é um tempo frequente. Pode ocorrer em textos de
        diferentes níveis de complexidade. Não está claro qual sua relação com o nível de complexidade do texto.

        **Descrição da métrica**: proporção de verbos no pretérito imperfeito do modo indicativo, em relação ao total
        de verbos do texto

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: contam-se todas as ocorrências da etiqueta V.*IMPF.*IND e divide-se o resultado
        pelo total de verbos flexionados do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**: a precisão da métrica depende do desempenho do parser Palavras

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: A conclusão da investigação do exército sobre o caso, que vazou para a imprensa, afirma que as
        acusações de homicídio doloso (com intenção) eram "infundadas".

        **Contagens**: 3 verbos flexionados (vazou, afirma, eram); 1 verbo no pretérito imperfeito do indicativo (eram)

        **Resultado Esperado**: 0,333 (1/3)

        **Resultado Obtido**: 0,333

        **Status**: correto
    """

    name = 'Ratio of Indicative Imperfect Mood'
    column_name = 'indicative_imperfect_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            present_ind = len(re.findall('V.*PR.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            present_ind = 0
        try:
            imperfect_ind = len(re.findall('V.*IMPF.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            imperfect_ind = 0
        try:
            perfect_ind = len(re.findall('V.*PS.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            perfect_ind = 0
        try:
            pluperfect_ind = len(re.findall('V.*MQP.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            pluperfect_ind = 0
        try:
            future_ind = len(re.findall('V.*FUT.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            future_ind = 0
        try:
            conditional_ind = len(re.findall('V.*COND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            conditional_ind = 0
        try:
            present_subj = len(re.findall('V.*PR.*SUBJ', flat)) / flat.count(' V ')
        except:
            present_subj = 0
        try:
            imperfect_subj = len(re.findall('V.*IMPF.*SUBJ', flat)) / flat.count(' V ')
        except:
            imperfect_subj = 0
        try:
            future_subj = len(re.findall('V.*FUT.*SUBJ', flat)) / flat.count(' V ')
        except:
            future_subj = 0
        try:
            imperative = len(re.findall('V.*IMP ', flat)) / flat.count(' V ')
        except:
            imperative = 0
        tenses = [present_ind, imperfect_ind, perfect_ind, pluperfect_ind,
                  future_ind, conditional_ind, present_subj, imperfect_subj,
                  future_subj, imperative]
        return tenses[1] / sum(tenses)


class IndicativePluperfectMoodRatio(base.Metric):
    """
        **Nome da Métrica**: indicative_pluperfect_ratio

        **Interpretação**: o pretérito mais que perfeito é um verbo pouco frequente e está associado a alta complexidade.
        Portanto, quanto maior a métrica, maior a complexidade.

        **Descrição da métrica**: proporção de verbos no pretérito mais que perfeito do modo indicativo, em relação ao
        total de verbos do texto

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: contam-se todas as ocorrências da etiqueta V MQP.*IND e divide-se o resultado
        pelo total de verbos flexionados do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**: a precisão da métrica depende do desempenho do parser Palavras. A terceira pessoa do
        plural dos tempos pretérito perfeito e pretérito mais que perfeito são iguais, o que gera ambiguidade,
        aumentando indevidamente a estatística desta métrica.

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: Trechos do vídeo foram exibidos pela rede britânica BBC e mostram uma fileira de corpos com
        ferimentos claramente provocados por tiros.

        **Contagens**: 2 verbos, os dois flexionados (foram, mostram).

        A forma “foram” está usada no pretérito perfeito, mas como se trata de uma forma ambígua, o parser a reconheceu
        com 2 etiquetas: pretérito mais que perfeito e pretérito perfeito, fazendo o cálculo da métrica computar 3
        verbos flexionados

        **Resultado Esperado**: 0

        **Resultado Obtido**: 0,333 (1/3)

        **Status**: correto, considerando a limitação do parser
    """

    name = 'Ratio of Indicative Pluperfect Mood'
    column_name = 'indicative_pluperfect_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            present_ind = len(re.findall('V.*PR.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            present_ind = 0
        try:
            imperfect_ind = len(re.findall('V.*IMPF.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            imperfect_ind = 0
        try:
            perfect_ind = len(re.findall('V.*PS.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            perfect_ind = 0
        try:
            pluperfect_ind = len(re.findall('V.*MQP.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            pluperfect_ind = 0
        try:
            future_ind = len(re.findall('V.*FUT.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            future_ind = 0
        try:
            conditional_ind = len(re.findall('V.*COND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            conditional_ind = 0
        try:
            present_subj = len(re.findall('V.*PR.*SUBJ', flat)) / flat.count(' V ')
        except:
            present_subj = 0
        try:
            imperfect_subj = len(re.findall('V.*IMPF.*SUBJ', flat)) / flat.count(' V ')
        except:
            imperfect_subj = 0
        try:
            future_subj = len(re.findall('V.*FUT.*SUBJ', flat)) / flat.count(' V ')
        except:
            future_subj = 0
        try:
            imperative = len(re.findall('V.*IMP ', flat)) / flat.count(' V ')
        except:
            imperative = 0
        tenses = [present_ind, imperfect_ind, perfect_ind, pluperfect_ind,
                  future_ind, conditional_ind, present_subj, imperfect_subj,
                  future_subj, imperative]
        return tenses[3] / sum(tenses)


class IndicativeFutureMoodRatio(base.Metric):
    """
        **Nome da Métrica**: indicative_future_ratio

        **Interpretação**: o futuro do indicativo não está entre os tempos verbais mais frequentes e pode estar
        associado a complexidade mediana. Portanto, quanto maior a métrica, maior a complexidade.

        **Descrição da métrica**: proporção de verbos no futuro do indicativo em relação ao total de verbos flexionados
        do texto

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: contam-se todas as ocorrências da etiqueta 'V.*FUT*IND e divide-se o resultado
        pelo total de verbos flexionados do texto (VFIN).

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**: a precisão da métrica depende do desempenho do parser Palavras

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: Pescadores tentarão retirar o maior número de peixes da espécie, que pode atingir 20 centímetros de
        comprimento e um quilo.

        **Contagens**: 2 verbos flexionados (tentarão, pode), 1 no futuro do indicativo (tentarão)

        **Resultado Esperado**: 0,50 (1/2)

        **Resultado Obtido**: 0,50

        **Status**: correto
    """

    name = 'Ratio of Indicative Future mood'
    column_name = 'indicative_future_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            present_ind = len(re.findall('V.*PR.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            present_ind = 0
        try:
            imperfect_ind = len(re.findall('V.*IMPF.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            imperfect_ind = 0
        try:
            perfect_ind = len(re.findall('V.*PS.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            perfect_ind = 0
        try:
            pluperfect_ind = len(re.findall('V.*MQP.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            pluperfect_ind = 0
        try:
            future_ind = len(re.findall('V.*FUT.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            future_ind = 0
        try:
            conditional_ind = len(re.findall('V.*COND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            conditional_ind = 0
        try:
            present_subj = len(re.findall('V.*PR.*SUBJ', flat)) / flat.count(' V ')
        except:
            present_subj = 0
        try:
            imperfect_subj = len(re.findall('V.*IMPF.*SUBJ', flat)) / flat.count(' V ')
        except:
            imperfect_subj = 0
        try:
            future_subj = len(re.findall('V.*FUT.*SUBJ', flat)) / flat.count(' V ')
        except:
            future_subj = 0
        try:
            imperative = len(re.findall('V.*IMP ', flat)) / flat.count(' V ')
        except:
            imperative = 0
        tenses = [present_ind, imperfect_ind, perfect_ind, pluperfect_ind,
                  future_ind, conditional_ind, present_subj, imperfect_subj,
                  future_subj, imperative]
        return tenses[4] / sum(tenses)


class IndicativeConditionMoodRatio(base.Metric):
    """
        **Nome da Métrica**: indicative_condition_ratio

        **Interpretação**: o futuro do pretérito do indicativo não está entre os tempos verbais mais frequentes e pode
        estar associado a complexidade mediana. Portanto, quanto maior a métrica, maior a complexidade.

        **Descrição da métrica**: proporção de verbos no futuro do pretérito do indicativo em relação ao total de verbos
        flexionados do texto

        **Definição dos termos que aparecem na descrição da métrica**: o futuro do pretérito do indicativo chama-se
        “condicional” na gramática do português europeu e corresponde ao “conditional” do inglês.

        **Forma de cálculo da métrica**: contam-se todas as ocorrências da etiqueta 'V.*COND’ e divide-se o resultado
        pelo total de verbos flexionados do texto (VFIN).

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**: a precisão da métrica depende do desempenho do parser Palavras

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: Com a oferta do uniforme, as escolas públicas poderão torná-lo obrigatório, o que eliminaria a roupa
        como um indicador de diferenças sociais nas escolas e não criaria constrangimento aos alunos mais pobres.

        **Contagens**: 3 verbos flexionados (poderão, eliminaria, criaria), dois dos quais no futuro do pretérito do
        indicativo (eliminaria e criaria)

        **Resultado Esperado**: 0,666 (2/3)

        **Resultado Obtido**: 0,666

        **Status**: correto
    """

    name = 'Ratio of Indicative Condition Mood'
    column_name = 'indicative_condition_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            present_ind = len(re.findall('V.*PR.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            present_ind = 0
        try:
            imperfect_ind = len(re.findall('V.*IMPF.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            imperfect_ind = 0
        try:
            perfect_ind = len(re.findall('V.*PS.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            perfect_ind = 0
        try:
            pluperfect_ind = len(re.findall('V.*MQP.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            pluperfect_ind = 0
        try:
            future_ind = len(re.findall('V.*FUT.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            future_ind = 0
        try:
            conditional_ind = len(re.findall('V.*COND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            conditional_ind = 0
        try:
            present_subj = len(re.findall('V.*PR.*SUBJ', flat)) / flat.count(' V ')
        except:
            present_subj = 0
        try:
            imperfect_subj = len(re.findall('V.*IMPF.*SUBJ', flat)) / flat.count(' V ')
        except:
            imperfect_subj = 0
        try:
            future_subj = len(re.findall('V.*FUT.*SUBJ', flat)) / flat.count(' V ')
        except:
            future_subj = 0
        try:
            imperative = len(re.findall('V.*IMP ', flat)) / flat.count(' V ')
        except:
            imperative = 0
        tenses = [present_ind, imperfect_ind, perfect_ind, pluperfect_ind,
                  future_ind, conditional_ind, present_subj, imperfect_subj,
                  future_subj, imperative]
        return tenses[5] / sum(tenses)


class SubjunctivePresentMoodRatio(base.Metric):
    """
        **Nome da Métrica**: subjunctive_present_ratio

        **Interpretação**: o presente do subjuntivo é um tempo usado com menor frequência e sua ocorrência pode estar
        associada a maior complexidade

        **Descrição da métrica**: Proporção de Verbos no Presente do Subjuntivo em relação ao total de verbos
        flexionados do texto

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: contam-se todas as ocorrências da etiqueta V.*PR.*SUBJ e divide-se o resultado
        pelo total de verbos flexionados (VFIN) do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**:  a precisão da métrica depende do desempenho do parser Palavras. Como as terceiras
        pessoas do presente do subjuntivo são ambíguas com as terceiras pessoas do imperativo e o parser por default
        anota todas como subjuntivo, esta métrica pode ser superestimada.

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: A regra obriga as legendas a fechar nos Estados apenas coligações que não colidam com as nacionais.

        **Contagens**: 2 verbos flexionados, 1 no presente do indicativo (obriga) e 1 no presente do subjuntivo
        (colidam)

        **Resultado Esperado**: 0,50 (1/2)

        **Resultado Obtido**: 0,50

        **Status**: correto
    """

    name = 'Ratio of Subjunctive Present mood'
    column_name = 'subjunctive_present_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            present_ind = len(re.findall('V.*PR.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            present_ind = 0
        try:
            imperfect_ind = len(re.findall('V.*IMPF.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            imperfect_ind = 0
        try:
            perfect_ind = len(re.findall('V.*PS.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            perfect_ind = 0
        try:
            pluperfect_ind = len(re.findall('V.*MQP.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            pluperfect_ind = 0
        try:
            future_ind = len(re.findall('V.*FUT.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            future_ind = 0
        try:
            conditional_ind = len(re.findall('V.*COND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            conditional_ind = 0
        try:
            present_subj = len(re.findall('V.*PR.*SUBJ', flat)) / flat.count(' V ')
        except:
            present_subj = 0
        try:
            imperfect_subj = len(re.findall('V.*IMPF.*SUBJ', flat)) / flat.count(' V ')
        except:
            imperfect_subj = 0
        try:
            future_subj = len(re.findall('V.*FUT.*SUBJ', flat)) / flat.count(' V ')
        except:
            future_subj = 0
        try:
            imperative = len(re.findall('V.*IMP ', flat)) / flat.count(' V ')
        except:
            imperative = 0
        tenses = [present_ind, imperfect_ind, perfect_ind, pluperfect_ind,
                  future_ind, conditional_ind, present_subj, imperfect_subj,
                  future_subj, imperative]
        return tenses[6] / sum(tenses)


class SubjunctiveImperfectMoodRatio(base.Metric):
    """
        **Nome da Métrica**: subjunctive_imperfect_ratio

        **Interpretação**: o imperfeito do subjuntivo é um tempo usado com menor frequência e sua ocorrência pode estar
        associada a maior complexidade

        **Descrição da métrica**: Proporção de Verbos no Pretérito Imperfeito do Subjuntivo em relação ao total de
        verbos flexionados

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: contam-se todas as ocorrências da etiqueta 'V.*IMPF.*SUBJ' e divide-se o
        resultado pelo total de verbos flexionados do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**:  a precisão da métrica depende do desempenho do parser Palavras

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: Não fosse o aspecto financeiro, o projeto teria o apoio incondicional de Balzano e do presidente da
        Famurs, Flávio Luiz Lammel.

        **Contagens**: 2 verbos flexionados, 1 no imperfeito do subjuntivo (fosse) e 1 no futuro do pretérito do
        indicativo (teria).

        **Resultado Esperado**: 0,50 (1/2)

        **Resultado Obtido**: 0,50

        **Status**: correto
    """

    name = 'Ratio of Subjunctive Imperfect mood'
    column_name = 'subjunctive_imperfect_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            present_ind = len(re.findall('V.*PR.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            present_ind = 0
        try:
            imperfect_ind = len(re.findall('V.*IMPF.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            imperfect_ind = 0
        try:
            perfect_ind = len(re.findall('V.*PS.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            perfect_ind = 0
        try:
            pluperfect_ind = len(re.findall('V.*MQP.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            pluperfect_ind = 0
        try:
            future_ind = len(re.findall('V.*FUT.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            future_ind = 0
        try:
            conditional_ind = len(re.findall('V.*COND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            conditional_ind = 0
        try:
            present_subj = len(re.findall('V.*PR.*SUBJ', flat)) / flat.count(' V ')
        except:
            present_subj = 0
        try:
            imperfect_subj = len(re.findall('V.*IMPF.*SUBJ', flat)) / flat.count(' V ')
        except:
            imperfect_subj = 0
        try:
            future_subj = len(re.findall('V.*FUT.*SUBJ', flat)) / flat.count(' V ')
        except:
            future_subj = 0
        try:
            imperative = len(re.findall('V.*IMP ', flat)) / flat.count(' V ')
        except:
            imperative = 0
        tenses = [present_ind, imperfect_ind, perfect_ind, pluperfect_ind,
                  future_ind, conditional_ind, present_subj, imperfect_subj,
                  future_subj, imperative]
        return tenses[7] / sum(tenses)


class SubjunctiveFutureMoodRatio(base.Metric):
    """
        **Nome da Métrica**: subjunctive_future_ratio

        **Interpretação**: o futuro do subjuntivo é um tempo usado com menor frequência e sua ocorrência pode estar
        associada a maior complexidade

        **Descrição da métrica**: Proporção de Verbos no Futuro do Subjuntivo em relação ao total de verbos flexionados

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: contam-se todas as ocorrências da etiqueta V.*FUT.*SUBJ e divide-se o resultado
        pelo total de verbos flexionados do texto.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**:  a precisão da métrica depende do desempenho do parser Palavras

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: Se a idéia for aprovada, os estudantes receberão dois conjuntos anuais, completados por calçado,
        meias, calça e camiseta.

        **Contagens**: 2 verbos flexionados, 1 no futuro do subjuntivo (for) e 1 no futuro do indicativo (receberão)

        **Resultado Esperado**: 0,50 (1/2)

        **Resultado Obtido**: 0,50

        **Status**: correto
    """

    name = 'Ratio of Future Imperfect mood'
    column_name = 'subjunctive_future_ratio'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            present_ind = len(re.findall('V.*PR.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            present_ind = 0
        try:
            imperfect_ind = len(re.findall('V.*IMPF.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            imperfect_ind = 0
        try:
            perfect_ind = len(re.findall('V.*PS.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            perfect_ind = 0
        try:
            pluperfect_ind = len(re.findall('V.*MQP.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            pluperfect_ind = 0
        try:
            future_ind = len(re.findall('V.*FUT.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            future_ind = 0
        try:
            conditional_ind = len(re.findall('V.*COND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            conditional_ind = 0
        try:
            present_subj = len(re.findall('V.*PR.*SUBJ', flat)) / flat.count(' V ')
        except:
            present_subj = 0
        try:
            imperfect_subj = len(re.findall('V.*IMPF.*SUBJ', flat)) / flat.count(' V ')
        except:
            imperfect_subj = 0
        try:
            future_subj = len(re.findall('V.*FUT.*SUBJ', flat)) / flat.count(' V ')
        except:
            future_subj = 0
        try:
            imperative = len(re.findall('V.*IMP ', flat)) / flat.count(' V ')
        except:
            imperative = 0
        tenses = [present_ind, imperfect_ind, perfect_ind, pluperfect_ind,
                  future_ind, conditional_ind, present_subj, imperfect_subj,
                  future_subj, imperative]
        return tenses[8] / sum(tenses)


# class ImperativeMoodRatio(base.Metric):
#     """
#         **Nome da Métrica**: imperative_ratio
#
#         **Interpretação**: verbos no modo imperativo sugerem diálogo com o leitor, o que pode estar associado à
#         facilitação da leitura. Portanto, quanto maior a proporção, menor a complexidade.
#
#         **Descrição da métrica**: Proporção de verbos no modo imperativo em relação ao total de verbos
#
#         **Definição dos termos que aparecem na descrição da métrica**:
#
#         **Forma de cálculo da métrica**: contam-se todas as ocorrências de V*IMP e divide-se o resultado pela quantidade
#         total de verbos.
#
#         **Recursos de PLN utilizados durante o cálculo**: parser Palavras
#
#         **Limitações da métrica**: a precisão da métrica depende do desempenho do parser Palavras. As formas mais usadas
#         do imperativo são as segundas pessoas. Como no português do Brasil as segundas pessoas mais usadas são “você” e
#         “vocês” (ao invés de “tu” e “vós”), que conjugam como terceiras pessoas, o imperativo acaba não sendo
#         reconhecido nessas pessoas, a forma verbal é ambígua com o presente do subjuntivo e, diante de dessas formas
#         ambíguas, o parser prioriza a forma do subjuntivo em detrimento da forma do imperativo (já que o imperativo na
#         terceira pessoa é raro em português europeu).
#
#         **Crítica**: Como o parser prioriza a forma do subjuntivo, esta métrica não consegue ser capturada e o resultado
#         sempre dará 0, motivo pelo qual é melhor comentá-la
#
#         **Projeto**: GUTEN
#
#         **Teste**: Não precisa viajar para comer: confira como fazer em casa uma torta de abóbora americana incrível!
#
#         **Contagens**:
#
#         2 verbos flexionados, 1 no presente do indicativo (precisa) e 1 no imperativo (confira).
#
#         O parser, contudo, anotou “confira” como terceira pessoa do presente do subjuntivo, quando na verdade se trata
#         da segunda pessoa do imperativo. Como as formas são ambíguas, o parser teria que usar pistas de contexto para
#         desambiguizar e anotar corretamente.
#
#         **Resultado Esperado**: 0,50 (1/2)
#
#         **Resultado Obtido**: 0
#
#         **Status**: correto, considerando limitação do parser
#     """
#
#     name = 'Ratio of Imperative mood'
#     column_name = 'imperative_ratio'
#
#     def value_for_text(self, t, rp=default_rp):
#         flat = rp.palavras_flat(t)
#         try:
#             present_ind = len(re.findall('V.*PR.*IND', flat)) / flat.count(' V ')
#         except ZeroDivisionError:
#             present_ind = 0
#         try:
#             imperfect_ind = len(re.findall('V.*IMPF.*IND', flat)) / flat.count(' V ')
#         except ZeroDivisionError:
#             imperfect_ind = 0
#         try:
#             perfect_ind = len(re.findall('V.*PS.*IND', flat)) / flat.count(' V ')
#         except ZeroDivisionError:
#             perfect_ind = 0
#         try:
#             pluperfect_ind = len(re.findall('V.*MQP.*IND', flat)) / flat.count(' V ')
#         except ZeroDivisionError:
#             pluperfect_ind = 0
#         try:
#             future_ind = len(re.findall('V.*FUT.*IND', flat)) / flat.count(' V ')
#         except ZeroDivisionError:
#             future_ind = 0
#         try:
#             conditional_ind = len(re.findall('V.*COND', flat)) / flat.count(' V ')
#         except ZeroDivisionError:
#             conditional_ind = 0
#         try:
#             present_subj = len(re.findall('V.*PR.*SUBJ', flat)) / flat.count(' V ')
#         except:
#             present_subj = 0
#         try:
#             imperfect_subj = len(re.findall('V.*IMPF.*SUBJ', flat)) / flat.count(' V ')
#         except:
#             imperfect_subj = 0
#         try:
#             future_subj = len(re.findall('V.*FUT.*SUBJ', flat)) / flat.count(' V ')
#         except:
#             future_subj = 0
#         try:
#             imperative = len(re.findall('V.*IMP ', flat)) / flat.count(' V ')
#         except:
#             imperative = 0
#         tenses = [present_ind, imperfect_ind, perfect_ind, pluperfect_ind,
#                   future_ind, conditional_ind, present_subj, imperfect_subj,
#                   future_subj, imperative]
#         return tenses[9] / sum(tenses)


class VerbalMoodsTimeDiversity(base.Metric):
    """
        **Nome da Métrica**: verbal_time_moods_diversity

        **Interpretação**: quanto maior a diversidade de tempos e modos verbais, maior a complexidade

        **Descrição da métrica**: Quantidade de diferentes tempos e modos verbais no texto

        **Definição dos termos que aparecem na descrição da métrica**: há 10 tempos e modos verbais identificados pela
        métrica: indicativo: presente, pretérito imperfeito, pretérito perfeito, pretérito mais que perfeito, futuro,
        futuro do pretérito; subjuntivo: presente, pretérito imperfeito, futuro; imperativo. Portanto, o resultado pode
        variar de 0 a 10.

        **Forma de cálculo da métrica**: contam-se os diferentes tipos de tempo e modo verbal que ocorrem no texto.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**: a precisão da métrica depende do desempenho do parser Palavras

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: Ele fizera questão de enfatizar que, embora houvesse chegado tarde, havia telefonado antecipadamente
        para avisar do atraso. A empresa não quer ouvir os argumentos dele e afirmou que fará de tudo para demiti-lo
        dentro da lei.

        **Contagens**: 6 tempos/modo verbais (fizera, houvesse, havia, quer, afirmou, fará)

        **Resultado Esperado**: 6

        **Resultado Obtido**: 6

        **Status**: correto
    """

    name = 'Diversity of verbal time and mode inflections'
    column_name = 'verbal_time_moods_diversity'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        try:
            present_ind = len(re.findall('V.*PR.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            present_ind = 0
        try:
            imperfect_ind = len(re.findall('V.*IMPF.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            imperfect_ind = 0
        try:
            perfect_ind = len(re.findall('V.*PS.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            perfect_ind = 0
        try:
            pluperfect_ind = len(re.findall('V.*MQP.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            pluperfect_ind = 0
        try:
            future_ind = len(re.findall('V.*FUT.*IND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            future_ind = 0
        try:
            conditional_ind = len(re.findall('V.*COND', flat)) / flat.count(' V ')
        except ZeroDivisionError:
            conditional_ind = 0
        try:
            present_subj = len(re.findall('V.*PR.*SUBJ', flat)) / flat.count(' V ')
        except:
            present_subj = 0
        try:
            imperfect_subj = len(re.findall('V.*IMPF.*SUBJ', flat)) / flat.count(' V ')
        except:
            imperfect_subj = 0
        try:
            future_subj = len(re.findall('V.*FUT.*SUBJ', flat)) / flat.count(' V ')
        except:
            future_subj = 0
        try:
            imperative = len(re.findall('V.*IMP ', flat)) / flat.count(' V ')
        except:
            imperative = 0
        tenses = [present_ind, imperfect_ind, perfect_ind, pluperfect_ind,
                  future_ind, conditional_ind, present_subj, imperfect_subj,
                  future_subj, imperative]

        return sum([1 for i in tenses if i > 0])



class InfiniteSubordinateClauses(base.Metric):
    """
        **Nome da Métrica**: infinite_subordinate_clauses

        **Interpretação**:.quanto maior o resultado da métrica, maior a complexidade

        **Descrição da métrica**: Proporção de orações subordinadas reduzidas em relação à quantidade de orações do
        texto

        **Definição dos termos que aparecem na descrição da métrica**: orações subordinadas reduzidas são aquelas
        formadas por verbos nas formas nominais (ou infinitas): infinitivo, gerúndio e particípio

        **Forma de cálculo da métrica**: contam-se as ocorrências de verbo principal nas formas infinitas
        (<mv>.*V (INF|PCP|GER)(?!.*ICL-AUX) e divide-se pelo total de orações (V – aux).

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras

        **Limitações da métrica**:

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: Ele, determinado a entrar na universidade e sempre estudando horas a fio, foi o único a se lembrar
        do prazo final para inscrição no vestibular.

        **Contagens**: 5 verbos, 4 nas formas infinitas (determinado, entrar, estudando, lembrar)

        **Resultado Esperado**: 0,80

        **Resultado Obtido**: 0,80

        **Status**: correto
    """

    name = 'Ratio of Infinite Subordinate Clauses per Number of Clauses'
    column_name = 'infinite_subordinate_clauses'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        infinite_subordinate = re.findall(r"<mv>.*V (INF|PCP|GER)(?!.*ICL-AUX)", flat)
        return len(infinite_subordinate) / rp.num_clauses(t)


class SubordinateClauses(base.Metric):
    """
        **Nome da Métrica**: subordinate_clauses

        **Interpretação**: as orações subordinadas são estruturas mais complexas, que demandam mais esforço de
        processamento. Portanto, quanto maior o resultado, maior a complexidade.

        **Descrição da métrica**: proporção de orações subordinadas em relação a todas orações do texto.

        **Definição dos termos que aparecem na descrição da métrica**:

        **Forma de cálculo da métrica**: computam-se as orações subordinadas reduzidas de infinitivo, gerúndio e
        particípio. Para isso, buscam-se verbos nas formas infinitivas, excluindo-se aqueles precedidos de verbos
        auxiliares (<mv>.*V (INF|PCP|GER) menos ICL-AUX); somam-se, a essas, as orações subordinadas introduzidas por
        conjunções subordinativas (KS) e orações subordinadas relativas (<rel>). Essa métrica tem origem no AIC, mas
        foi refeita durante do convênio ICMC-GUTEN.

        **Recursos de PLN utilizados durante o cálculo**: parser Palavras.

        **Limitações da métrica**: a precisão da métrica depende do desempenho do parser na atribuição das etiquetas
        utilizadas no cálculo.

        **Crítica**:

        **Projeto**: GUTEN

        **Teste**: Ele e amigos, como Giovane Silva Ferreira, 13 anos, passam as tardes pescando o peixe, depois levado
        para uma associação de artesãos que faz o curtimento da pele do animal.

        **Contagens**: 4 orações (passam, pescando, levado, faz), 1 subordinada reduzida de gerúndio (pescando), 1
        subordinada reduzida de particípio (levado), 1 subordinada relativa (que faz).

        **Resultado Esperado**: 0,75

        **Resultado Obtido**: 0,75

        **Status**: correto
    """

    name = 'Ratio of Subordinate Clauses per Number of Clauses'
    column_name = 'subordinate_clauses'

    def value_for_text(self, t, rp=default_rp):
        flat = rp.palavras_flat(t)
        num_subordinate = re.findall(r"<mv>.*V (INF|PCP|GER)(?!.*ICL-AUX)| KS |<rel>", flat)
        return len(num_subordinate) / rp.num_clauses(t)


class GUTEN_Palavras(base.Category):
    name = 'GUTEN (Palavras)'
    table_name = 'guten_palavras'

    def __init__(self):
        super(GUTEN_Palavras, self).__init__()
        self._set_metrics_from_module(__name__)
        self.metrics = [m for m in self.metrics if m.name != 'AnaphoricReferencesBase']
