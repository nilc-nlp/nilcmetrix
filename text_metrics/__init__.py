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

from text_metrics.base import (
    Text,
    Category,
    Metric,
    MetricsSet,
    ResultSet,
)
from text_metrics.metrics.extra import EXTRA

from text_metrics.resource_pool import (
    ResourcePool,
    DefaultResourcePool,
    rp,
)

from text_metrics.metrics import *
from text_metrics.tools import *
from text_metrics.conf import config

# XXX: this is obsolete. It will be removed in future versions.
# It's here only for back compatibility.
all_metrics = MetricsSet([BasicCounts(),
                          LogicOperators(),
                          Frequencies(),
                          Hypernyms(),
                          Tokens(),
                          Connectives(),
                          Ambiguity(),
                          # SyntacticalComplexity(),
                          Category([YngveComplexity(),
                                   FrazierComplexity(),
                                   DependencyDistance()],
                                   name='Syntactical Complexity',
                                   table_name='syntax'),
                          # SemanticDensity(),
                          Category([ContentDensity()],
                                   name='Semantic Density',
                                   table_name='semantic_density'),
                          Constituents(),
                          Anaphoras(),
                          Coreference(),
                          Lsa(),
                          # Disfluencies(),
                          AIC(),
                          AIC_Palavras(),
                          LIWC(),
                          GUTEN(),
                          GUTEN_Palavras(),
                          EXTRA(),
                          ])


ALL_METRICS = MetricsSet([BasicCounts(),
                          LogicOperators(),
                          Frequencies(),
                          Hypernyms(),
                          Tokens(),
                          Connectives(),
                          Ambiguity(),
                          SyntacticalComplexity(),
                          Category([ContentDensity(),],
                                   name='Semantic Density',
                                   table_name='semantic_density'),
                          Constituents(),
                          Anaphoras(),
                          Coreference(),
                          Lsa(),
                          Disfluencies(),
                         ])


CMP_METRICS = MetricsSet([BasicCounts(),
                          LogicOperators(),
                          Frequencies(),
                          Hypernyms(),
                          Category([PersonalPronounsRatio(),
                                    # PronounsPerNounPhrase(),
                                    TypeTokenRatio()],
                                   name='Tokens',
                                   table_name="tokens"),
                          Constituents(),
                          Connectives(),
                          Ambiguity(),
                          Coreference(),
                          Anaphoras(),
                          Category([MeanNounPhrase()],
                                   name='Mean Noun Phrase',
                                   table_name='mean_noun_phrase'),
                         ])


NEW_METRICS = MetricsSet([Category([BrunetIndex(),
                                    HoroneStatistic(),
                                    # MeanClauseSentence(),
                                   ],
                                   name='Tokens',
                                   table_name="tokens"),
                          SyntacticalComplexity(),
                          Category([ContentDensity()],
                                   name='Semantic Density',
                                   table_name='semantic_density'),
                          Lsa(),
                          Disfluencies(),
                         ])


COMMENTS_METRICS = MetricsSet([ManualPrint()])


sentence_metrics = MetricsSet([BasicCounts(),
                          LogicOperators(),
                          #Frequencies(),
                          #Hypernyms(),
                          Tokens(),
                          Connectives(),
                          Ambiguity(),
                          # SyntacticalComplexity(),
                          Category([#YngveComplexity(),
                                   FrazierComplexity(),
                                   DependencyDistance()],
                                   name='Syntactical Complexity',
                                   table_name='syntax'),
                          # SemanticDensity(),
                          #Category([ContentDensity()],
                          #         name='Semantic Density',
                          #         table_name='semantic_density'),
                          #Constituents(),
                          #Anaphoras(),
                          #Coreference(),
                          #Lsa(),
                          # Disfluencies(),
                          AIC(),
                          AIC_Palavras(),
                          # LIWC(),
                          GUTEN(),
                          GUTEN_Palavras(),
                          EXTRA(),
                          ])

nilc_metrics = MetricsSet([BasicCounts(),
                          LogicOperators(),
                          Frequencies(),
                          Hypernyms(),
                          Tokens(),
                          Connectives(),
                          Ambiguity(),
                          Category([YngveComplexity(),
                                   FrazierComplexity(),
                                   DependencyDistance(),
                                   CrossEntropy()],
                                   name='Syntactical Complexity',
                                   table_name='syntax'),
                          Category([ContentDensity()],
                                   name='Semantic Density',
                                   table_name='semantic_density'),
                          Constituents(),
                          Anaphoras(),
                          Coreference(),
                          Lsa(),
                          AIC(),
                          AIC_Palavras(),
                          LIWC(),
                          GUTEN(),
                          GUTEN_Palavras(),
                          EXTRA(),
                          ])

no_palavras_metrics = MetricsSet([BasicCounts(),
                          LogicOperators(),
                          Frequencies(),
                          Hypernyms(),
                          Tokens(),
                          Connectives(),
                          Ambiguity(),
                          Category([YngveComplexity(),
                                   FrazierComplexity(),
                                   DependencyDistance(),
                                   CrossEntropy()],
                                   name='Syntactical Complexity',
                                   table_name='syntax'),
                          Category([ContentDensity()],
                                   name='Semantic Density',
                                   table_name='semantic_density'),
                          #Constituents(),
                          Anaphoras(),
                          Coreference(),
                          Lsa(),
                          AIC(),
                          LIWC(),
                          GUTEN(),
                          EXTRA(),
                          ])

rp = DefaultResourcePool()


__all__ = sorted([m for m in locals().keys()
                  if not m.startswith('_')])
