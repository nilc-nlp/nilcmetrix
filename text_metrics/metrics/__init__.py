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

from text_metrics.metrics.basic_counts import *
from text_metrics.metrics.logic_ops import *
from text_metrics.metrics.freq import *
from text_metrics.metrics.hypernyms import *
from text_metrics.metrics.tokens import *
from text_metrics.metrics.connectives import *
from text_metrics.metrics.ambiguity import *
from text_metrics.metrics.syntax import *
from text_metrics.metrics.sem import *
from text_metrics.metrics.constituents import *
from text_metrics.metrics.anaphoras import *
from text_metrics.metrics.coref import *
from text_metrics.metrics.disfluencies import *
from text_metrics.metrics.lsa import *
from text_metrics.metrics.aic import *
from text_metrics.metrics.aic_palavras import *
from text_metrics.metrics.liwc import *
from text_metrics.metrics.guten import *
from text_metrics.metrics.guten_palavras import *
from text_metrics.metrics.manual import *
from text_metrics.metrics.manual_print import *
