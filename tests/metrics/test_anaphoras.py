# -*- coding: utf-8 -*-
"""Behavioral tests for metrics in text_metrics/metrics/anaphoras.py.

The four anaphora metrics (adjacent_refs, anaphoric_refs,
coreference_pronoun_ratio, demonstrative_pronoun_ratio) share
AnaphoricReferencesBase: each is the mean number of candidate referents per
anaphoric pronoun. AdjacentAnaphoricReferences exercises that shared logic
while looking only at the immediately preceding sentence, which keeps the
candidate counts small enough to state by hand.
"""

import pytest

import text_metrics
from text_metrics.metrics.anaphoras import AdjacentAnaphoricReferences


def adjacent_refs(text):
    return AdjacentAnaphoricReferences().value_for_text(text_metrics.Text(text))


# ---------------------------------------------------------------------------
# adjacent_refs = candidate referents / anaphoric pronouns
# (averaged per pronoun, not per sentence)
# ---------------------------------------------------------------------------

class TestAdjacentAnaphoricReferences:

    def test_one_candidate_per_one_pronoun(self):
        # "Elas" (fem. plural) has one agreeing referent — "meninas" — in the
        # preceding sentence; the last sentence has no pronoun. One candidate
        # for one pronoun → 1.0 (averaged over the single pronoun, not the two
        # sentence transitions).
        assert adjacent_refs(
            "As meninas chegaram. Elas sorriram. O dia terminou."
        ) == pytest.approx(1.0)

    def test_averaged_over_every_pronoun(self):
        # Two pronouns — "Elas" (→ meninas) and "eles" (→ meninos) — each with
        # one agreeing referent. Two candidates over two pronouns → 1.0.
        assert adjacent_refs(
            "As meninas e os meninos chegaram. Elas sorriram e eles acenaram."
        ) == pytest.approx(1.0)

    def test_a_pronoun_can_have_several_candidates(self):
        # "Elas" agrees with two referents — "meninas" and "professoras". One
        # pronoun with two candidates → 2.0.
        assert adjacent_refs(
            "As meninas e as professoras chegaram. Elas sorriram. O sino tocou."
        ) == pytest.approx(2.0)
