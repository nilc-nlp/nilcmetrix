# -*- coding: utf-8 -*-
"""Behavioral tests for metrics in text_metrics/metrics/guten.py."""

import pytest

import text_metrics
from text_metrics.metrics.guten import (
    GunningFog,
    PunctuationRatio,
    PunctuationDiversity,
    AdverbsMin,
    AdverbsMax,
    AdverbsStandardDeviation,
)


def gunning_fox(text):
    return GunningFog().value_for_text(text_metrics.Text(text))


def ratio(text):
    return PunctuationRatio().value_for_text(text_metrics.Text(text))


def diversity(text):
    return PunctuationDiversity().value_for_text(text_metrics.Text(text))


# ---------------------------------------------------------------------------
# gunning_fox = 0.4 × (words / sentences + 100 × complex_words / words)
# A "complex" word has 3 or more syllables. The complex-word term is a
# percentage (0–100), so it dominates the score on dense text.
# ---------------------------------------------------------------------------

class TestGunningFog:

    def test_without_complex_words_only_the_length_term_remains(self):
        # No word has 3+ syllables, so the complex term is 0 and the score is
        # 0.4 × words / sentences = 0.4 × 5 / 1.
        assert gunning_fox("O gato bebeu o leite.") == pytest.approx(2.0)

    def test_all_words_complex_makes_the_complex_term_a_full_100(self):
        # Every word has 3+ syllables, so complex / words = 1 → the term is a
        # full 100%: 0.4 × (4 / 1 + 100) = 41.6.
        assert gunning_fox(
            "Professores estudiosos analisam estatísticas."
        ) == pytest.approx(41.6)

    @pytest.mark.parametrize(
        "label, text, expected",
        [
            # 0.4 × (6/1 + 100 × 4/6)
            ("one sentence, 4 of 6 complex",
             "A professora explicou a matéria complicada.", 29.066667),
            # 0.4 × (6/2 + 100 × 3/6)
            ("two sentences, 3 of 6 complex",
             "Comprei pão. Ela estuda medicina veterinária.", 21.2),
        ],
    )
    def test_complex_words_enter_as_a_percentage(self, label, text, expected):
        assert gunning_fox(text) == pytest.approx(expected), label


# ---------------------------------------------------------------------------
# punctuation_ratio = punctuation signs / all tokens
# ---------------------------------------------------------------------------

class TestPunctuationRatio:

    def test_text_without_punctuation_is_zero(self):
        assert ratio("casa azul grande") == 0.0

    @pytest.mark.parametrize(
        "label, text, expected",
        [
            # comma + exclamation over 4 tokens
            ("comma and bang", "Olá, mundo!", 0.5),
            # travessão counts as a sign: travessão + comma + period over 6 tokens
            ("travessão", "— Olá, disse ele.", 0.5),
            # reticências count as a single sign: ellipsis + period over 5 tokens
            ("ellipsis", "Espere... já vou.", 0.4),
            # every sign type at once: , ; : . ? over 12 tokens
            ("all sign types", "A casa, azul; o teto: branco. Sim?", 5 / 12),
        ],
    )
    def test_counts_each_punctuation_sign(self, label, text, expected):
        assert ratio(text) == pytest.approx(expected), label

    def test_numbers_are_not_punctuation(self):
        # Replacing a number with a same-length-in-tokens word leaves the ratio
        # unchanged: digits and slashes contribute no punctuation.
        assert ratio("Custou 1000 reais.") == ratio("Custou mil reais.")
        assert ratio("Em 25/12/2017, comemoramos.") == ratio(
            "Em dezembro, comemoramos."
        )

    def test_compound_word_hyphens_are_not_punctuation(self):
        # The hyphen joining a compound word is part of the word, not a sign.
        assert ratio("Reunião na terça-feira.") == ratio("Reunião na segunda.")


# ---------------------------------------------------------------------------
# punctuation_diversity = variety of punctuation signs among the signs used
# ---------------------------------------------------------------------------

class TestPunctuationDiversity:

    def test_text_without_punctuation_is_zero(self):
        assert diversity("casa azul grande") == 0.0

    def test_all_distinct_signs_is_maximal(self):
        assert diversity("A casa, azul; o teto: branco. Sim?") == pytest.approx(1.0)

    def test_repeated_signs_lower_diversity(self):
        # Three commas and a period are two types among four signs.
        assert diversity("Vim, vi, venci.") < diversity("Olá, mundo!")

    def test_numbers_are_not_punctuation(self):
        assert diversity("Custou 1000 reais.") == diversity("Custou mil reais.")

    def test_compound_word_hyphens_are_not_punctuation(self):
        assert diversity("Reunião na terça-feira.") == diversity("Reunião na segunda.")


def adverbs_min(text):
    return AdverbsMin().value_for_text(text_metrics.Text(text))


def adverbs_max(text):
    return AdverbsMax().value_for_text(text_metrics.Text(text))


def adverbs_std(text):
    return AdverbsStandardDeviation().value_for_text(text_metrics.Text(text))


# ---------------------------------------------------------------------------
# adverbs_min / _max / _standard_deviation — statistics of the per-sentence
# adverb ratio. "Adverb" includes ADV, PREP+ADV, and PDEN (denotative words
# such as "também", "só", "ainda"), matching the adverbs / adverbs_diversity
# definitions.
# ---------------------------------------------------------------------------

class TestAdverbsPerSentence:

    # Three sentences of three tokens each, one adverb-like word per sentence:
    # "rapidamente" (ADV), "também" (PDEN), "diretamente" (ADV).
    _TEXT = "Pedro correu rapidamente. Maria também correu. João correu diretamente."

    def test_pden_words_count_as_adverbs(self):
        # Counting the PDEN "também" makes every sentence 1/3 adverbs, so the
        # per-sentence minimum is 1/3 (not 0) and the spread is zero.
        assert adverbs_min(self._TEXT) == pytest.approx(1 / 3)
        assert adverbs_max(self._TEXT) == pytest.approx(1 / 3)
        assert adverbs_std(self._TEXT) == pytest.approx(0.0)
