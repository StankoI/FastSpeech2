import unittest

from bulgarian_normalization import (
    normalize_for_mfa,
    normalize_with_punctuation,
    prosody_words,
    render_prosody_words,
)
from prosody_alignment import align_with_prosody
from text import text_to_sequence
from textgrid_utils import Interval, Tier


class NormalizationTests(unittest.TestCase):
    def test_dual_representation(self):
        text = "Имам 12 ябълки, нали? Да — точно така!"
        self.assertEqual(
            normalize_for_mfa(text),
            "имам дванадесет ябълки нали да точно така",
        )
        self.assertEqual(
            normalize_with_punctuation(text),
            "имам дванадесет ябълки, нали? да — точно така!",
        )
        self.assertEqual(
            [token for _, token in prosody_words(text) if token],
            ["p_comma", "p_question", "p_dash", "p_exclamation"],
        )

    def test_decimal_comma_is_not_a_pause(self):
        self.assertEqual(normalize_with_punctuation("1,5 литра."), "едно пет литра.")

    def test_render_pairs_does_not_renormalize_words(self):
        self.assertEqual(
            render_prosody_words([("спокойной", "p_comma"), ("ночи", "p_period")]),
            "спокойной, ночи.",
        )

    def test_control_tokens_are_in_symbol_abi(self):
        sequence = text_to_sequence(
            "{a wb b p_comma v p_question}", ["bulgarian_cleaners"]
        )
        self.assertEqual(len(sequence), 6)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.words = Tier(
            "words",
            (
                Interval(0.0, 0.1, ""),
                Interval(0.1, 0.4, "аз"),
                Interval(0.4, 0.6, ""),
                Interval(0.6, 0.9, "тук"),
                Interval(0.9, 1.0, ""),
            ),
        )
        self.phones = Tier(
            "phones",
            (
                Interval(0.0, 0.1, ""),
                Interval(0.1, 0.25, "a"),
                Interval(0.25, 0.4, "z̪"),
                Interval(0.4, 0.6, ""),
                Interval(0.6, 0.75, "t̪"),
                Interval(0.75, 0.9, "u"),
                Interval(0.9, 1.0, ""),
            ),
        )

    def align(self, text):
        return align_with_prosody(
            self.phones, self.words, prosody_words(text), sampling_rate=100, hop_length=10
        )

    def test_word_boundary_keeps_acoustic_pause(self):
        tokens, durations, _, _ = self.align("аз тук")
        self.assertEqual(tokens, ["a", "z̪", "wb", "sp", "t̪", "u"])
        self.assertEqual(durations, [1, 2, 0, 2, 2, 1])
        self.assertEqual(sum(durations), 8)

    def test_punctuation_consumes_pause_without_changing_frames(self):
        tokens, durations, _, _ = self.align("аз, тук?")
        self.assertEqual(tokens, ["a", "z̪", "p_comma", "t̪", "u", "p_question"])
        self.assertEqual(durations, [1, 2, 2, 2, 1, 0])
        self.assertEqual(sum(durations), 8)


if __name__ == "__main__":
    unittest.main()
