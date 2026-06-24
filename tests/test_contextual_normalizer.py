import unittest

from bg_text_normalizer import normalize as contextual_normalize
from bulgarian_normalization import normalize_for_mfa, normalize_with_punctuation


class ContextualNormalizerTests(unittest.TestCase):
    def normalize_for_inference(self, text):
        return normalize_with_punctuation(contextual_normalize(text))

    def test_date_is_expanded_before_mfa_cleanup(self):
        text = self.normalize_for_inference("Роден съм на 14.12.2004.")

        self.assertEqual(
            text,
            "роден съм на четиринадесети декември две хиляди четвърта година.",
        )
        self.assertEqual(
            normalize_for_mfa(text),
            "роден съм на четиринадесети декември две хиляди четвърта година",
        )

    def test_currency_and_date_are_expanded_together(self):
        self.assertEqual(
            self.normalize_for_inference("Сумата е 15 лв. на 21.04.2026."),
            (
                "сумата е петнадесет лева на двадесет и първи април "
                "две хиляди двадесет и шеста година."
            ),
        )

    def test_plain_decimal_keeps_fractional_part(self):
        self.assertEqual(
            self.normalize_for_inference("1,5 литра."),
            "едно цяло и пет литра.",
        )

    def test_phone_digits_are_spelled_out(self):
        self.assertEqual(
            self.normalize_for_inference("Тел: 0888123456"),
            "тел: нула осем осем осем едно две три четири пет шест",
        )


if __name__ == "__main__":
    unittest.main()
