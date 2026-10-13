import itertools
import unittest

import numpy as np

from freetranscribe.parakeet.segment import HOP_SECONDS, energy_scores, mincut


class MincutTest(unittest.TestCase):
    def setUp(self):
        # 100 seconds of speech with pauses at 20-21 s, 44-45 s, and 70-80 s.
        self.scores = np.ones(5000)
        for begin, finish in ((1000, 1050), (2200, 2250), (3500, 4000)):
            self.scores[begin:finish] = 0.0

    def test_pieces_cover_the_recording_without_gaps(self):
        pieces = mincut(self.scores, 30)
        self.assertEqual(pieces[0][0], 0.0)
        self.assertAlmostEqual(pieces[-1][1], 100.0)
        for (_, finish), (begin, _) in itertools.pairwise(pieces):
            self.assertEqual(finish, begin)

    def test_cuts_fall_in_pauses_and_respect_the_limit(self):
        pieces = mincut(self.scores, 30)
        for begin, finish in pieces:
            self.assertLessEqual(finish - begin, 30.0 + 1e-9)
        for _, finish in pieces[:-1]:
            self.assertEqual(self.scores[round(finish / HOP_SECONDS)], 0.0)

    def test_short_recording_is_one_piece(self):
        self.assertEqual(mincut(np.ones(500), 30), [(0.0, 10.0)])


class EnergyScoresTest(unittest.TestCase):
    def test_speech_scores_high_and_silence_low(self):
        rng = np.random.default_rng(0)
        wave = np.concatenate([
            rng.normal(0, 0.2, 16000 * 3), rng.normal(0, 0.001, 16000 * 2),
            rng.normal(0, 0.2, 16000 * 3),
        ]).astype(np.float32)
        scores = energy_scores(wave, 16000)
        self.assertEqual(len(scores), 400)
        self.assertGreater(scores[50], 0.8)
        self.assertLess(scores[200], 0.2)


if __name__ == "__main__":
    unittest.main()
