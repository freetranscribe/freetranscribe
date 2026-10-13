import importlib.util
import unittest

import numpy as np

from freetranscribe.parakeet import load_model
from freetranscribe.parakeet.decode import mono_16k


class InputTest(unittest.TestCase):
    def test_other_sample_rates_are_resampled_to_16_khz(self):
        time = np.arange(44100) / 44100
        wave = mono_16k(np.sin(2 * np.pi * 440 * time).astype(np.float32), 44100)
        self.assertEqual(len(wave), 16000)
        spectrum = np.abs(np.fft.rfft(wave))
        self.assertEqual(int(spectrum.argmax()), 440)

    def test_channels_are_averaged(self):
        stereo = np.stack([np.ones(16000), -np.ones(16000)], axis=1).astype(np.float32)
        self.assertTrue(np.allclose(mono_16k(stereo, 16000), 0))


@unittest.skipIf(importlib.util.find_spec("nemo"), "NeMo is installed")
class MissingNemoTest(unittest.TestCase):
    def test_loading_a_model_without_nemo_names_the_extra(self):
        with self.assertRaisesRegex(ImportError, r"freetranscribe\[nemo\]"):
            load_model("nvidia/parakeet-tdt-0.6b-v3")


if __name__ == "__main__":
    unittest.main()
