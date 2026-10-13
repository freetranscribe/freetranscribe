import subprocess
import sys
import unittest


class CliTest(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "freetranscribe", *arguments],
                              capture_output=True, text=True, check=False)

    def test_help_lists_the_transcribe_command(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("transcribe", result.stdout)

    def test_transcribe_help_runs_without_nemo_installed(self):
        result = self.run_cli("transcribe", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cutting long ones at pauses", result.stdout)


if __name__ == "__main__":
    unittest.main()
