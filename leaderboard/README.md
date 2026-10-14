# Open ASR Leaderboard evaluation

`run_eval_long.py` is the long-form evaluation script from the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)'s [NeMo evaluation Space](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-nemo), with one change: with `--longform` it decodes each recording through `freetranscribe.parakeet` instead of a single whole-recording NeMo call. It also reads audio samples that arrive as decoder objects, which current `datasets` releases produce and the upstream script stops on. Timing, manifest writing and scoring are untouched. The header comment in the file lists the changes.

[freetranscribe/open-asr-leaderboard-parakeet-ft](https://huggingface.co/spaces/freetranscribe/open-asr-leaderboard-parakeet-ft) is a copy of the NeMo evaluation Space with this script and the package installed in its `Dockerfile`. The decoding settings are the same for every dataset.

To reproduce the results table on the [model card](https://huggingface.co/freetranscribe/parakeet-ft-v1):

- Data: [hf-audio/asr-leaderboard-longform](https://huggingface.co/datasets/hf-audio/asr-leaderboard-longform) at revision `74e296a`, the last to include TED-LIUM, and [bezzam/coraal](https://huggingface.co/datasets/bezzam/coraal).
- Recipe columns: run this script with `--longform --device 0 --batch_size 8`.
- Leaderboard-decoding columns: run the Space's unmodified script.
- Scoring: the normalizer and compound-aware word alignment from the [leaderboard repository](https://github.com/huggingface/open_asr_leaderboard) at commit `090df12`.

The script and `LICENSE` in this directory are under the Apache License 2.0, as upstream. Everything else in the repository is MIT.
