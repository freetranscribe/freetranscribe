"""Command line for FreeTranscribe's open tooling."""

import argparse
import json

from .parakeet.decode import MAX_SECONDS, OVERLAP_SECONDS


def transcribe(args: argparse.Namespace) -> None:
    import soundfile

    from .parakeet import load_model
    from .parakeet import transcribe as decode

    try:
        from nemo.utils import logging
    except ImportError:
        pass  # load_model reports the missing extra
    else:
        # NeMo logs to standard output, where it would mix with the transcript.
        logging.setLevel(logging.ERROR)
    asr = load_model(args.model, args.device)
    for path in args.audio:
        wave, rate = soundfile.read(path, dtype="float32")
        result = decode(asr, wave, rate, max_seconds=args.max_seconds,
                        overlap_seconds=args.overlap_seconds, batch_size=args.batch_size)
        if args.json:
            print(json.dumps({"audio": path, **result}))
        else:
            print(result["text"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="freetranscribe", description="Open tooling from FreeTranscribe.org.")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "transcribe",
        help="transcribe recordings with a Parakeet model, cutting long ones at pauses",
        description="Transcribe recordings with a Parakeet model, cutting long ones at pauses.",
    )
    command.add_argument("audio", nargs="+", help="audio files readable by soundfile")
    command.add_argument("--model", default="freetranscribe/parakeet-ft-v1",
                         help="a .nemo file or a Hugging Face model id")
    command.add_argument("--device", default="cuda")
    command.add_argument("--max-seconds", type=float, default=MAX_SECONDS,
                         help="longest piece decoded at once")
    command.add_argument("--overlap-seconds", type=float, default=OVERLAP_SECONDS,
                         help="extra audio decoded on each side of a piece")
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--json", action="store_true",
                         help="print one JSON object per file with word times")
    command.set_defaults(run=transcribe)
    args = parser.parse_args(argv)
    args.run(args)
