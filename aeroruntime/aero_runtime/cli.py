"""CLI entry-point for aero-runtime.

Usage::

    # Single prompt
    aero-run model.aero --prompt "What is 2+2?" --model-id Qwen/Qwen3-1.7B

    # Interactive chat
    aero-run model.aero --interactive --model-id Qwen/Qwen3-1.7B

    # Just inspect the .aero file (no inference)
    aero-run model.aero --info
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


# ── Terminal UI (stdlib only: colors when TTY) ──────────────────────────────

def _color_enabled() -> bool:
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    return os.environ.get("NO_COLOR", "").strip() == ""

_COLORS = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "cyan": "\033[36m",
    "bold": "\033[1m",
    "yellow": "\033[33m",
}

def _c(name: str, text: str) -> str:
    if not _color_enabled() or name not in _COLORS:
        return text
    return f"{_COLORS[name]}{text}{_COLORS['reset']}"

def _section(title: str) -> str:
    return _c("cyan", f"\n  ◆ {title}")

def _cmd_info(args: argparse.Namespace) -> None:
    """Print .aero file metadata without loading weights."""
    try:
        from .loader import AeroModelLoader

        loader = AeroModelLoader(args.aero_path)
    except FileNotFoundError as e:
        print(f"aero-run: File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"aero-run: Invalid path or file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"aero-run: Could not read AERO file: {e}", file=sys.stderr)
        sys.exit(1)

    print(_c("bold", "\n  aero-run  ") + _c("dim", args.aero_path))
    print(_section("Model"))
    print(f"    path           {loader.path}")
    print(f"    model_name     {loader.model_name}")
    print(f"    architecture  {loader.architecture}")
    print(f"    source_format  {loader.source_format or '(none)'}")
    print(_section("Weights"))
    print(f"    tensor_count   {loader.tensor_count}")
    print(f"    file_size_mb   {round(loader.file_size_mb, 2)}")
    print()


def _cmd_generate(args: argparse.Namespace) -> None:
    """Load model and generate text."""
    try:
        import torch as _torch
    except ImportError:
        print(
            "aero-run: PyTorch is required for generation. Install with:\n"
            "  pip install aero-runtime[inference]",
            file=sys.stderr,
        )
        sys.exit(1)
    from .model import AeroModel

    # Pass dtype as string so model resolves it via _get_torch() (avoids NameError across modules)
    dtype_str = args.dtype if args.dtype in ("float32", "float16", "bfloat16") else "float32"

    try:
        model = AeroModel(
            args.aero_path,
            model_id=args.model_id or None,
            device=args.device,
            dtype=dtype_str,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"aero-run: {e}", file=sys.stderr)
        sys.exit(1)
    except NameError as e:
        if "torch" in str(e).lower():
            print(
                "aero-run: PyTorch not found in this Python. Use the same Python for run and install:\n"
                "  python -m pip install aero-runtime[inference]\n"
                "  python -m aero_runtime.cli model.aero --prompt 'Hi' --model-id Qwen/Qwen3-1.7B",
                file=sys.stderr,
            )
        else:
            print(f"aero-run: Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"aero-run: Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.interactive:
            _interactive_loop(model, args)
        else:
            prompt = args.prompt or "Hello!"
            print(_c("bold", "\n  Prompt"))
            print(_c("dim", "  ─────"))
            print(f"  {prompt}")
            print()
            output = model.generate(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(_c("bold", "  Output"))
            print(_c("dim", "  ──────"))
            print(f"  {output}")
            print()
    except ValueError as e:
        print(f"aero-run: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"aero-run: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"aero-run: Generation failed: {e}", file=sys.stderr)
        sys.exit(1)


def _interactive_loop(model, args: argparse.Namespace) -> None:
    """Simple interactive REPL."""
    print(_c("bold", "\n  Interactive  ") + _c("dim", "type 'quit' or Ctrl-C to exit"))
    print()
    while True:
        try:
            prompt = input(_c("cyan", "  You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(_c("dim", "\n  Bye!"))
            break
        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            print(_c("dim", "  Bye!"))
            break
        output = model.generate(
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        response = output
        if response.startswith(prompt):
            response = response[len(prompt):].lstrip()
        print(_c("green", "  Model: ") + response)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aero-run",
        description="Run inference directly from .aero model files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aero-run model.aero --info
  aero-run model.aero --prompt "Hello!" --model-id Qwen/Qwen3-1.7B
  aero-run model.aero --interactive --model-id Qwen/Qwen3-1.7B
  aero-run model.aero --prompt "Explain AI" --dtype float16 --device cuda
""",
    )
    parser.add_argument(
        "aero_path",
        help="Path to the .aero model file",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="HuggingFace model ID for config/tokenizer (auto-detected if omitted)",
    )
    parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Prompt text for generation",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive chat session",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print .aero file info and exit (no inference)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Weight dtype after loading (default: float32)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: cpu, cuda, mps, or auto (default: auto)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"aero-runtime {__import__('aero_runtime').__version__}",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.info:
        _cmd_info(args)
    elif args.prompt or args.interactive:
        _cmd_generate(args)
    else:
        # Default: show info
        print(_c("dim", "No --prompt or --interactive specified. Showing file info."))
        _cmd_info(args)
        print(
            _c("dim", "To run inference:")
            + f"\n  aero-run {args.aero_path} --prompt 'Hello!' --model-id <HF_MODEL_ID>\n"
        )


if __name__ == "__main__":
    main()
