from __future__ import annotations

import asyncio

from .app import build_arg_parser, configure_logging, run_voice_mvp


def main() -> int:
    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args()
    return asyncio.run(run_voice_mvp(test_mode=args.test))


if __name__ == "__main__":
    raise SystemExit(main())
