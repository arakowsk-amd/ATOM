# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM Prefill/Decode Simulator Server.

Thin wrapper that defaults ``--simulator`` and delegates to the
OpenAI-compatible ``api_server.main()`` so all standard routes stay live.

Usage:
    python -m atom.entrypoints.simulator_server --model <model> [options]
"""

import sys

from atom.utils import set_ulimit


def main():
    # Inject --simulator into argv if the user didn't already pass it,
    # so the downstream arg parser sees it without requiring the flag.
    if "--simulator" not in sys.argv:
        sys.argv.insert(1, "--simulator")

    from atom.entrypoints.openai.api_server import main as api_main

    api_main()


if __name__ == "__main__":
    set_ulimit()
    main()
