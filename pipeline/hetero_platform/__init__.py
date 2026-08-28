# SPDX-License-Identifier: Apache-2.0
"""Deeploy platform for the hetero_soc board.

Deeploy is used exactly as the pipeline already uses it -- to turn ONNX into C
-- but with three engines instead of one, so each node is emitted as a call on
the core that runs it fastest: the CVA6 host, the Snitch cluster, or the
Snitch/Spatz pair.

Nothing here patches Deeploy; it is imported as a library and its Generic
parsers, layers and type checkers are reused. The only new pieces are the
templates that emit an offload instead of a direct call, the mapper that
chooses between engines, and the pass that brackets every node with a progress
beacon.
"""
