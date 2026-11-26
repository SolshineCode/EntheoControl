"""ONLINE plugin: LLM assistant (Claude Code CLI / agy) + Nano Banana image engine. OFF by default.

Only `protocol` (pure Python: command validation + the hooks control.py calls) is imported while ONLINE is off.
Everything that can start a subprocess or touch the network lives in the other modules, which are imported by
`protocol._load()` the moment the performer switches ONLINE on, and not before. See HANDOFF.md "Online plugin".
"""
