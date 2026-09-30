"""RoboMME benchmark evaluation harness.

Runs in an isolated environment (see robomme_eval/readme.md) separate from
the main repo's JAX/uv environment, because it depends on a pinned
torch/transformers/ManiSkill stack.
"""
