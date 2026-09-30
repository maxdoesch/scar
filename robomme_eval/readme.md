# RoboMME Evaluation

`robomme_eval` is a standalone uv project — it is **not** part of the root
repo's uv workspace, because it pins `torch==2.9.1`/`transformers==4.57.3`
and depends on ManiSkill, which conflict with the main JAX-based
environment. It is installed into its own virtualenv, alongside the
`robomme` (`third_party/robomme_benchmark`) and `openpi-client` packages.

## Install
From the repository root:
```
uv venv .venv-robomme --python 3.11
uv pip install -e robomme_eval --python .venv-robomme
pip install flash-attn==2.8.3 --no-build-isolation # Optional, it may take 20-30 min to build, you can always choose 'sdpa' in transformer attn_impl
```

Test if the install is correct:
```
source .venv-robomme/bin/activate
robomme-eval-smoke-test
```

## Running evaluation
```
source .venv-robomme/bin/activate
robomme-eval --args.model_seed=7 --args.port=8000 --args.policy_name=<your_specify_policy_name> --args.model_ckpt_id=79999
```
(Equivalent to the old direct-script invocation from the former examples
tree — that form no longer works because the package now uses
relative imports. Use the `robomme-eval` command, or `python -m
robomme_eval.eval --args...` if you need to invoke without the entry point.)

## Structure
```
robomme_eval
├── pyproject.toml
├── readme.md
└── src/robomme_eval
    ├── __init__.py
    ├── env_runner.py           # A wrapper of robomme simulator for eval
    ├── eval.py                 # The main script for eval
    ├── simple_test.py          # sanity check
    └── utils.py
```
