# RATEX: RAf via pyTorch EXtension

![CI-Lass-Pass](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/aire-meta-bot/aeb41ce3096bc1aaeb671f2c58836a3f/raw/awslabs-ratex-ci-badge-last-pass.json)

It aims to bridge torch models and RAF backends.
Please refer to our [wiki](docs/) for more information.

## PyTorch Compatbility

The main branch is compatible with torch 1.12.0 stable release.

PyTorch upstream may include non back compatible changes, so we maintain the compaitble Ratex in
a feature branch `pt-nightly-compatible`. We will
1. Rebase `pt-nightly-compatible` regularly to catch up the latest Ratex.
2. Test `pt-nightly-compatible` with the latest PyTorch nightly regularly.
3. Merge `pt-nightly-compatible` to `main` when there is a new stable release.

Please switch to the `pt-nightly-compatible` branch and find `scripts/pinned_torch_nightly.txt`
to see the current compatible PyTorch nightly version.

