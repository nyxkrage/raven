# Xet development package

Xet is an experimental package backed by the Rust
[`huggingface/xet-core`](https://github.com/huggingface/xet-core) sources. It is
excluded from normal Raven builds because those sources are not part of release
archives.

To build and test it locally, check out `xet-core` at `vendor/xet-core` and run:

```sh
RAVEN_BUILD_XET=true dune runtest dev/xet
```
