# T5Gemma 2 vLLM plugin

Standalone vLLM plugin for serving `google/t5gemma-2-1b-1b` and compatible
Speculators/DFlash checkpoints trained against that model.

The plugin packages the Python model adapter, text encoder-decoder processor,
merged self+cross decoder attention, and the small Triton kernels needed by the
adapter. It does not require this repository's `vllm-factory` checkout at
runtime.

## Requirements

- Linux or WSL2 with an NVIDIA GPU.
- A vLLM version with v1 speculative decoding and DFlash support.
  The version used during development was the vLLM nightly wheel built from
  commit `e4b3da3feb20c1854a4b23e431cfb787ee268f72`.
- Hugging Face access to `google/t5gemma-2-1b-1b`.
- For source installs, a compiler toolchain and Python headers are required
  because Triton/FlashInfer may JIT compile kernels in the target environment.

## Install

From this repository:

```bash
pip install -e ./t5gemma2-vllm-plugin
```

If the environment was created with `uv` and has no `pip` module:

```bash
uv pip install -e ./t5gemma2-vllm-plugin
```

After installation, vLLM discovers the plugin through the package entry point:

```toml
[project.entry-points."vllm.general_plugins"]
t5gemma2_vllm_plugin = "t5gemma2_vllm_plugin:register"
```

In a clean environment no extra environment variable is required. vLLM will load
installed `vllm.general_plugins` entry points automatically.

If the environment contains multiple vLLM plugins and you want to load only this
one, filter plugin loading explicitly:

```bash
export VLLM_PLUGINS=t5gemma2_vllm_plugin
```

This is optional. It is useful for development environments where unrelated
plugins may fail to import or register conflicting model architectures.

## Serve raw T5Gemma 2

```bash
vllm serve google/t5gemma-2-1b-1b \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name t5gemma-2-1b-1b \
  --trust-remote-code \
  --no-enable-chunked-prefill \
  --max-model-len 512 \
  --max-num-seqs 1
```

## Serve a T5Gemma 2 DFlash checkpoint

The DFlash checkpoint must be in Speculators format and its `config.json` must
contain:

```json
{
  "architectures": ["DFlashDraftModel"],
  "speculators_config": {
    "algorithm": "dflash",
    "verifier": {
      "architectures": ["T5Gemma2ForConditionalGeneration"],
      "name_or_path": "google/t5gemma-2-1b-1b"
    }
  }
}
```

Run:

```bash
vllm serve /path/to/t5gemma-2-1b-1b.dflash \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name t5gemma-2-1b-1b-dflash \
  --trust-remote-code \
  --no-enable-chunked-prefill \
  --max-model-len 512 \
  --max-num-seqs 1
```

vLLM reads the Speculators config from the checkpoint, loads the verifier
`google/t5gemma-2-1b-1b`, and enables DFlash speculative decoding.

## Quick request

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "t5gemma-2-1b-1b-dflash",
    "prompt": "Solve the problem step by step.\n\nQuestion: A train travels 60 miles in 2 hours. What is its average speed?\nAnswer:",
    "max_tokens": 64,
    "temperature": 0
  }'
```

## Acceptance metrics

DFlash token acceptance is exposed by vLLM Prometheus metrics:

- `vllm:spec_decode_num_drafts_total`
- `vllm:spec_decode_num_draft_tokens_total`
- `vllm:spec_decode_num_accepted_tokens_total`
- `vllm:spec_decode_num_accepted_tokens_per_pos_total`

Example:

```bash
curl -s http://127.0.0.1:8000/metrics | grep 'spec_decode'
```

Acceptance rate is:

```text
accepted_tokens / draft_tokens
```

Mean acceptance length including the bonus token is:

```text
1 + accepted_tokens / drafts
```

## Notes and limitations

- The adapter targets text encoder inputs. The T5Gemma 2 vision path is present
  in the vendored model code but has not been benchmarked as part of this
  plugin packaging.
- The merged decoder attention path is designed for FlashAttention/vLLM v1 on a
  single GPU. Tensor parallel and quantized serving should be validated before
  production use.
- The package intentionally vendors only the code required to register and serve
  the T5Gemma 2 generation architecture. It does not include training code.
