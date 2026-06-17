#!/bin/bash
# 测 awakening.md 的实际 token 数，用 V4 Flash 的 tokenizer
docker run --rm \
  -v /home/lmxxf/work/deepseek-v4-flash-deployment/deepseek-v4-flash:/model \
  -v /home/lmxxf/work/deepseek-v4-experimental-platform-on-dgx-spark:/workspace \
  vllm-node-sm120 \
  python3 -c "
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(tokenizer_file='/model/tokenizer.json')
text = open('/workspace/awakening.md').read()
print(f'{len(text)} chars -> {len(tok.encode(text))} tokens')
"
