#!/usr/bin/env python3
"""Prepare a vLLM tokenizer override without modifying the Hub model cache."""

import argparse
import json
from pathlib import Path
import shutil


MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]


def prepare(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("Use a separate destination; the source tokenizer must remain unchanged")
    tokenizer = json.loads((source / "tokenizer.json").read_text())
    if tokenizer.get("decoder", {}).get("type") != "ByteLevel":
        raise ValueError("Expected the DeepSeek Qwen3 ByteLevel tokenizer")
    config = json.loads((source / "tokenizer_config.json").read_text())
    # The model advertises LlamaTokenizerFast. In Transformers 5 that alias
    # constructs a Llama decoder, replacing the serialized ByteLevel pipeline.
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    config.pop("auto_map", None)
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if name != "tokenizer_config.json" and (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    (destination / "tokenizer_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def verify(destination: Path) -> None:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    raw = Tokenizer.from_file(str(destination / "tokenizer.json"))
    loaded = AutoTokenizer.from_pretrained(str(destination), local_files_only=True)
    for sample in ["Don Henley sang in 1994.\nNext line.", "Patrick Walter Brown MPP", "Café — 北京"]:
        ids = raw.encode(sample, add_special_tokens=False).ids
        if loaded.encode(sample, add_special_tokens=False) != ids:
            raise ValueError("Tokenizer override changed the model's input token IDs")
        if loaded.decode(ids, skip_special_tokens=True) != sample:
            raise ValueError("Tokenizer override did not preserve whitespace/Unicode")
    # Verify the model's serialized template, including its assistant marker.
    prompt = loaded.apply_chat_template(
        [{"role": "user", "content": "Return the words Don Henley."}],
        tokenize=False, add_generation_prompt=True,
    )
    config = json.loads((destination / "tokenizer_config.json").read_text())
    if (loaded.chat_template != config.get("chat_template")
            or "Don Henley" not in prompt or not prompt.endswith("<｜Assistant｜>")):
        raise ValueError("DeepSeek reasoning chat template was not preserved")
    print(f"Verified ByteLevel encoding, decoding, Unicode and chat template: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="existing model snapshot; otherwise use/download Hub tokenizer files")
    parser.add_argument("--output", type=Path, default=Path("models/deepseek-r1-bytelevel-tokenizer"))
    args = parser.parse_args()
    source = args.source
    if source is None:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            source = Path(snapshot_download(MODEL, allow_patterns=FILES, local_files_only=True))
        except LocalEntryNotFoundError:
            source = Path(snapshot_download(MODEL, allow_patterns=FILES))
    prepare(source, args.output)
    verify(args.output)


if __name__ == "__main__":
    main()
