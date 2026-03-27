# tokc

Count tokens from stdin or files, with colorized tokenization display.

Supports OpenAI (GPT-3.5, GPT-4, GPT-4o, o1, o3, etc.) and Anthropic (Claude) models.

## Install

```sh
cargo install tokc
```

Or build from source:

```sh
git clone https://github.com/abmfy/tokc.git
cd tokc
cargo install --path .
```

## Usage

```sh
# Count tokens from a file (default: gpt-4o)
tokc input.txt

# Count tokens from stdin
cat input.txt | tokc

# Specify a model
tokc -m claude-3.5-sonnet input.txt
tokc -m gpt-4 input.txt

# Show colorized tokenization breakdown
tokc -c input.txt

# List all supported models and encodings
tokc --list-models
```

## Options

```
  [FILE]                Input file (reads from stdin if omitted)
  -m, --model <MODEL>   Model or encoding name [default: gpt-4o]
  -c, --colorize        Show colorized tokenization breakdown
      --list-models     List all supported model aliases
  -h, --help            Print help
  -V, --version         Print version
```

## Supported Models

| Group | Models | Encoding |
|---|---|---|
| OpenAI (latest) | gpt-4o, gpt-4.1, gpt-4.5, gpt-5, o1, o3, o4 | o200k_base |
| OpenAI (older) | gpt-4, gpt-3.5-turbo | cl100k_base |
| Anthropic | claude-3/4, opus, sonnet, haiku | Claude tokenizer |

You can also use encoding names directly: `o200k_base`, `cl100k_base`, `p50k_base`, `r50k_base`, etc.

## License

MIT
