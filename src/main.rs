use std::fs;
use std::io::{self, Read};

use anyhow::{bail, Result};
use clap::Parser;
use colored::Colorize;
use tiktoken_rs::tokenizer::Tokenizer;

/// Count tokens from stdin or a file.
#[derive(Parser)]
#[command(name = "tokc", version, about)]
struct Cli {
    /// Input file (reads from stdin if omitted)
    file: Option<String>,

    /// Model or encoding name (e.g. gpt-4o, opus-4.6, cl100k_base)
    #[arg(short, long, default_value = "gpt-4o")]
    model: String,

    /// Show colorized tokenization breakdown
    #[arg(short = 'c', long)]
    colorize: bool,

    /// List all supported model aliases
    #[arg(long)]
    list_models: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    Tiktoken(Tokenizer),
    Claude,
}

fn resolve_model(name: &str) -> Result<Backend> {
    let lower = name.to_lowercase();
    let lower = lower.as_str();

    match lower {
        // Direct encoding names
        "o200k_base" | "o200k" => Ok(Backend::Tiktoken(Tokenizer::O200kBase)),
        "o200k_harmony" => Ok(Backend::Tiktoken(Tokenizer::O200kHarmony)),
        "cl100k_base" | "cl100k" => Ok(Backend::Tiktoken(Tokenizer::Cl100kBase)),
        "p50k_base" | "p50k" => Ok(Backend::Tiktoken(Tokenizer::P50kBase)),
        "p50k_edit" => Ok(Backend::Tiktoken(Tokenizer::P50kEdit)),
        "r50k_base" | "r50k" | "gpt2" => Ok(Backend::Tiktoken(Tokenizer::R50kBase)),

        // OpenAI model aliases → o200k_base
        s if s.starts_with("gpt-4o")
            || s.starts_with("gpt-5")
            || s.starts_with("gpt-4.1")
            || s.starts_with("gpt-4.5")
            || s.starts_with("o1")
            || s.starts_with("o3")
            || s.starts_with("o4") =>
        {
            Ok(Backend::Tiktoken(Tokenizer::O200kBase))
        }

        // OpenAI model aliases → cl100k_base
        s if s.starts_with("gpt-4") || s.starts_with("gpt-3.5") => {
            Ok(Backend::Tiktoken(Tokenizer::Cl100kBase))
        }

        // Anthropic / Claude model aliases
        s if s.starts_with("claude")
            || s.starts_with("opus")
            || s.starts_with("sonnet")
            || s.starts_with("haiku") =>
        {
            Ok(Backend::Claude)
        }

        _ => bail!(
            "Unknown model or encoding: '{}'\nRun with --list-models to see supported names.",
            name
        ),
    }
}

const PALETTE: &[fn(&str) -> colored::ColoredString] = &[
    |s| s.red(),
    |s| s.green(),
    |s| s.yellow(),
    |s| s.blue(),
    |s| s.magenta(),
    |s| s.cyan(),
    |s| s.bright_red(),
    |s| s.bright_green(),
    |s| s.bright_yellow(),
    |s| s.bright_blue(),
    |s| s.bright_magenta(),
    |s| s.bright_cyan(),
];

fn print_colorized(tokens: &[String]) {
    for (i, tok) in tokens.iter().enumerate() {
        let color_fn = PALETTE[i % PALETTE.len()];
        let display = tok.replace('\n', "⏎\n").replace('\t', "→\t");
        print!("{}", color_fn(&display));
    }
    println!();
}

/// Decode sentencepiece-style BPE tokens (Ġ → space, Ċ → newline, etc.)
fn decode_spm_token(tok: &str) -> String {
    tok.chars()
        .map(|c| {
            // sentencepiece uses a byte-to-unicode mapping where:
            // Ġ (U+0120) = space (0x20), Ċ (U+010A) = newline (0x0A), etc.
            let code = c as u32;
            if code >= 0x100 {
                // Map back: the original byte = code - 0x100 for range 0x100..0x1FF
                // Actually the sentencepiece mapping is: byte 0x20 -> Ġ (0x120)
                // So byte = code - 0x100
                let byte = (code - 0x100) as u8;
                byte as char
            } else {
                c
            }
        })
        .collect()
}

fn read_input(file: Option<&str>) -> Result<String> {
    match file {
        Some(path) => Ok(fs::read_to_string(path)?),
        None => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            Ok(buf)
        }
    }
}

fn encoding_name(tokenizer: Tokenizer) -> &'static str {
    match tokenizer {
        Tokenizer::O200kBase => "o200k_base",
        Tokenizer::O200kHarmony => "o200k_harmony",
        Tokenizer::Cl100kBase => "cl100k_base",
        Tokenizer::P50kBase => "p50k_base",
        Tokenizer::P50kEdit => "p50k_edit",
        Tokenizer::R50kBase => "r50k_base",
        Tokenizer::Gpt2 => "gpt2",
    }
}

fn print_model_list() {
    let entries = [
        ("OpenAI models (o200k_base)", &[
            "gpt-4o, gpt-4o-mini",
            "gpt-4.1, gpt-4.1-mini, gpt-4.1-nano",
            "gpt-4.5",
            "gpt-5, gpt-5.2",
            "o1, o1-mini, o1-pro",
            "o3, o3-mini, o3-pro",
            "o4-mini",
        ] as &[&str]),
        ("OpenAI models (cl100k_base)", &[
            "gpt-4, gpt-4-turbo",
            "gpt-3.5-turbo",
        ]),
        ("Anthropic models (Claude tokenizer), may not be very accurate", &[
            "claude-3.5-sonnet, claude-3.5-haiku",
            "claude-3-opus, claude-3-sonnet, claude-3-haiku",
            "claude-4-opus, claude-4-sonnet",
            "opus-4.6, sonnet-4.6, haiku-4.5",
        ]),
        ("Direct encoding names", &[
            "o200k_base, o200k_harmony",
            "cl100k_base",
            "p50k_base, p50k_edit",
            "r50k_base (gpt2)",
        ]),
    ];

    println!("{}", "Supported models and encodings:".bold());
    println!();
    for (group, models) in entries {
        println!("  {}", group.underline());
        for m in models {
            println!("    {}", m);
        }
        println!();
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    if cli.list_models {
        print_model_list();
        return Ok(());
    }

    let backend = resolve_model(&cli.model)?;
    let text = read_input(cli.file.as_deref())?;

    if text.is_empty() {
        println!("0");
        return Ok(());
    }

    match backend {
        Backend::Tiktoken(tokenizer) => {
            let bpe = tiktoken_rs::get_bpe_from_tokenizer(tokenizer)?;
            let enc_name = encoding_name(tokenizer);

            if cli.colorize {
                let tokens = bpe.split_by_token(&text, true)?;
                let count = tokens.len();
                print_colorized(&tokens);
                println!();
                println!(
                    "{} {} (encoding: {})",
                    count.to_string().bold(),
                    "tokens".dimmed(),
                    enc_name.dimmed()
                );
            } else {
                let tokens = bpe.encode_with_special_tokens(&text);
                println!("{}", tokens.len());
            }
        }
        Backend::Claude => {
            if cli.colorize {
                let tokens = claude_tokenizer::tokenize(&text)?;
                let count = tokens.len();
                let token_strings: Vec<String> =
                    tokens.into_iter().map(|(_, s)| decode_spm_token(&s)).collect();
                print_colorized(&token_strings);
                println!();
                println!(
                    "{} {} (tokenizer: {})",
                    count.to_string().bold(),
                    "tokens".dimmed(),
                    "claude".dimmed()
                );
            } else {
                let count = claude_tokenizer::count_tokens(&text)?;
                println!("{}", count);
            }
        }
    }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{}: {}", "error".red().bold(), e);
        std::process::exit(1);
    }
}
