// build.rs
use std::{
    env,
    io::{BufReader, BufWriter, Write},
    fs::File,
    path::{Path, PathBuf},
};
use literate::{LiterateError, CodeMatcher};
use cmd_lib::run_cmd;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    process_file("docs/public/content/home/bouncing-ball.md", "bouncing-ball.py", Language::Python);
    process_file("docs/public/content/home/3-body.md", "3-body.py", Language::Python);
}

#[derive(Clone, Copy, Debug)]
enum Language {
    Rust,
    Python,
}

impl Language {
    fn extension(&self) -> &'static str {
        match self {
            Language::Rust => "rs",
            Language::Python => "py",
        }
    }

    fn block(&self) -> &'static str {
        match self {
            Language::Rust => "rust",
            Language::Python => "python",
        }
    }

    fn comment(&self) -> &'static str {
        match self {
            Language::Rust => "//",
            Language::Python => "#",
        }
    }
}

/// Provide the path from the root elodin directory.
fn process_file(source: &str, output_file_name: &str, language: Language) {

    let mut output = PathBuf::from(env::var("OUT_DIR").unwrap());
    output.push(output_file_name);
    // Tell Cargo to re-run the build script if the Markdown file changes
    println!("cargo:rerun-if-changed={}", source);

    let mut input = PathBuf::from("../..");
    input.push(source);
    let _ = extract_file(&input, &output, language)
        .expect("Failed to process literate file");

    match language {
        Language::Rust => {
            // These files can now be included in the code.
        }
        Language::Python => {
            // It'd be nice to do a format check here at least.
            run_cmd!{
                ruff check ${output}
            }.expect("Failed python format check");
        }
    }
}

// This will extract code blocks and write them to a new file.
fn extract_file(source: &Path, target: &Path, language: Language) -> Result<usize, LiterateError> {
    // Read the source markdown file
    let input = BufReader::new(File::open(source)?);

    // Create the target directory if it doesn't exist
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    // Create the output file
    let mut output = BufWriter::new(File::create(target)?);

    let comment_prefix = language.comment();
    // Write a warning header comment to put the file into read-only mode in Emacs and vim.
    writeln!(output, r#"{} vim:set ro: -*- buffer-read-only:t -*-
{} This file was automatically generated from {:?}. Please edit that file."#, comment_prefix, comment_prefix, source)?;

    // Extract Rust code blocks using literate
    literate::extract(input, output, language.block())
}

