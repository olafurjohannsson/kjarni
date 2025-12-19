mod commands;

use anyhow::Result;
use clap::{Parser, Subcommand};

use kjarni_cli::{Cli, Commands};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let log_level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    unsafe {
        std::env::set_var("RUST_LOG", log_level);
    }
    env_logger::init();

    match cli.command {
        Commands::Model { action } => commands::model::run(action).await,

        Commands::Generate {
            prompt,
            model,
            model_path,
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            greedy,
            gpu,
            no_stream,
            quiet,
        } => {
            commands::generate::run(
                prompt.as_deref(),
                &model,
                model_path.as_deref(),
                max_tokens,
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                greedy,
                gpu,
                no_stream,
                quiet,
            )
            .await
        }

        Commands::Summarize {
            input,
            model,
            model_path,
            min_length,
            max_length,
            num_beams,
            length_penalty,
            no_repeat_ngram,
            gpu,
            quiet,
        } => {
            commands::summarize::run(
                input.as_deref(),
                &model,
                model_path.as_deref(),
                min_length,
                max_length,
                num_beams,
                length_penalty,
                no_repeat_ngram,
                gpu,
                quiet,
            )
            .await
        }

        Commands::Translate {
            input,
            model,
            model_path,
            src,
            dst,
        } => {
            commands::translate::run(
                input.as_deref(),
                &model,
                model_path.as_deref(),
                src.as_deref(),
                dst.as_deref(),
            )
            .await
        }

        Commands::Transcribe {
            file,
            model,
            model_path,
            language,
        } => {
            commands::transcribe::run(&file, &model, model_path.as_deref(), language.as_deref())
                .await
        }

        Commands::Encode {
            input,
            model,
            model_path,
            format,
            normalize,
            pooling,
            gpu,
        } => {
            commands::encode::run(
                input.as_deref(),
                &model,
                model_path.as_deref(),
                &format,
                normalize,
                &pooling,
                gpu,
            )
            .await
        }

        Commands::Classify {
            input,
            model,
            top_k,
            format,
            labels,
            gpu,
            quiet,
        } => {
            unimplemented!()
            // commands::classify::run(
            //     input.as_deref(),
            //     &model,
            //     model_path.as_deref(),
            //     &format,
            //     gpu,
            // )
            // .await
        }

        Commands::Rerank {
            query,
            documents,
            model,
            model_path,
            top_k,
            format,
            gpu,
            quiet,
        } => {
            commands::rerank::run(
                &query,
                &documents,
                &model,
                model_path.as_deref(),
                top_k,
                &format,
                gpu,
                quiet,
            )
            .await
        }

        Commands::Chat {
            model,
            model_path,
            system,
            temperature,
            max_tokens,
            gpu,
            quiet,
        } => {
            commands::chat::run(
                &model,
                model_path.as_deref(),
                system.as_deref(),
                temperature,
                max_tokens,
                gpu,
                quiet,
            )
            .await
        }

        Commands::Repl {
            model,
            model_path,
            mode,
            gpu,
        } => commands::repl::run(&model, model_path.as_deref(), &mode, gpu).await,

        Commands::Index { action } => commands::index::run(action).await,

        Commands::Search {
            index_path,
            query,
            top_k,
            mode,
            model,
            format,
            gpu,
            quiet,
        } => {
            commands::search::run(
                &index_path,
                &query,
                top_k,
                &mode,
                &model,
                &format,
                gpu,
                quiet,
            )
            .await
        }

        Commands::Similarity {
            text1,
            text2,
            model,
            gpu,
            quiet,
        } => commands::similarity::run(&text1, &text2, &model, gpu, quiet).await,


        
    }
}
