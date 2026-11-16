// The command futures nest deeply enough, through the decoder loading chain
// and the server's routers, to pass the default limit when the compiler checks
// that they are Send and Sync.
#![recursion_limit = "256"]

mod commands;

use anyhow::Result;
use clap::Parser;

use kjarni_cli::{Cli, Commands, verbosity_to_log_level};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // CLI verbosity flag takes precedence, otherwise use RUST_LOG env var
    if cli.verbose > 0 {
        let log_level = verbosity_to_log_level(cli.verbose);
        unsafe {
            std::env::set_var("RUST_LOG", log_level);
        }
    }

    env_logger::init();

    let mut command = cli.command;
    commands::hub::resolve_models(&mut command).await?;

    match command {
        Commands::Model { action } => commands::model::run(action).await,

        Commands::Inspect { path } => commands::inspect::run(path.as_ref()).await,

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
            gguf,
            draft,
            draft_tokens,
            prompt_lookup,
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
                gguf,
                draft.as_deref(),
                draft_tokens,
                prompt_lookup,
            )
            .await
        }

        Commands::Embed {
            input,
            model,
            model_path,
            format,
            no_normalize,
            normalize: _,
            pooling,
            gpu,
            quiet,
        } => {
            commands::embed::run(
                input.as_deref(),
                &model,
                model_path.as_deref(),
                &format,
                !no_normalize,
                &pooling,
                gpu,
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
            greedy,
            no_stream,
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
                greedy,
                no_stream,
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
            max_length,
            num_beams,
            length_penalty,
            no_repeat_ngram,
            greedy,
            no_stream,
            gpu,
            quiet,
        } => {
            commands::translate::run(
                input.as_deref(),
                model.as_deref(),
                model_path.as_deref(),
                src.as_deref(),
                dst.as_deref(),
                max_length,
                num_beams,
                length_penalty,
                no_repeat_ngram,
                greedy,
                no_stream,
                gpu,
                quiet,
            )
            .await
        }

        Commands::Mcp => commands::mcp::run().await,
        Commands::Serve {
            host,
            port,
            chat,
            embed,
            classify,
            zero_shot,
            rerank,
            transcribe,
            translate,
            summarize,
            ner,
            preload,
            api_key,
            cors,
            prefix_cache,
            gpu,
            quiet,
        } => {
            use commands::serve::{Defaults, Endpoint, ServeOptions};
            let named = [
                (Endpoint::Chat, &chat),
                (Endpoint::Embed, &embed),
                (Endpoint::Classify, &classify),
                (Endpoint::Classify, &zero_shot),
                (Endpoint::Rerank, &rerank),
                (Endpoint::Transcribe, &transcribe),
                (Endpoint::Translate, &translate),
                (Endpoint::Summarize, &summarize),
                (Endpoint::Ner, &ner),
            ];
            let preload = if preload {
                named
                    .iter()
                    .filter_map(|(endpoint, name)| name.as_ref().map(|n| (*endpoint, n.clone())))
                    .collect()
            } else {
                Vec::new()
            };
            let defaults = Defaults::default();
            let defaults = Defaults {
                chat: chat.unwrap_or(defaults.chat),
                embed: embed.unwrap_or(defaults.embed),
                classify: classify.unwrap_or(defaults.classify),
                zero_shot: zero_shot.unwrap_or(defaults.zero_shot),
                rerank: rerank.unwrap_or(defaults.rerank),
                transcribe: transcribe.unwrap_or(defaults.transcribe),
                translate: translate.or(defaults.translate),
                summarize: summarize.unwrap_or(defaults.summarize),
                ner: ner.unwrap_or(defaults.ner),
            };
            let api_key = api_key
                .or_else(|| std::env::var("KJARNI_API_KEY").ok())
                .filter(|k| !k.is_empty());
            commands::serve::run(ServeOptions {
                host,
                port,
                defaults,
                preload,
                api_key,
                cors,
                gpu,
                prefix_cache,
                quiet,
            })
            .await
        }
        Commands::Transcribe {
            file,
            model,
            model_path,
            language,
            translate,
            timestamps,
            max_tokens,
            no_stream,
            gpu,
            quiet,
        } => {
            commands::transcribe::run(
                &file,
                &model,
                model_path.as_deref(),
                language.as_deref(),
                translate,
                timestamps,
                max_tokens,
                no_stream,
                gpu,
                quiet,
            )
            .await
        }

        Commands::Ner {
            input,
            model,
            model_path,
            aggregation,
            format,
            gpu,
            quiet,
        } => {
            commands::ner::run(
                &input,
                &model,
                model_path.as_deref(),
                &aggregation,
                &format,
                gpu,
                quiet,
            )
            .await
        }

        Commands::Classify {
            input,
            model,
            model_path,
            labels,
            hypothesis,
            top_k,
            threshold,
            max_length,
            batch_size,
            multi_label,
            format,
            gpu,
            dtype,
            quiet,
        } => {
            commands::classify::run(
                &input,
                &model,
                model_path.as_deref(),
                labels.as_deref(),
                hypothesis.as_deref(),
                top_k,
                threshold,
                max_length,
                batch_size,
                multi_label,
                &format,
                gpu,
                dtype.as_deref(),
                quiet,
            )
            .await
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

        Commands::Agent {
            model,
            model_path,
            full_precision,
            dir,
            yes,
            continue_last,
            resume,
            sessions,
            no_memory,
            prompt,
            temperature,
            max_tokens,
            max_steps,
            quiet,
        } => {
            commands::agent::run(commands::agent::AgentArgs {
                model,
                model_path,
                full_precision,
                dir,
                yes,
                continue_last,
                resume,
                no_memory,
                prompt,
                temperature,
                max_tokens,
                max_steps,
                quiet,
                list_sessions: sessions,
            })
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
            gguf,
            draft,
            draft_tokens,
            think,
        } => {
            commands::chat::run(
                &model,
                model_path.as_deref(),
                system.as_deref(),
                temperature,
                max_tokens,
                gpu,
                quiet,
                gguf,
                draft.as_deref(),
                draft_tokens,
                think,
            )
            .await
        }

        Commands::Index { action } => commands::index::run(action).await,

        #[cfg(feature = "image-io")]
        Commands::Image { action } => match action {
            kjarni_cli::ImageCommands::Index {
                inputs,
                output,
                quiet,
            } => commands::image::index(&inputs, &output, quiet).await,
            kjarni_cli::ImageCommands::Search {
                index_path,
                query,
                top_k,
                quiet,
            } => commands::image::search(&index_path, &query, top_k, quiet).await,
        },

        Commands::Search {
            index_path,
            query,
            top_k,
            mode,
            model,
            rerank_model,
            format,
            gpu,
            quiet,
        } => {
            commands::search::run(
                &index_path,
                &query,
                top_k,
                &mode,
                model.as_deref(),
                rerank_model.as_deref(),
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
