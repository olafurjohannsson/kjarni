//! Document loading and chunking utilities

use crate::{Chunk, ChunkMetadata, TextSplitter, SplitterConfig};
use anyhow::Result;
use std::fs;
use std::path::Path;

/// Supported file extensions for text loading
pub const TEXT_EXTENSIONS: &[&str] = &[
    // Documents
    "txt", "md", "markdown", "rst", "org",
    // Data
    "json", "yaml", "yml", "toml", "xml", "csv",
    // Web
    "html", "htm", "css",
    // Code
    "rs", "py", "js", "ts", "go", "java", "c", "cpp", "h", "hpp",
    "cs", "rb", "sh", "bash", "zsh", "fish", "ps1",
    "sql", "r", "scala", "kt", "swift", "m", "mm",
    "lua", "pl", "php", "ex", "exs", "clj", "hs",
];

/// Configuration for document loading
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Splitter configuration
    pub splitter: SplitterConfig,
    /// Whether to recurse into subdirectories
    pub recursive: bool,
    /// File extensions to include (empty = use defaults)
    pub extensions: Vec<String>,
    /// Whether to include hidden files
    pub include_hidden: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            splitter: SplitterConfig::default(),
            recursive: true,
            extensions: vec![],
            include_hidden: false,
        }
    }
}

/// Load and chunk documents from files/directories
pub struct DocumentLoader {
    config: LoaderConfig,
    splitter: TextSplitter,
}

impl DocumentLoader {
    pub fn new(config: LoaderConfig) -> Self {
        let splitter = TextSplitter::new(config.splitter.clone());
        Self { config, splitter }
    }

    pub fn with_defaults() -> Self {
        Self::new(LoaderConfig::default())
    }

    /// Load chunks from a single file
    pub fn load_file(&self, path: &Path) -> Result<Vec<Chunk>> {
        let content = fs::read_to_string(path)?;
        eprintln!("Splitting file: {} (Size: {} bytes)", path.display(), content.len());
        let texts = self.splitter.split(&content);
        eprintln!("  -> Generated {} chunks", texts.len()); // If you never see this, the splitter is infinite looping
        let total = texts.len();

        let chunks: Vec<Chunk> = texts
            .into_iter()
            .enumerate()
            .map(|(i, text)| {
                let metadata = ChunkMetadata {
                    source: Some(path.display().to_string()),
                    chunk_index: Some(i),
                    total_chunks: Some(total),
                    ..Default::default()
                };
                Chunk::new(text).with_metadata(metadata)
            })
            .collect();

        Ok(chunks)
    }

    /// Load chunks from a directory
    pub fn load_directory(&self, dir: &Path) -> Result<Vec<Chunk>> {
        let mut all_chunks = Vec::new();

        let walker = if self.config.recursive {
            walkdir::WalkDir::new(dir)
        } else {
            walkdir::WalkDir::new(dir).max_depth(1)
        };

        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();

            // Skip directories
            if !path.is_file() {
                continue;
            }

            // Skip hidden files
            if !self.config.include_hidden {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with('.') {
                        continue;
                    }
                }
            }

            // Check extension
            if !self.is_supported_extension(path) {
                continue;
            }

            // Load file
            match self.load_file(path) {
                Ok(chunks) => all_chunks.extend(chunks),
                Err(e) => {
                    log::warn!("Failed to load {}: {}", path.display(), e);
                }
            }
        }

        Ok(all_chunks)
    }

    /// Load chunks from multiple paths (files or directories)
    pub fn load_paths(&self, paths: &[&Path]) -> Result<Vec<Chunk>> {
        let mut all_chunks = Vec::new();

        for path in paths {
            if path.is_dir() {
                all_chunks.extend(self.load_directory(path)?);
            } else if path.is_file() {
                all_chunks.extend(self.load_file(path)?);
            } else {
                log::warn!("Path not found: {}", path.display());
            }
        }

        Ok(all_chunks)
    }

    /// Check if a file has a supported extension
    pub fn is_supported_extension(&self, path: &Path) -> bool {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());

        match ext {
            Some(ext) => {
                if self.config.extensions.is_empty() {
                    TEXT_EXTENSIONS.contains(&ext.as_str())
                } else {
                    self.config.extensions.iter().any(|e| e == &ext)
                }
            }
            None => false,
        }
    }
}

/// Convenience function to load and chunk from paths
pub fn load_documents(paths: &[&str], config: Option<LoaderConfig>) -> Result<Vec<Chunk>> {
    let loader = DocumentLoader::new(config.unwrap_or_default());
    let path_refs: Vec<&Path> = paths.iter().map(|p| Path::new(*p)).collect();
    loader.load_paths(&path_refs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_loader_default() {
        let loader = DocumentLoader::with_defaults();
        assert!(loader.config.recursive);
    }

    #[test]
    fn test_is_supported_extension() {
        let loader = DocumentLoader::with_defaults();
        
        assert!(loader.is_supported_extension(Path::new("file.txt")));
        assert!(loader.is_supported_extension(Path::new("file.md")));
        assert!(loader.is_supported_extension(Path::new("file.rs")));
        assert!(loader.is_supported_extension(Path::new("file.py")));
        
        assert!(!loader.is_supported_extension(Path::new("file.pdf")));
        assert!(!loader.is_supported_extension(Path::new("file.docx")));
        assert!(!loader.is_supported_extension(Path::new("file.exe")));
        assert!(!loader.is_supported_extension(Path::new("file")));
    }

    #[test]
    fn test_load_file() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        
        let mut file = fs::File::create(&file_path).unwrap();
        writeln!(file, "First paragraph.\n\nSecond paragraph.").unwrap();

        let loader = DocumentLoader::with_defaults();
        let chunks = loader.load_file(&file_path).unwrap();

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].metadata.source, Some(file_path.display().to_string()));
    }

    #[test]
    fn test_load_directory() {
        let dir = TempDir::new().unwrap();
        
        // Create test files
        fs::write(dir.path().join("a.txt"), "Content A").unwrap();
        fs::write(dir.path().join("b.md"), "Content B").unwrap();
        fs::write(dir.path().join("c.pdf"), "PDF content").unwrap(); // Should be skipped

        let loader = DocumentLoader::with_defaults();
        let chunks = loader.load_directory(dir.path()).unwrap();

        // Should have chunks from a.txt and b.md, not c.pdf
        let sources: Vec<_> = chunks
            .iter()
            .filter_map(|c| c.metadata.source.as_ref())
            .collect();

        assert!(sources.iter().any(|s| s.contains("a.txt")));
        assert!(sources.iter().any(|s| s.contains("b.md")));
        assert!(!sources.iter().any(|s| s.contains("c.pdf")));
    }

    #[test]
    fn test_skip_hidden_files() {
        let dir = TempDir::new().unwrap();
        
        fs::write(dir.path().join("visible.txt"), "Visible").unwrap();
        fs::write(dir.path().join(".hidden.txt"), "Hidden").unwrap();

        let loader = DocumentLoader::with_defaults();
        let chunks = loader.load_directory(dir.path()).unwrap();

        let sources: Vec<_> = chunks
            .iter()
            .filter_map(|c| c.metadata.source.as_ref())
            .collect();

        assert!(sources.iter().any(|s| s.contains("visible.txt")));
        assert!(!sources.iter().any(|s| s.contains(".hidden.txt")));
    }

    #[test]
    fn test_custom_extensions() {
        let dir = TempDir::new().unwrap();
        
        fs::write(dir.path().join("a.txt"), "Text").unwrap();
        fs::write(dir.path().join("b.custom"), "Custom").unwrap();

        let config = LoaderConfig {
            extensions: vec!["custom".to_string()],
            ..Default::default()
        };

        let loader = DocumentLoader::new(config);
        let chunks = loader.load_directory(dir.path()).unwrap();

        let sources: Vec<_> = chunks
            .iter()
            .filter_map(|c| c.metadata.source.as_ref())
            .collect();

        assert!(!sources.iter().any(|s| s.contains("a.txt")));
        assert!(sources.iter().any(|s| s.contains("b.custom")));
    }
}