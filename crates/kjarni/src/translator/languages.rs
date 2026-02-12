//! Language normalization and validation.

use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Normalize language input to the canonical form used in prompts.
///
/// Accepts:
/// - ISO codes: "en", "de", "fr"
/// - Full names: "English", "German", "French"  
/// - Lowercase names: "english", "german", "french"
///
/// Returns the canonical name (e.g., "English") or None if unknown.
pub fn normalize_language(input: &str) -> Option<&'static str> {
    let lower = input.to_lowercase();
    LANGUAGE_MAP.get(lower.as_str()).copied()
}

/// Get the ISO code for a language.
pub fn language_code(canonical: &str) -> Option<&'static str> {
    LANGUAGE_CODES.get(canonical).copied()
}

/// Check if a language is supported by FLAN-T5.
pub fn is_supported_language(input: &str) -> bool {
    normalize_language(input).is_some()
}

/// List all supported languages.
pub fn supported_languages() -> Vec<&'static str> {
    SUPPORTED_LANGUAGES.to_vec()
}

/// Languages known to work well with FLAN-T5.
pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "English",
    "German",
    "French",
    "Spanish",
    "Italian",
    "Portuguese",
    "Dutch",
    "Russian",
    "Chinese",
    "Japanese",
    "Korean",
    "Arabic",
    "Hindi",
    "Turkish",
    "Polish",
    "Romanian",
];

static LANGUAGE_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();

    // English
    m.insert("en", "English");
    m.insert("eng", "English");
    m.insert("english", "English");

    // German
    m.insert("de", "German");
    m.insert("deu", "German");
    m.insert("ger", "German");
    m.insert("german", "German");
    m.insert("deutsch", "German");

    // French
    m.insert("fr", "French");
    m.insert("fra", "French");
    m.insert("fre", "French");
    m.insert("french", "French");
    m.insert("français", "French");
    m.insert("francais", "French");

    // Spanish
    m.insert("es", "Spanish");
    m.insert("spa", "Spanish");
    m.insert("spanish", "Spanish");
    m.insert("español", "Spanish");
    m.insert("espanol", "Spanish");

    // Italian
    m.insert("it", "Italian");
    m.insert("ita", "Italian");
    m.insert("italian", "Italian");
    m.insert("italiano", "Italian");

    // Portuguese
    m.insert("pt", "Portuguese");
    m.insert("por", "Portuguese");
    m.insert("portuguese", "Portuguese");
    m.insert("português", "Portuguese");
    m.insert("portugues", "Portuguese");

    // Dutch
    m.insert("nl", "Dutch");
    m.insert("nld", "Dutch");
    m.insert("dut", "Dutch");
    m.insert("dutch", "Dutch");
    m.insert("nederlands", "Dutch");

    // Russian
    m.insert("ru", "Russian");
    m.insert("rus", "Russian");
    m.insert("russian", "Russian");
    m.insert("русский", "Russian");

    // Chinese
    m.insert("zh", "Chinese");
    m.insert("zho", "Chinese");
    m.insert("chi", "Chinese");
    m.insert("chinese", "Chinese");
    m.insert("中文", "Chinese");

    // Japanese
    m.insert("ja", "Japanese");
    m.insert("jpn", "Japanese");
    m.insert("japanese", "Japanese");
    m.insert("日本語", "Japanese");

    // Korean
    m.insert("ko", "Korean");
    m.insert("kor", "Korean");
    m.insert("korean", "Korean");
    m.insert("한국어", "Korean");

    // Arabic
    m.insert("ar", "Arabic");
    m.insert("ara", "Arabic");
    m.insert("arabic", "Arabic");
    m.insert("العربية", "Arabic");

    // Hindi
    m.insert("hi", "Hindi");
    m.insert("hin", "Hindi");
    m.insert("hindi", "Hindi");
    m.insert("हिन्दी", "Hindi");

    // Turkish
    m.insert("tr", "Turkish");
    m.insert("tur", "Turkish");
    m.insert("turkish", "Turkish");
    m.insert("türkçe", "Turkish");

    // Polish
    m.insert("pl", "Polish");
    m.insert("pol", "Polish");
    m.insert("polish", "Polish");
    m.insert("polski", "Polish");

    // Romanian
    m.insert("ro", "Romanian");
    m.insert("ron", "Romanian");
    m.insert("rum", "Romanian");
    m.insert("romanian", "Romanian");
    m.insert("română", "Romanian");

    m
});

static LANGUAGE_CODES: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("English", "en");
    m.insert("German", "de");
    m.insert("French", "fr");
    m.insert("Spanish", "es");
    m.insert("Italian", "it");
    m.insert("Portuguese", "pt");
    m.insert("Dutch", "nl");
    m.insert("Russian", "ru");
    m.insert("Chinese", "zh");
    m.insert("Japanese", "ja");
    m.insert("Korean", "ko");
    m.insert("Arabic", "ar");
    m.insert("Hindi", "hi");
    m.insert("Turkish", "tr");
    m.insert("Polish", "pl");
    m.insert("Romanian", "ro");
    m
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_iso_codes() {
        assert_eq!(normalize_language("en"), Some("English"));
        assert_eq!(normalize_language("de"), Some("German"));
        assert_eq!(normalize_language("fr"), Some("French"));
    }

    #[test]
        
    fn test_normalize_full_names() {
        // Implementation lowercases first, so all case variants work
        assert_eq!(normalize_language("English"), Some("English"));
        assert_eq!(normalize_language("english"), Some("English"));
        assert_eq!(normalize_language("ENGLISH"), Some("English"));
        assert_eq!(normalize_language("German"), Some("German"));
        assert_eq!(normalize_language("german"), Some("German"));
    }

    #[test]
    fn test_normalize_native_names() {
        assert_eq!(normalize_language("deutsch"), Some("German"));
        assert_eq!(normalize_language("français"), Some("French"));
        assert_eq!(normalize_language("español"), Some("Spanish"));
    }

    #[test]
    fn test_unknown_language() {
        assert_eq!(normalize_language("klingon"), None);
        assert_eq!(normalize_language("xyz"), None);
    }

    #[test]
    fn test_language_code() {
        assert_eq!(language_code("English"), Some("en"));
        assert_eq!(language_code("German"), Some("de"));
    }
}
