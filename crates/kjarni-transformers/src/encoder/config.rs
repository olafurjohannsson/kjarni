//! Encoder configuration types
use std::fmt;

/// Pooling strategies for sequence outputs
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum PoolingStrategy {
    #[default]
    Mean,
    Max,
    Cls,
    LastToken,
}

impl fmt::Display for PoolingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PoolingStrategy::Mean => write!(f, "Mean"),
            PoolingStrategy::Max => write!(f, "Max"),
            PoolingStrategy::Cls => write!(f, "CLS"),
            PoolingStrategy::LastToken => write!(f, "LastToken"),
        }
    }
}

/// Configuration for encoding/embedding operations
#[derive(Clone, Debug)]
pub struct EncodingConfig {
    pub pooling_strategy: PoolingStrategy,
    pub normalize: bool,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            pooling_strategy: PoolingStrategy::Mean,
            normalize: true,
        }
    }
}

impl fmt::Display for EncodingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EncodingConfig {{ pooling: {}, normalize: {} }}",
            self.pooling_strategy, self.normalize
        )
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pooling_strategy_display() {
        assert_eq!(PoolingStrategy::Mean.to_string(), "Mean");
        assert_eq!(PoolingStrategy::Max.to_string(), "Max");
        assert_eq!(PoolingStrategy::Cls.to_string(), "CLS");
        assert_eq!(PoolingStrategy::LastToken.to_string(), "LastToken");
    }

    #[test]
    fn test_pooling_strategy_default() {
        let default: PoolingStrategy = Default::default();
        assert_eq!(default, PoolingStrategy::Mean);
    }

    #[test]
    fn test_encoding_config_defaults() {
        let config = EncodingConfig::default();
        assert_eq!(config.pooling_strategy, PoolingStrategy::Mean);
        assert!(config.normalize);
        assert_eq!(
            config.to_string(),
            "EncodingConfig { pooling: Mean, normalize: true }"
        );
    }

    #[test]
    fn test_encoding_config_custom() {
        let config = EncodingConfig {
            pooling_strategy: PoolingStrategy::Cls,
            normalize: false,
        };
        assert_eq!(config.pooling_strategy, PoolingStrategy::Cls);
        assert!(!config.normalize);
        assert_eq!(
            config.to_string(),
            "EncodingConfig { pooling: CLS, normalize: false }"
        );
    }
}
