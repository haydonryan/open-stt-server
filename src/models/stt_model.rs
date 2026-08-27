use std::str::FromStr;

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum STTModel {
    #[default]
    WhisperTiny,
    WhisperTinyEn,
    WhisperBase,
    WhisperBaseEn,
    WhisperSmall,
    WhisperSmallEn,
    WhisperMedium,
    WhisperMediumEn,
    WhisperLarge,
    WhisperLargeV2,
    WhisperLargeV3,
    WhisperLargeV3Turbo,
    WhisperDistilMediumEn,
    WhisperDistilLargeV2,
    WhisperDistilLargeV3,
    VoxtralMini,
    VoxtralSmall,
}

impl std::fmt::Display for STTModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl STTModel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WhisperTiny => "whisper-tiny",
            Self::WhisperTinyEn => "whisper-tiny.en",
            Self::WhisperBase => "whisper-base",
            Self::WhisperBaseEn => "whisper-base.en",
            Self::WhisperSmall => "whisper-small",
            Self::WhisperSmallEn => "whisper-small.en",
            Self::WhisperMedium => "whisper-medium",
            Self::WhisperMediumEn => "whisper-medium.en",
            Self::WhisperLarge => "whisper-large",
            Self::WhisperLargeV2 => "whisper-large-v2",
            Self::WhisperLargeV3 => "whisper-large-v3",
            Self::WhisperLargeV3Turbo => "whisper-large-v3-turbo",
            Self::WhisperDistilMediumEn => "whisper-distil-medium.en",
            Self::WhisperDistilLargeV2 => "whisper-distil-large-v2",
            Self::WhisperDistilLargeV3 => "whisper-distil-large-v3",
            Self::VoxtralMini => "voxtral-mini",
            Self::VoxtralSmall => "voxtral-small",
        }
    }

    pub const fn is_voxtral(self) -> bool {
        matches!(self, Self::VoxtralMini | Self::VoxtralSmall)
    }

    pub const fn model_and_revision(self) -> (&'static str, &'static str) {
        match self {
            Self::WhisperTiny => ("openai/whisper-tiny", "main"),
            Self::WhisperTinyEn => ("openai/whisper-tiny.en", "main"),
            Self::WhisperBase => ("openai/whisper-base", "main"),
            Self::WhisperBaseEn => ("openai/whisper-base.en", "main"),
            Self::WhisperSmall => ("openai/whisper-small", "main"),
            Self::WhisperSmallEn => ("openai/whisper-small.en", "main"),
            Self::WhisperMedium => ("openai/whisper-medium", "main"),
            Self::WhisperMediumEn => ("openai/whisper-medium.en", "main"),
            Self::WhisperLarge => ("openai/whisper-large", "main"),
            Self::WhisperLargeV2 => ("openai/whisper-large-v2", "main"),
            Self::WhisperLargeV3 => ("openai/whisper-large-v3", "main"),
            Self::WhisperLargeV3Turbo => ("openai/whisper-large-v3-turbo", "main"),
            Self::WhisperDistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::WhisperDistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
            Self::WhisperDistilLargeV3 => ("distil-whisper/distil-large-v3", "main"),
            Self::VoxtralMini => ("mistralai/Voxtral-Mini-3B-2507", "main"),
            Self::VoxtralSmall => ("mistralai/Voxtral-Small-24B-2507", "main"),
        }
    }
}

impl FromStr for STTModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Support OpenAI model aliases
        match s {
            "whisper-1" | "whisper-tiny" => Ok(Self::WhisperTiny),
            "whisper-tiny.en" => Ok(Self::WhisperTinyEn),
            "whisper-base" => Ok(Self::WhisperBase),
            "whisper-base.en" => Ok(Self::WhisperBaseEn),
            "whisper-small" => Ok(Self::WhisperSmall),
            "whisper-small.en" => Ok(Self::WhisperSmallEn),
            "whisper-medium" => Ok(Self::WhisperMedium),
            "whisper-medium.en" => Ok(Self::WhisperMediumEn),
            "whisper-large" => Ok(Self::WhisperLarge),
            "whisper-large-v2" => Ok(Self::WhisperLargeV2),
            "whisper-large-v3" => Ok(Self::WhisperLargeV3),
            "whisper-large-v3-turbo" => Ok(Self::WhisperLargeV3Turbo),
            "whisper-distil-medium.en" => Ok(Self::WhisperDistilMediumEn),
            "whisper-distil-large-v2" => Ok(Self::WhisperDistilLargeV2),
            "whisper-distil-large-v3" => Ok(Self::WhisperDistilLargeV3),
            "voxtral-mini" => Ok(Self::VoxtralMini),
            "voxtral-small" => Ok(Self::VoxtralSmall),
            _ => Err(format!("Unknown model: {s}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_alias() {
        let parsed = "whisper-1".parse::<STTModel>().expect("alias should parse");
        assert_eq!(parsed, STTModel::WhisperTiny);
    }

    #[test]
    fn rejects_unknown_model() {
        let err = "whisper-ultra".parse::<STTModel>().unwrap_err();
        assert!(err.contains("Unknown model"));
    }

    #[test]
    fn voxtral_detection() {
        assert!(STTModel::VoxtralMini.is_voxtral());
        assert!(STTModel::VoxtralSmall.is_voxtral());
        assert!(!STTModel::WhisperBase.is_voxtral());
    }

    #[test]
    fn model_and_revision_matches_expected() {
        let (id, rev) = STTModel::WhisperLargeV3.model_and_revision();
        assert_eq!(id, "openai/whisper-large-v3");
        assert_eq!(rev, "main");
    }
}
