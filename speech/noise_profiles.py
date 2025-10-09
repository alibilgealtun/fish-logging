"""Noise profile presets for different acoustic environments.

Profiles are intentionally conservative; they tune VAD aggressiveness, segment
constraints, and suppressor parameters (via SuppressorConfig) without changing
public recognizer APIs. A recognizer can request a profile name and apply the
returned overrides when constructing its NoiseController.

Profile keys (case-insensitive):
  clean  : Mostly quiet environment, minimal suppression / latency.
  human  : Background conversations; moderate suppression, slightly longer padding.
  engine : Continuous low-frequency + broadband engine/boat noise; aggressive VAD.
  mixed  : Default balanced setting (both human + engine present variably).

Each profile dict may contain:
  VAD_MODE, MIN_SPEECH_S, MAX_SEGMENT_S, PADDING_MS
  SUPPRESSOR (nested kwargs for SuppressorConfig overrides)

Unspecified fields fall back to recognizer defaults.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
from copy import deepcopy
from noise.suppressor import SuppressorConfig

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


_PROFILES: Dict[str, Dict[str, Any]] = {
    "clean": {
        "VAD_MODE": 1,
        "MIN_SPEECH_S": 0.30,
        "MAX_SEGMENT_S": 4.0,
        "PADDING_MS": 400,
        "SUPPRESSOR": {
            "gain_floor": 0.12,  # gentler suppression for quiet environments
            "enable_loudness_gate": False,  # no gate needed in clean conditions
            "speech_band_boost": 0.12,  # moderate boost sufficient
        },
    },
    "human": {
        "VAD_MODE": 2,  # Moderate VAD for natural conversational speech with pauses
        "MIN_SPEECH_S": 0.35,
        "MAX_SEGMENT_S": 3.5,
        "PADDING_MS": 600,
        "SUPPRESSOR": {
            "gain_floor": 0.07,  # More suppression to reject competing voices
            "noise_update_alpha": 0.985,  # Track shifting background chatter
            "speech_band_boost": 0.15,  # Emphasize speech frequencies
            "enable_loudness_gate": True,  # Gate out distant voices
        },
    },
    "engine": {
        "VAD_MODE": 3,  # Most aggressive VAD to cut through steady mechanical noise
        "MIN_SPEECH_S": 0.40,
        "MAX_SEGMENT_S": 3.0,
        "PADDING_MS": 700,
        "SUPPRESSOR": {
            "gain_floor": 0.04,  # Strong suppression for broadband noise
            "noise_update_alpha": 0.995,  # Very slow update - steady noise means less adaptation needed
            "speech_band_boost": 0.08,  # Moderate boost (engine already filtered by HPF)
            "enable_loudness_gate": True,
            "gate_threshold_db": 10.0,  # Tighter gate for constant noise floor
            "gate_max_att_db": 32.0,  # Stronger attenuation
        },
    },
    "mixed": {  # Default baseline similar to existing hardcoded values
        "VAD_MODE": 2,
        "MIN_SPEECH_S": 0.40,
        "MAX_SEGMENT_S": 3.0,
        "PADDING_MS": 600,
        "SUPPRESSOR": {
            "gain_floor": 0.05,  # Balanced suppression
            "noise_update_alpha": 0.98,  # Standard adaptation rate
            "speech_band_boost": 0.10,  # Mild emphasis
            "enable_loudness_gate": True,
        },
    },
}


class NoiseProfileManager:
    """Manages noise profile presets with validation and safe access.

    This class centralizes profile management, ensuring profiles are validated
    at initialization and providing type-safe, immutable access to profile data.
    """

    VALID_PROFILES = {"clean", "human", "engine", "mixed"}
    DEFAULT_PROFILE = "mixed"

    # Valid ranges for profile parameters
    VALID_VAD_MODES = {0, 1, 2, 3}
    MIN_SPEECH_RANGE = (0.1, 2.0)  # seconds
    MAX_SEGMENT_RANGE = (1.0, 10.0)  # seconds
    PADDING_MS_RANGE = (0, 2000)  # milliseconds

    def __init__(self):
        """Initialize the profile manager and validate all profiles."""
        self._profiles = _PROFILES
        self._validate_all_profiles()
        logger.info(f"NoiseProfileManager initialized with profiles: {list(self._profiles.keys())}")

    def _validate_all_profiles(self) -> None:
        """Validate all profiles at initialization to catch configuration errors early."""
        for name, profile in self._profiles.items():
            try:
                self._validate_profile(name, profile)
            except ValueError as e:
                logger.error(f"Profile '{name}' validation failed: {e}")
                raise ValueError(f"Invalid profile configuration '{name}': {e}") from e
        logger.debug("All noise profiles validated successfully")

    def _validate_profile(self, name: str, profile: Dict[str, Any]) -> None:
        """Validate a single profile's parameters.

        Raises:
            ValueError: If profile contains invalid parameters
        """
        # Validate VAD_MODE
        vad_mode = profile.get("VAD_MODE")
        if vad_mode is not None and vad_mode not in self.VALID_VAD_MODES:
            raise ValueError(f"VAD_MODE={vad_mode} not in valid range {self.VALID_VAD_MODES}")

        # Validate MIN_SPEECH_S
        min_speech = profile.get("MIN_SPEECH_S")
        if min_speech is not None:
            if not (self.MIN_SPEECH_RANGE[0] <= min_speech <= self.MIN_SPEECH_RANGE[1]):
                raise ValueError(f"MIN_SPEECH_S={min_speech} outside valid range {self.MIN_SPEECH_RANGE}")

        # Validate MAX_SEGMENT_S
        max_segment = profile.get("MAX_SEGMENT_S")
        if max_segment is not None:
            if not (self.MAX_SEGMENT_RANGE[0] <= max_segment <= self.MAX_SEGMENT_RANGE[1]):
                raise ValueError(f"MAX_SEGMENT_S={max_segment} outside valid range {self.MAX_SEGMENT_RANGE}")

        # Validate PADDING_MS
        padding = profile.get("PADDING_MS")
        if padding is not None:
            if not (self.PADDING_MS_RANGE[0] <= padding <= self.PADDING_MS_RANGE[1]):
                raise ValueError(f"PADDING_MS={padding} outside valid range {self.PADDING_MS_RANGE}")

        # Validate suppressor config if present
        suppressor = profile.get("SUPPRESSOR")
        if suppressor is not None:
            self._validate_suppressor_fields(name, suppressor)

    def _validate_suppressor_fields(self, profile_name: str, suppressor: Dict[str, Any]) -> None:
        """Validate suppressor configuration fields.

        Logs warnings for unknown fields but doesn't raise errors to allow forward compatibility.
        """
        # Get valid fields from SuppressorConfig
        valid_fields = {f.name for f in SuppressorConfig.__dataclass_fields__.values()}

        for field_name in suppressor.keys():
            if field_name not in valid_fields:
                logger.warning(
                    f"Profile '{profile_name}' contains unknown suppressor field '{field_name}'. "
                    f"Valid fields: {sorted(valid_fields)}"
                )

    def get_profile(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Return a deep copy of the profile dict for the given name.

        Args:
            name: Profile name (case-insensitive). Falls back to DEFAULT_PROFILE if None or invalid.

        Returns:
            Deep copy of the profile configuration dictionary.
        """
        if not name:
            name = self.DEFAULT_PROFILE
            logger.debug(f"No profile specified, using default: {self.DEFAULT_PROFILE}")

        key = name.lower().strip()
        if key not in self._profiles:
            logger.warning(
                f"Unknown profile '{name}', falling back to '{self.DEFAULT_PROFILE}'. "
                f"Valid profiles: {sorted(self._profiles.keys())}"
            )
            key = self.DEFAULT_PROFILE

        # Deep copy to prevent any mutation of the original profile data
        profile = deepcopy(self._profiles[key])
        logger.debug(f"Loaded noise profile '{key}': VAD_MODE={profile.get('VAD_MODE')}, "
                    f"MIN_SPEECH_S={profile.get('MIN_SPEECH_S')}, "
                    f"MAX_SEGMENT_S={profile.get('MAX_SEGMENT_S')}")
        return profile

    def make_suppressor_config(self, profile: Dict[str, Any], sample_rate: int) -> SuppressorConfig:
        """Build a SuppressorConfig by overlaying profile SUPPRESSOR fields.

        Only recognized SuppressorConfig fields are applied; unknown fields are logged as warnings.

        Args:
            profile: Profile dictionary (typically from get_profile())
            sample_rate: Audio sample rate in Hz

        Returns:
            SuppressorConfig instance with profile-specific overrides applied
        """
        sup_over = profile.get("SUPPRESSOR", {}) or {}

        # Start from default then override
        cfg = SuppressorConfig(sample_rate=sample_rate)

        applied_fields = []
        for field_name, value in sup_over.items():
            if hasattr(cfg, field_name):
                setattr(cfg, field_name, value)
                applied_fields.append(f"{field_name}={value}")
            else:
                logger.warning(f"Profile contains unknown suppressor field '{field_name}', ignoring")

        if applied_fields:
            logger.debug(f"Applied suppressor overrides: {', '.join(applied_fields)}")

        return cfg

    def list_profiles(self) -> list[str]:
        """Return list of available profile names."""
        return sorted(self._profiles.keys())

    def get_profile_description(self, name: str) -> str:
        """Get a human-readable description of a profile.

        Args:
            name: Profile name

        Returns:
            Description string, or empty string if profile not found
        """
        descriptions = {
            "clean": "Mostly quiet environment, minimal suppression, lowest latency",
            "human": "Background conversations, moderate suppression for competing voices",
            "engine": "Continuous mechanical noise, aggressive VAD and suppression",
            "mixed": "Balanced default for variable human + engine noise",
        }
        return descriptions.get(name.lower().strip(), "")


# Global singleton instance
_manager: Optional[NoiseProfileManager] = None


def get_manager() -> NoiseProfileManager:
    """Get the global NoiseProfileManager singleton instance."""
    global _manager
    if _manager is None:
        _manager = NoiseProfileManager()
    return _manager


def get_noise_profile(name: Optional[str] = None) -> Dict[str, Any]:
    """Return a deep copy of the profile dict for the given name.

    This is a convenience function that delegates to the global NoiseProfileManager.

    Args:
        name: Profile name. Falls back to 'mixed' if None or invalid.

    Returns:
        Deep copy of the profile configuration dictionary.
    """
    return get_manager().get_profile(name)


def make_suppressor_config(profile: Dict[str, Any], sample_rate: int) -> SuppressorConfig:
    """Build a SuppressorConfig by overlaying profile SUPPRESSOR fields.

    This is a convenience function that delegates to the global NoiseProfileManager.

    Args:
        profile: Profile dictionary
        sample_rate: Audio sample rate in Hz

    Returns:
        SuppressorConfig instance with profile overrides
    """
    return get_manager().make_suppressor_config(profile, sample_rate)


__all__ = ["NoiseProfileManager", "get_manager", "get_noise_profile", "make_suppressor_config"]
