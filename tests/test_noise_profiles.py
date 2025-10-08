"""Tests for noise profile management and validation."""
from __future__ import annotations

import pytest
from speech.noise_profiles import (
    NoiseProfileManager,
    get_noise_profile,
    make_suppressor_config,
    get_manager,
)
from noise.suppressor import SuppressorConfig


class TestNoiseProfileManager:
    """Test suite for NoiseProfileManager."""
    
    def test_singleton_pattern(self):
        """Verify get_manager returns the same instance."""
        manager1 = get_manager()
        manager2 = get_manager()
        assert manager1 is manager2
    
    def test_all_profiles_valid(self):
        """Ensure all predefined profiles pass validation."""
        manager = NoiseProfileManager()
        # Should not raise any exceptions
        assert manager is not None
    
    def test_list_profiles(self):
        """Verify all expected profiles are available."""
        manager = get_manager()
        profiles = manager.list_profiles()
        assert "clean" in profiles
        assert "human" in profiles
        assert "engine" in profiles
        assert "mixed" in profiles
        assert len(profiles) == 4
    
    def test_get_profile_returns_dict(self):
        """Ensure get_profile returns a dictionary."""
        for name in ["clean", "human", "engine", "mixed"]:
            profile = get_noise_profile(name)
            assert isinstance(profile, dict)
            assert len(profile) > 0
    
    def test_get_profile_deep_copy(self):
        """Verify profiles are deep copied (mutation doesn't affect original)."""
        profile1 = get_noise_profile("mixed")
        profile2 = get_noise_profile("mixed")
        
        # Mutate first copy
        profile1["VAD_MODE"] = 999
        profile1["SUPPRESSOR"]["gain_floor"] = 0.999
        
        # Second copy should be unchanged
        assert profile2["VAD_MODE"] != 999
        assert profile2["SUPPRESSOR"]["gain_floor"] != 0.999
    
    def test_invalid_profile_fallback(self):
        """Unknown profile names should fall back to 'mixed'."""
        profile = get_noise_profile("invalid_profile_name")
        mixed_profile = get_noise_profile("mixed")
        assert profile["VAD_MODE"] == mixed_profile["VAD_MODE"]
    
    def test_none_profile_fallback(self):
        """None profile should fall back to 'mixed'."""
        profile = get_noise_profile(None)
        mixed_profile = get_noise_profile("mixed")
        assert profile["VAD_MODE"] == mixed_profile["VAD_MODE"]
    
    def test_profile_case_insensitive(self):
        """Profile names should be case-insensitive."""
        lower = get_noise_profile("clean")
        upper = get_noise_profile("CLEAN")
        mixed = get_noise_profile("ClEaN")
        assert lower["VAD_MODE"] == upper["VAD_MODE"] == mixed["VAD_MODE"]
    
    def test_all_profiles_have_required_fields(self):
        """Ensure all profiles contain expected top-level fields."""
        required = ["VAD_MODE", "MIN_SPEECH_S", "MAX_SEGMENT_S", "PADDING_MS"]
        for name in ["clean", "human", "engine", "mixed"]:
            profile = get_noise_profile(name)
            for field in required:
                assert field in profile, f"Profile '{name}' missing field '{field}'"
    
    def test_vad_mode_in_valid_range(self):
        """Verify all profiles have VAD_MODE in range [0, 3]."""
        for name in ["clean", "human", "engine", "mixed"]:
            profile = get_noise_profile(name)
            vad = profile["VAD_MODE"]
            assert 0 <= vad <= 3, f"Profile '{name}' has invalid VAD_MODE={vad}"
    
    def test_min_speech_positive(self):
        """Verify all profiles have positive MIN_SPEECH_S."""
        for name in ["clean", "human", "engine", "mixed"]:
            profile = get_noise_profile(name)
            min_s = profile["MIN_SPEECH_S"]
            assert min_s > 0, f"Profile '{name}' has invalid MIN_SPEECH_S={min_s}"
    
    def test_max_segment_greater_than_min(self):
        """Verify MAX_SEGMENT_S > MIN_SPEECH_S for all profiles."""
        for name in ["clean", "human", "engine", "mixed"]:
            profile = get_noise_profile(name)
            assert profile["MAX_SEGMENT_S"] > profile["MIN_SPEECH_S"]
    
    def test_suppressor_config_creation(self):
        """Ensure suppressor config can be created from all profiles."""
        for name in ["clean", "human", "engine", "mixed"]:
            profile = get_noise_profile(name)
            cfg = make_suppressor_config(profile, sample_rate=16000)
            assert isinstance(cfg, SuppressorConfig)
            assert cfg.sample_rate == 16000
    
    def test_suppressor_overrides_applied(self):
        """Verify profile suppressor settings override defaults."""
        profile = get_noise_profile("engine")
        cfg = make_suppressor_config(profile, sample_rate=16000)
        
        # Engine profile should have very low gain_floor
        assert cfg.gain_floor == 0.04
        
        # And high noise_update_alpha
        assert cfg.noise_update_alpha == 0.995
    
    def test_clean_profile_no_gate(self):
        """Clean profile should disable loudness gate."""
        profile = get_noise_profile("clean")
        cfg = make_suppressor_config(profile, sample_rate=16000)
        assert cfg.enable_loudness_gate is False
    
    def test_get_profile_description(self):
        """Ensure profile descriptions are available."""
        manager = get_manager()
        for name in ["clean", "human", "engine", "mixed"]:
            desc = manager.get_profile_description(name)
            assert len(desc) > 0
            assert isinstance(desc, str)


class TestProfileValidation:
    """Test validation logic catches invalid configurations."""
    
    def test_invalid_vad_mode_raises(self):
        """Profile with invalid VAD_MODE should raise ValueError."""
        manager = NoiseProfileManager()
        with pytest.raises(ValueError, match="VAD_MODE"):
            manager._validate_profile("test", {"VAD_MODE": 99})
    
    def test_negative_min_speech_raises(self):
        """Profile with negative MIN_SPEECH_S should raise ValueError."""
        manager = NoiseProfileManager()
        with pytest.raises(ValueError, match="MIN_SPEECH_S"):
            manager._validate_profile("test", {"MIN_SPEECH_S": -0.5})
    
    def test_huge_max_segment_raises(self):
        """Profile with excessive MAX_SEGMENT_S should raise ValueError."""
        manager = NoiseProfileManager()
        with pytest.raises(ValueError, match="MAX_SEGMENT_S"):
            manager._validate_profile("test", {"MAX_SEGMENT_S": 999.0})
    
    def test_negative_padding_raises(self):
        """Profile with negative PADDING_MS should raise ValueError."""
        manager = NoiseProfileManager()
        with pytest.raises(ValueError, match="PADDING_MS"):
            manager._validate_profile("test", {"PADDING_MS": -100})


class TestProfileCharacteristics:
    """Test specific characteristics of each profile."""
    
    def test_clean_profile_least_aggressive(self):
        """Clean profile should have lowest VAD aggressiveness."""
        clean = get_noise_profile("clean")
        assert clean["VAD_MODE"] == 1  # Least aggressive
    
    def test_engine_profile_most_aggressive(self):
        """Engine profile should have highest VAD aggressiveness."""
        engine = get_noise_profile("engine")
        assert engine["VAD_MODE"] == 3  # Most aggressive
    
    def test_human_profile_moderate_vad(self):
        """Human profile should have moderate VAD (was incorrectly 3, now 2)."""
        human = get_noise_profile("human")
        assert human["VAD_MODE"] == 2  # Moderate for natural speech
    
    def test_mixed_profile_balanced(self):
        """Mixed profile should be balanced (VAD=2)."""
        mixed = get_noise_profile("mixed")
        assert mixed["VAD_MODE"] == 2
    
    def test_clean_shortest_min_speech(self):
        """Clean profile should have shortest MIN_SPEECH_S for low latency."""
        clean = get_noise_profile("clean")
        human = get_noise_profile("human")
        engine = get_noise_profile("engine")
        mixed = get_noise_profile("mixed")
        
        assert clean["MIN_SPEECH_S"] <= human["MIN_SPEECH_S"]
        assert clean["MIN_SPEECH_S"] <= engine["MIN_SPEECH_S"]
        assert clean["MIN_SPEECH_S"] <= mixed["MIN_SPEECH_S"]
    
    def test_engine_longest_padding(self):
        """Engine profile should have longest padding for steady noise."""
        engine = get_noise_profile("engine")
        clean = get_noise_profile("clean")
        
        assert engine["PADDING_MS"] > clean["PADDING_MS"]


class TestIntegrationWithRecognizers:
    """Test profile integration with recognizer workflow."""
    
    def test_profile_to_noise_controller_params(self):
        """Simulate extracting params for NoiseController initialization."""
        profile = get_noise_profile("engine")
        
        # These are the params a recognizer would extract
        vad_mode = profile.get("VAD_MODE")
        min_speech_s = profile.get("MIN_SPEECH_S")
        max_segment_s = profile.get("MAX_SEGMENT_S")
        
        assert vad_mode is not None
        assert min_speech_s is not None
        assert max_segment_s is not None
        
        # Verify they're sensible values
        assert 0 <= vad_mode <= 3
        assert 0 < min_speech_s < max_segment_s
    
    def test_full_workflow_all_profiles(self):
        """Simulate full workflow: get profile → extract params → build suppressor."""
        for profile_name in ["clean", "human", "engine", "mixed"]:
            # Step 1: Get profile
            profile = get_noise_profile(profile_name)
            
            # Step 2: Extract params (what recognizer does)
            vad_mode = profile["VAD_MODE"]
            min_speech = profile["MIN_SPEECH_S"]
            max_segment = profile["MAX_SEGMENT_S"]
            padding = profile["PADDING_MS"]
            
            # Step 3: Build suppressor config
            suppressor_cfg = make_suppressor_config(profile, 16000)
            
            # Verify all steps succeeded
            assert isinstance(vad_mode, int)
            assert isinstance(min_speech, (int, float))
            assert isinstance(max_segment, (int, float))
            assert isinstance(padding, int)
            assert isinstance(suppressor_cfg, SuppressorConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

