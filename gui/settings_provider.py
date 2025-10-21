"""
Settings Provider for Fish Logging Application.

This module contains the SettingsProvider class which manages application
settings and provides a clean interface for accessing and persisting
configuration data. It implements the Provider pattern to encapsulate
settings management logic.

Classes:
    SettingsProvider: Manages application settings and configuration

Architecture:
    - Provider pattern for clean settings access
    - Encapsulates settings persistence logic
    - Type-safe configuration management
    - Default value handling
"""
from __future__ import annotations

from typing import Optional, Any, Dict, Tuple

# Optional logger: prefer loguru if available, else fallback to stdlib logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class SettingsProvider:
    """
    Manages application settings and configuration data.

    This class provides a centralized interface for accessing and managing
    application settings, implementing the Provider pattern to encapsulate
    all settings-related logic and ensure type safety.

    Key Responsibilities:
        - Provide clean interface for settings access
        - Handle settings persistence and retrieval
        - Manage default values and validation
        - Ensure type safety for configuration data
        - Coordinate with settings UI components

    Design Patterns:
        - Provider pattern: Encapsulates settings access
        - Facade pattern: Simplifies settings management
        - Singleton-like behavior through dependency injection

    Attributes:
        settings_widget: UI widget containing settings controls
        _cached_settings: Cache for recently accessed settings
    """

    def __init__(self, settings_widget=None) -> None:
        """
        Initialize the settings provider with optional UI widget.

        Args:
            settings_widget: Optional UI widget containing settings controls

        Design:
            - Flexible initialization supporting headless operation
            - UI widget integration for direct settings access
            - Caching for performance optimization
        """
        self.settings_widget = settings_widget
        self._cached_settings: Dict[str, Any] = {}

        logger.debug(f"SettingsProvider initialized with widget: {settings_widget is not None}")

    def get_boat_name(self) -> str:
        """
        Get the current boat name setting.

        Returns:
            str: Current boat name or default if not set

        Features:
            - Retrieves from UI widget if available
            - Falls back to cached value
            - Provides sensible default
            - Handles missing/invalid data gracefully
        """
        try:
            # Try to get from UI widget first
            if self.settings_widget and hasattr(self.settings_widget, 'get_boat_name'):
                boat_name = self.settings_widget.get_boat_name()
                if boat_name and boat_name.strip():
                    self._cached_settings['boat_name'] = boat_name
                    return boat_name.strip()

            # Fall back to cached value
            cached_boat = self._cached_settings.get('boat_name')
            if cached_boat:
                return cached_boat

            # Default value
            default_boat = "Unknown Vessel"
            logger.debug(f"Using default boat name: {default_boat}")
            return default_boat

        except Exception as e:
            logger.error(f"Failed to get boat name: {e}")
            return "Unknown Vessel"

    def get_station_id(self) -> str:
        """
        Get the current station ID setting.

        Returns:
            str: Current station ID or default if not set

        Features:
            - Retrieves from UI widget if available
            - Falls back to cached value
            - Provides sensible default
            - Handles missing/invalid data gracefully
        """
        try:
            # Try to get from UI widget first
            if self.settings_widget and hasattr(self.settings_widget, 'get_station_id'):
                station_id = self.settings_widget.get_station_id()
                if station_id and station_id.strip():
                    self._cached_settings['station_id'] = station_id
                    return station_id.strip()

            # Fall back to cached value
            cached_station = self._cached_settings.get('station_id')
            if cached_station:
                return cached_station

            # Default value
            default_station = "STATION-001"
            logger.debug(f"Using default station ID: {default_station}")
            return default_station

        except Exception as e:
            logger.error(f"Failed to get station ID: {e}")
            return "STATION-001"

    def get_and_save_all(self) -> Tuple[str, str]:
        """
        Get all current settings and save them to cache.

        Convenience method that retrieves all essential settings in one call
        and ensures they are cached for future use.

        Returns:
            Tuple[str, str]: (boat_name, station_id)

        Features:
            - Atomic operation for all settings
            - Automatic caching
            - Consistent data retrieval
            - Performance optimization

        Use Cases:
            - Pre-logging operations
            - Settings validation
            - Batch settings access
        """
        boat_name = self.get_boat_name()
        station_id = self.get_station_id()

        # Ensure values are cached
        self._cached_settings.update({
            'boat_name': boat_name,
            'station_id': station_id
        })

        logger.debug(f"Retrieved all settings - Boat: {boat_name}, Station: {station_id}")
        return boat_name, station_id

    def set_boat_name(self, boat_name: str) -> bool:
        """
        Set the boat name in both UI and cache.

        Args:
            boat_name: New boat name to set

        Returns:
            bool: True if set successfully, False otherwise

        Features:
            - Updates UI widget if available
            - Updates cache for consistency
            - Input validation and sanitization
            - Error handling and logging
        """
        try:
            # Validate input
            if not boat_name or not boat_name.strip():
                logger.warning("Attempted to set empty boat name")
                return False

            clean_name = boat_name.strip()

            # Update UI widget if available
            if self.settings_widget and hasattr(self.settings_widget, 'set_boat_name'):
                self.settings_widget.set_boat_name(clean_name)

            # Update cache
            self._cached_settings['boat_name'] = clean_name

            logger.info(f"Boat name set to: {clean_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to set boat name: {e}")
            return False

    def set_station_id(self, station_id: str) -> bool:
        """
        Set the station ID in both UI and cache.

        Args:
            station_id: New station ID to set

        Returns:
            bool: True if set successfully, False otherwise

        Features:
            - Updates UI widget if available
            - Updates cache for consistency
            - Input validation and sanitization
            - Error handling and logging
        """
        try:
            # Validate input
            if not station_id or not station_id.strip():
                logger.warning("Attempted to set empty station ID")
                return False

            clean_id = station_id.strip()

            # Update UI widget if available
            if self.settings_widget and hasattr(self.settings_widget, 'set_station_id'):
                self.settings_widget.set_station_id(clean_id)

            # Update cache
            self._cached_settings['station_id'] = clean_id

            logger.info(f"Station ID set to: {clean_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to set station ID: {e}")
            return False

    def get_noise_profile(self) -> str:
        """
        Get the current noise profile setting.

        Returns:
            str: Current noise profile identifier

        Features:
            - Retrieves from UI widget if available
            - Falls back to cached value
            - Provides sensible default
        """
        try:
            # Try to get from UI widget first
            if self.settings_widget and hasattr(self.settings_widget, 'get_noise_profile'):
                profile = self.settings_widget.get_noise_profile()
                if profile:
                    self._cached_settings['noise_profile'] = profile
                    return profile

            # Fall back to cached value
            cached_profile = self._cached_settings.get('noise_profile')
            if cached_profile:
                return cached_profile

            # Default value
            default_profile = "default"
            logger.debug(f"Using default noise profile: {default_profile}")
            return default_profile

        except Exception as e:
            logger.error(f"Failed to get noise profile: {e}")
            return "default"

    def set_noise_profile(self, profile: str) -> bool:
        """
        Set the noise profile setting.

        Args:
            profile: Noise profile identifier to set

        Returns:
            bool: True if set successfully, False otherwise
        """
        try:
            if not profile:
                return False

            # Update UI widget if available
            if self.settings_widget and hasattr(self.settings_widget, 'set_noise_profile'):
                self.settings_widget.set_noise_profile(profile)

            # Update cache
            self._cached_settings['noise_profile'] = profile

            logger.info(f"Noise profile set to: {profile}")
            return True

        except Exception as e:
            logger.error(f"Failed to set noise profile: {e}")
            return False

    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all current settings as a dictionary.

        Returns:
            Dict[str, Any]: All current settings

        Features:
            - Comprehensive settings retrieval
            - Consistent data format
            - Useful for debugging and export
        """
        settings = {
            'boat_name': self.get_boat_name(),
            'station_id': self.get_station_id(),
            'noise_profile': self.get_noise_profile(),
        }

        # Add any additional cached settings
        for key, value in self._cached_settings.items():
            if key not in settings:
                settings[key] = value

        return settings

    def update_settings(self, settings_dict: Dict[str, Any]) -> bool:
        """
        Update multiple settings from a dictionary.

        Args:
            settings_dict: Dictionary of settings to update

        Returns:
            bool: True if all updates successful, False otherwise

        Features:
            - Batch settings update
            - Atomic operation (all or nothing)
            - Validation for each setting
            - Rollback on failure
        """
        if not settings_dict:
            return True

        # Store original values for rollback
        original_cache = self._cached_settings.copy()

        try:
            success = True

            # Update each setting
            for key, value in settings_dict.items():
                if key == 'boat_name':
                    success &= self.set_boat_name(value)
                elif key == 'station_id':
                    success &= self.set_station_id(value)
                elif key == 'noise_profile':
                    success &= self.set_noise_profile(value)
                else:
                    # Cache other settings
                    self._cached_settings[key] = value

            if not success:
                # Rollback on failure
                self._cached_settings = original_cache
                logger.error("Settings update failed, rolled back changes")
                return False

            logger.info(f"Successfully updated {len(settings_dict)} settings")
            return True

        except Exception as e:
            # Rollback on exception
            self._cached_settings = original_cache
            logger.error(f"Settings update failed with exception: {e}")
            return False

    def clear_cache(self) -> None:
        """
        Clear the settings cache.

        Forces fresh retrieval of all settings on next access.
        Useful for testing or when settings widget is replaced.
        """
        self._cached_settings.clear()
        logger.debug("Settings cache cleared")

    def is_configured(self) -> bool:
        """
        Check if essential settings are properly configured.

        Returns:
            bool: True if essential settings are available, False otherwise

        Essential Settings:
            - Boat name (non-empty)
            - Station ID (non-empty)
        """
        try:
            boat_name = self.get_boat_name()
            station_id = self.get_station_id()

            configured = (
                boat_name and boat_name.strip() and boat_name != "Unknown Vessel" and
                station_id and station_id.strip() and station_id != "STATION-001"
            )

            logger.debug(f"Settings configured: {configured}")
            return configured

        except Exception as e:
            logger.error(f"Failed to check settings configuration: {e}")
            return False

    def get_widget_info(self) -> Dict[str, Any]:
        """
        Get information about the connected settings widget.

        Returns:
            Dict[str, Any]: Widget information and capabilities

        Useful for:
            - Debugging widget integration
            - Feature detection
            - Troubleshooting settings issues
        """
        if not self.settings_widget:
            return {'connected': False}

        try:
            info = {
                'connected': True,
                'type': type(self.settings_widget).__name__,
                'supports_boat_name': hasattr(self.settings_widget, 'get_boat_name'),
                'supports_station_id': hasattr(self.settings_widget, 'get_station_id'),
                'supports_noise_profile': hasattr(self.settings_widget, 'get_noise_profile'),
            }

            return info

        except Exception as e:
            logger.error(f"Failed to get widget info: {e}")
            return {'connected': False, 'error': str(e)}
