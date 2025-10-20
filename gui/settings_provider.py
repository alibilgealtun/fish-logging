"""Settings provider to abstract settings access from MainWindow."""
from __future__ import annotations

from typing import Protocol, Optional


class BoatInputWidget(Protocol):
    """Protocol for boat input widget."""

    def get_boat_name(self) -> str:
        """Get the boat name."""
        ...

    def save_boat_name(self) -> None:
        """Save the boat name."""
        ...


class StationInputWidget(Protocol):
    """Protocol for station input widget."""

    def get_station_id(self) -> str:
        """Get the station ID."""
        ...

    def save_station_id(self) -> None:
        """Save the station ID."""
        ...


class SettingsProvider:
    """Provides clean access to settings without violating Law of Demeter.

    This class encapsulates settings widget access, preventing MainWindow
    from reaching deep into widget hierarchies.
    """

    def __init__(self, settings_widget):
        """Initialize the settings provider.

        Args:
            settings_widget: The settings widget containing boat and station inputs
        """
        self._settings_widget = settings_widget

    def get_boat_name(self) -> str:
        """Get the current boat name.

        Returns:
            Boat name or empty string
        """
        boat_widget = getattr(self._settings_widget, 'boat_input', None)
        if boat_widget:
            return boat_widget.get_boat_name()
        return ""

    def get_station_id(self) -> str:
        """Get the current station ID.

        Returns:
            Station ID or empty string
        """
        station_widget = getattr(self._settings_widget, 'station_input', None)
        if station_widget:
            return station_widget.get_station_id()
        return ""

    def save_boat_name(self) -> None:
        """Save the current boat name."""
        boat_widget = getattr(self._settings_widget, 'boat_input', None)
        if boat_widget:
            boat_widget.save_boat_name()

    def save_station_id(self) -> None:
        """Save the current station ID."""
        station_widget = getattr(self._settings_widget, 'station_input', None)
        if station_widget:
            station_widget.save_station_id()

    def get_and_save_all(self) -> tuple[str, str]:
        """Get and save both boat name and station ID.

        Returns:
            Tuple of (boat_name, station_id)
        """
        boat_name = self.get_boat_name()
        station_id = self.get_station_id()
        self.save_boat_name()
        self.save_station_id()
        return boat_name, station_id

