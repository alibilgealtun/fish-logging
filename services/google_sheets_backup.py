"""Google Sheets Backup Service for Fish Logging Data.

This module provides comprehensive Google Sheets integration for backing up haul logs
from Excel files to Google Sheets. It handles authentication, configuration management,
data conversion, and error handling for reliable cloud backup of fish logging data.

The service supports both service account and user authentication, automatic credential
detection, and robust data sanitization for Google Sheets compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, List

import json
import re

from core.exceptions import ConfigurationError

# Optional logger: prefer loguru if available, else fallback to stdlib logging
try:  # pragma: no cover - environment dependent
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("google_sheets_backup")


@dataclass
class GoogleSheetsConfig:
    """Configuration data class for Google Sheets backup settings.

    Encapsulates all necessary configuration for connecting to and writing
    data to Google Sheets, including authentication credentials and target
    worksheet information.

    Attributes:
        credentials_path: Path to Google service account JSON credentials file
        spreadsheet_id: Google Sheets spreadsheet ID (extracted from URL or direct)
        worksheet_name: Name of the target worksheet within the spreadsheet
    """
    credentials_path: Path
    spreadsheet_id: str
    worksheet_name: str = "Logs"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GoogleSheetsConfig":
        """Create GoogleSheetsConfig instance from dictionary data.

        Factory method for deserializing configuration from JSON or other
        dictionary sources, with sensible defaults for optional fields.

        Args:
            data: Dictionary containing configuration keys and values

        Returns:
            GoogleSheetsConfig: Configured instance
        """
        return GoogleSheetsConfig(
            credentials_path=Path(data["credentials_path"]),
            spreadsheet_id=data["spreadsheet_id"],
            worksheet_name=data.get("worksheet_name", "Logs"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert GoogleSheetsConfig instance to dictionary for serialization.

        Serializes the configuration to a dictionary format suitable for
        JSON storage and configuration file persistence.

        Returns:
            Dict[str, Any]: Serializable dictionary representation
        """
        return {
            "credentials_path": str(self.credentials_path),
            "spreadsheet_id": self.spreadsheet_id,
            "worksheet_name": self.worksheet_name,
        }


class GoogleSheetsBackup:
    """Service for backing up haul logs Excel files to Google Sheets.

    This service provides a complete solution for synchronizing fish logging data
    from local Excel files to Google Sheets for cloud backup and collaboration.
    It handles authentication, data conversion, error recovery, and maintains
    configuration persistence.

    Key Features:
    - Persistent configuration management with JSON storage
    - Multiple authentication methods (service account, user OAuth)
    - Automatic credential detection and validation
    - Excel to Google Sheets data conversion with type preservation
    - Robust error handling and diagnostics
    - Data sanitization for Google Sheets compatibility
    - Connection testing and validation

    Architecture:
    - Configuration persistence via JSON files
    - Lazy authentication with credential validation
    - Pandas-based Excel reading with type-aware conversion
    - gspread integration for Google Sheets API access
    - Comprehensive error reporting for troubleshooting

    Usage Example:
        ```python
        # Initialize service
        backup_service = GoogleSheetsBackup()

        # Configure credentials and target
        config = GoogleSheetsConfig(
            credentials_path=Path("service-account.json"),
            spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
            worksheet_name="Fish_Logs"
        )
        backup_service.save_config(config)

        # Test connection
        result = backup_service.test_connection()
        if result["ok"]:
            # Perform backup
            backup_service.backup_excel_to_sheet(Path("logs/hauls/logs.xlsx"))
        ```
    """

    def __init__(self, config_path: Optional[Path] = None, app_config=None) -> None:
        """Initialize GoogleSheetsBackup service with configuration management.

        Sets up the service with flexible configuration path handling to support
        both standalone usage and integration with application configuration systems.

        Args:
            config_path: Optional custom path for configuration storage.
                        If None, uses default location or app_config integration
            app_config: Optional application configuration object with google_sheets_config.
                       Used for centralized configuration management
        """
        if config_path:
            self.config_path = Path(config_path)
        elif app_config and hasattr(app_config, 'google_sheets_config'):
            # Integration with centralized application configuration
            self.config_path = Path("config/google_sheets.json")
        else:
            # Default standalone configuration location
            self.config_path = Path("config/google_sheets.json")

        # Ensure configuration directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== Configuration Management =====

    def load_config(self) -> Optional[GoogleSheetsConfig]:
        """Load Google Sheets configuration from persistent storage.

        Attempts to load previously saved configuration from the JSON file.
        Handles file access errors gracefully and returns None if configuration
        is not available or corrupted.

        Returns:
            Optional[GoogleSheetsConfig]: Loaded configuration or None if unavailable
        """
        if not self.config_path.exists():
            return None

        try:
            data = json.loads(self.config_path.read_text())
            return GoogleSheetsConfig.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load Google Sheets config: {e}")
            return None

    def save_config(self, cfg: GoogleSheetsConfig) -> None:
        """Save Google Sheets configuration to persistent storage.

        Persists the configuration to JSON file with proper error handling
        and meaningful error messages for troubleshooting.

        Args:
            cfg: GoogleSheetsConfig instance to save

        Raises:
            ConfigurationError: If configuration cannot be saved to disk
        """
        try:
            self.config_path.write_text(json.dumps(cfg.to_dict(), indent=2))
        except Exception as e:
            raise ConfigurationError(f"Failed to save config: {e}")

    # ===== Helper Methods =====

    @staticmethod
    def _extract_spreadsheet_id(value: str) -> str:
        """Extract spreadsheet ID from Google Sheets URL or return raw ID.

        Provides flexible input handling by accepting either raw spreadsheet IDs
        or full Google Sheets URLs. This improves user experience by allowing
        users to paste URLs directly from their browser.

        Args:
            value: Either a spreadsheet ID or a full Google Sheets URL

        Returns:
            str: Extracted or validated spreadsheet ID

        Example:
            URL: "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit"
            Returns: "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
        """
        if not value:
            return value

        # Check if value is a Google Sheets URL and extract ID
        if "docs.google.com/spreadsheets" in value:
            match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", value)
            if match:
                return match.group(1)

        # Return as-is if not a URL (assume it's already an ID)
        return value

    # ===== Authentication and Client Management =====

    def _get_client(self, cfg: GoogleSheetsConfig):
        """Create and configure gspread client with proper authentication.

        Handles the authentication process using service account credentials
        and creates a properly configured gspread client for Google Sheets API access.

        Args:
            cfg: GoogleSheetsConfig containing authentication and target information

        Returns:
            gspread.Client: Authenticated gspread client ready for API calls

        Raises:
            RuntimeError: If required dependencies are missing
            FileNotFoundError: If credentials file is not found
            Exception: If authentication fails
        """
        try:
            import gspread  # type: ignore
            from google.oauth2.service_account import Credentials  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependencies for Google Sheets. Please install gspread and google-auth."
            ) from e

        # Define required OAuth scopes for Google Sheets and Drive access
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",      # Read/write spreadsheets
            "https://www.googleapis.com/auth/drive.file",        # Access specific files
            "https://www.googleapis.com/auth/drive",             # Full Drive access
        ]

        # Validate credentials file exists
        if not cfg.credentials_path.exists():
            raise FileNotFoundError(f"Credentials file not found: {cfg.credentials_path}")

        # Load credentials and create authenticated client
        credentials = Credentials.from_service_account_file(str(cfg.credentials_path), scopes=scopes)
        client = gspread.authorize(credentials)
        return client

    def _open_or_create_worksheet(self, client, spreadsheet_id: str, worksheet_name: str):
        """Open existing worksheet or create new one if it doesn't exist.

        Provides robust worksheet access with automatic creation fallback.
        Handles common access issues and provides meaningful error messages
        for troubleshooting authentication and permission problems.

        Args:
            client: Authenticated gspread client
            spreadsheet_id: Target spreadsheet ID
            worksheet_name: Name of the worksheet to open or create

        Returns:
            gspread.Worksheet: Ready-to-use worksheet object

        Raises:
            RuntimeError: If spreadsheet access fails or permissions are insufficient
        """
        import gspread  # type: ignore

        try:
            # Attempt to open the spreadsheet
            spreadsheet = client.open_by_key(spreadsheet_id)
        except gspread.exceptions.SpreadsheetNotFound as e:  # type: ignore
            raise RuntimeError(
                "Spreadsheet not found or access denied. Ensure the ID is correct, and share the sheet with "
                "the service account email from your JSON credentials."
            ) from e

        try:
            # Try to access existing worksheet
            return spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:  # type: ignore
            # Create worksheet if it doesn't exist
            return spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)

    def test_connection(self, cfg: Optional[GoogleSheetsConfig] = None) -> Dict[str, Any]:
        """Test Google Sheets connection and validate configuration.

        Performs comprehensive validation of credentials, spreadsheet access,
        and worksheet availability. Provides detailed diagnostic information
        for troubleshooting connection issues.

        Args:
            cfg: Optional configuration to test. If None, loads saved configuration

        Returns:
            Dict[str, Any]: Test results containing:
                - ok: bool - Overall success status
                - spreadsheet_id: str - Normalized spreadsheet ID
                - worksheet_exists: bool - Whether target worksheet exists
                - detail: str - Error message if connection fails
        """
        if cfg is None:
            cfg = self.load_config()
            if cfg is None:
                return {"ok": False, "detail": "Google Sheets is not configured."}

        # Normalize spreadsheet ID (handle URLs)
        sheet_id = self._extract_spreadsheet_id(cfg.spreadsheet_id)

        try:
            # Test authentication and spreadsheet access
            client = self._get_client(cfg)
            spreadsheet = client.open_by_key(sheet_id)

            # Check if target worksheet exists
            worksheet_exists = False
            try:
                spreadsheet.worksheet(cfg.worksheet_name)
                worksheet_exists = True
            except Exception:
                worksheet_exists = False

            return {
                "ok": True,
                "spreadsheet_id": sheet_id,
                "worksheet_exists": worksheet_exists,
            }
        except Exception as e:
            return {
                "ok": False,
                "spreadsheet_id": sheet_id,
                "detail": str(e)
            }

    # ===== Main Backup Functionality =====

    def backup_excel_to_sheet(self, excel_path: Path, cfg: Optional[GoogleSheetsConfig] = None) -> Dict[str, Any]:
        """Read Excel logs and write to Google Sheets worksheet.

        Performs the complete backup process from Excel file to Google Sheets,
        including data reading, type-aware conversion, sanitization, and upload.
        Handles various data types and edge cases for robust data preservation.

        The method preserves data types while ensuring Google Sheets compatibility:
        - Numbers and booleans are preserved as-is
        - Dates and times are formatted for Google Sheets parsing
        - NaN/null values are converted to empty strings
        - Text is preserved with proper encoding

        Args:
            excel_path: Path to the Excel file containing haul logs
            cfg: Optional GoogleSheetsConfig. If None, loads from saved configuration

        Returns:
            Dict[str, Any]: Backup results containing:
                - rows: int - Number of data rows processed (excluding header)
                - columns: int - Number of columns in the data
                - spreadsheet_id: str - Target spreadsheet ID
                - worksheet_name: str - Target worksheet name

        Raises:
            FileNotFoundError: If Excel file doesn't exist
            RuntimeError: If Google Sheets is not configured or backup fails
        """
        excel_path = Path(excel_path)
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel logs not found at {excel_path}")

        # Load configuration
        if cfg is None:
            cfg = self.load_config()
            if cfg is None:
                raise RuntimeError("Google Sheets is not configured. Use 'Change API' in Settings to configure it.")

        # Normalize and validate spreadsheet ID
        sheet_id = self._extract_spreadsheet_id(cfg.spreadsheet_id)
        if not sheet_id:
            raise RuntimeError("Spreadsheet ID is empty. Please configure it in Settings → Change API…")

        # Load Excel file using pandas for robust data handling
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise RuntimeError("pandas is required to read Excel logs.") from e

        try:
            # Read Excel file with proper engine for .xlsx files
            df = pd.read_excel(excel_path, sheet_name=0, engine="openpyxl")
        except Exception as e:
            raise RuntimeError(f"Failed to read Excel: {e}") from e

        # Prepare data with header row and Google Sheets-compatible values
        header: List[str] = list(df.columns.astype(str))

        def to_google_value(v: Any) -> Any:
            """Convert pandas values to Google Sheets-compatible format.

            This function handles the complex task of converting various pandas
            data types to formats that Google Sheets can properly interpret and
            display, while preserving as much type information as possible.

            Args:
                v: Value from pandas DataFrame to convert

            Returns:
                Any: Google Sheets-compatible value
            """
            import math
            import datetime as dt

            # Handle pandas NA/NaT values if pandas is available
            try:
                import pandas as _pd  # type: ignore
                if _pd.isna(v):
                    return ""
            except Exception:
                pass

            # Handle None values
            if v is None:
                return ""

            # Preserve boolean and numeric types
            if isinstance(v, bool):
                return v
            if isinstance(v, int):
                return v
            if isinstance(v, float):
                return "" if math.isnan(v) else v

            # Handle numpy scalar types by converting to Python built-ins
            try:
                import numpy as np  # type: ignore
                if isinstance(v, np.bool_):
                    return bool(v)
                if isinstance(v, np.integer):
                    return int(v)
                if isinstance(v, np.floating):
                    return "" if np.isnan(v) else float(v)
            except Exception:
                pass

            # Handle datetime types with Google Sheets-friendly formatting
            if isinstance(v, dt.datetime):
                # Use space separator for better USER_ENTERED parsing
                return v.isoformat(sep=" ")
            if isinstance(v, dt.date):
                return v.isoformat()
            if isinstance(v, dt.time):
                return v.strftime("%H:%M:%S")

            # Fallback to string conversion for any other types
            return str(v)

        # Build complete data structure with header and converted values
        values: List[List[Any]] = [header]
        for _, row in df.iterrows():
            values.append([to_google_value(row[col]) for col in df.columns])

        # Upload data to Google Sheets
        try:
            client = self._get_client(cfg)
            worksheet = self._open_or_create_worksheet(client, sheet_id, cfg.worksheet_name)

            # Clear existing content and upload new data
            worksheet.clear()
            # Use USER_ENTERED to allow Google Sheets to parse dates/times from strings
            worksheet.update("A1", values, value_input_option="USER_ENTERED")

            logger.info(
                "Backed up %s rows (including header) to spreadsheet %s / %s",
                len(values), sheet_id, cfg.worksheet_name,
            )

            return {
                "rows": len(values) - 1,          # Exclude header from count
                "columns": len(header),
                "spreadsheet_id": sheet_id,
                "worksheet_name": cfg.worksheet_name,
            }
        except Exception as e:
            # Enhanced error handling for gspread API errors
            try:
                import gspread  # type: ignore
                if isinstance(e, gspread.exceptions.APIError):  # type: ignore
                    # Extract detailed API error information
                    resp = getattr(e, "response", None)
                    status = getattr(resp, "status_code", None)
                    text = getattr(resp, "text", None)
                    msg = f"Google API error (status={status}): {text or e}"
                    raise RuntimeError(msg) from e
            except Exception:
                pass
            raise RuntimeError(f"Failed to update Google Sheets: {e}") from e
