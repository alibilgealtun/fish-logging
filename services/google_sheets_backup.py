from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, List

import json
import re

# Optional logger: prefer loguru if available, else fallback to stdlib logging
try:  # pragma: no cover - environment dependent
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("google_sheets_backup")


@dataclass
class GoogleSheetsConfig:
    credentials_path: Path
    spreadsheet_id: str
    worksheet_name: str = "Logs"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GoogleSheetsConfig":
        return GoogleSheetsConfig(
            credentials_path=Path(data["credentials_path"]),
            spreadsheet_id=data["spreadsheet_id"],
            worksheet_name=data.get("worksheet_name", "Logs"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "credentials_path": str(self.credentials_path),
            "spreadsheet_id": self.spreadsheet_id,
            "worksheet_name": self.worksheet_name,
        }


class GoogleSheetsBackup:
    """Service to back up haul logs Excel file to Google Sheets.

    Responsibilities:
    - Persist/load minimal configuration (credentials JSON path, spreadsheet id, worksheet name)
    - Authenticate via service account
    - Push the entire table (with header) to the worksheet, replacing existing content

    Usage:
        svc = GoogleSheetsBackup()
        cfg = svc.load_config()  # may be None
        svc.save_config(GoogleSheetsConfig(Path("cred.json"), "sheet_id", "Logs"))
        svc.backup_excel_to_sheet(Path("logs/hauls/logs.xlsx"))
    """

    def __init__(self, config_path: Optional[Path] = None, app_config=None) -> None:
        if config_path:
            self.config_path = Path(config_path)
        elif app_config and hasattr(app_config, 'google_sheets_config'):
            # Use centralized config if available
            self.config_path = Path("config/google_sheets.json")  # Keep same path for backward compatibility
        else:
            self.config_path = Path("config/google_sheets.json")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    # Configuration management
    def load_config(self) -> Optional[GoogleSheetsConfig]:
        if not self.config_path.exists():
            return None
        try:
            data = json.loads(self.config_path.read_text())
            return GoogleSheetsConfig.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load Google Sheets config: {e}")
            return None

    def save_config(self, cfg: GoogleSheetsConfig) -> None:
        try:
            self.config_path.write_text(json.dumps(cfg.to_dict(), indent=2))
        except Exception as e:
            raise RuntimeError(f"Failed to save config: {e}")

    # Helpers
    @staticmethod
    def _extract_spreadsheet_id(value: str) -> str:
        """Allow passing either the raw spreadsheet ID or a full Google Sheets URL.
        Extract the ID from URLs of the form https://docs.google.com/spreadsheets/d/<ID>/...
        """
        if not value:
            return value
        if "docs.google.com/spreadsheets" in value:
            m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", value)
            if m:
                return m.group(1)
        return value

    # Auth and client helpers
    def _get_client(self, cfg: GoogleSheetsConfig):
        try:
            import gspread  # type: ignore
            from google.oauth2.service_account import Credentials  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependencies for Google Sheets. Please install gspread and google-auth."
            ) from e

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/drive",
        ]
        if not cfg.credentials_path.exists():
            raise FileNotFoundError(f"Credentials file not found: {cfg.credentials_path}")
        credentials = Credentials.from_service_account_file(str(cfg.credentials_path), scopes=scopes)
        client = gspread.authorize(credentials)
        return client

    def _open_or_create_worksheet(self, client, spreadsheet_id: str, worksheet_name: str):
        import gspread  # type: ignore
        try:
            sh = client.open_by_key(spreadsheet_id)
        except gspread.exceptions.SpreadsheetNotFound as e:  # type: ignore
            raise RuntimeError(
                "Spreadsheet not found or access denied. Ensure the ID is correct, and share the sheet with "
                "the service account email from your JSON credentials."
            ) from e
        try:
            return sh.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:  # type: ignore
            # Create if not exists
            return sh.add_worksheet(title=worksheet_name, rows=1000, cols=20)

    def test_connection(self, cfg: Optional[GoogleSheetsConfig] = None) -> Dict[str, Any]:
        """Validate that credentials work and spreadsheet is accessible.

        Returns dict with:
          - ok: bool
          - spreadsheet_id: normalized id
          - worksheet_exists: bool | None (None if cannot check)
          - detail: str (optional message)
        """
        if cfg is None:
            cfg = self.load_config()
            if cfg is None:
                return {"ok": False, "detail": "Google Sheets is not configured."}
        sheet_id = self._extract_spreadsheet_id(cfg.spreadsheet_id)
        try:
            client = self._get_client(cfg)
            sh = client.open_by_key(sheet_id)
            # Try to fetch worksheet without creating it
            worksheet_exists = False
            try:
                sh.worksheet(cfg.worksheet_name)
                worksheet_exists = True
            except Exception:
                worksheet_exists = False
            return {
                "ok": True,
                "spreadsheet_id": sheet_id,
                "worksheet_exists": worksheet_exists,
            }
        except Exception as e:
            return {"ok": False, "spreadsheet_id": sheet_id, "detail": str(e)}

    # Public API
    def backup_excel_to_sheet(self, excel_path: Path, cfg: Optional[GoogleSheetsConfig] = None) -> Dict[str, Any]:
        """Read Excel logs and write to Google Sheets worksheet.

        Returns a dict with counts and destination info for UI messages.
        """
        excel_path = Path(excel_path)
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel logs not found at {excel_path}")

        if cfg is None:
            cfg = self.load_config()
            if cfg is None:
                raise RuntimeError("Google Sheets is not configured. Use 'Change API' in Settings to configure it.")

        # Normalize spreadsheet id (accept URL or raw id)
        sheet_id = self._extract_spreadsheet_id(cfg.spreadsheet_id)
        if not sheet_id:
            raise RuntimeError("Spreadsheet ID is empty. Please configure it in Settings -> Change API…")

        # Load excel to list of lists
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise RuntimeError("pandas is required to read Excel logs.") from e

        try:
            df = pd.read_excel(excel_path, sheet_name=0, engine="openpyxl")
        except Exception as e:
            raise RuntimeError(f"Failed to read Excel: {e}") from e

        # Prepare data with header row and JSON-serializable, Google-friendly values
        header: List[str] = list(df.columns.astype(str))

        # Sanitizer keeps numbers and booleans, converts NaN/NaT to empty string, and formats dates/times
        def to_google_value(v: Any) -> Any:
            import math
            import datetime as dt
            # Try pandas NA/NaT detection if available
            try:
                import pandas as _pd  # type: ignore
                if _pd.isna(v):
                    return ""
            except Exception:
                pass
            # None -> empty
            if v is None:
                return ""
            # Builtin numeric/bool types
            if isinstance(v, bool):
                return v
            if isinstance(v, int):
                return v
            if isinstance(v, float):
                return "" if math.isnan(v) else v
            # Numpy scalar types -> cast to builtin
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
            # Datetime-like
            if isinstance(v, dt.datetime):
                # Use space separator for USER_ENTERED parsing
                return v.isoformat(sep=" ")
            if isinstance(v, dt.date):
                return v.isoformat()
            if isinstance(v, dt.time):
                return v.strftime("%H:%M:%S")
            # Fallback to string
            return str(v)

        values: List[List[Any]] = [header]
        for _, row in df.iterrows():
            values.append([to_google_value(row[col]) for col in df.columns])

        # Push to Google Sheets
        try:
            client = self._get_client(cfg)
            ws = self._open_or_create_worksheet(client, sheet_id, cfg.worksheet_name)
            ws.clear()
            # Use USER_ENTERED so Sheets can parse dates/times from strings
            ws.update("A1", values, value_input_option="USER_ENTERED")
            logger.info(
                "Backed up %s rows (including header) to spreadsheet %s / %s",
                len(values), sheet_id, cfg.worksheet_name,
            )
            return {
                "rows": len(values) - 1,
                "columns": len(header),
                "spreadsheet_id": sheet_id,
                "worksheet_name": cfg.worksheet_name,
            }
        except Exception as e:
            # Try to provide richer diagnostics for common gspread API errors
            try:
                import gspread  # type: ignore
                if isinstance(e, gspread.exceptions.APIError):  # type: ignore
                    resp = getattr(e, "response", None)
                    status = getattr(resp, "status_code", None)
                    text = getattr(resp, "text", None)
                    msg = f"Google API error (status={status}): {text or e}"
                    raise RuntimeError(msg) from e
            except Exception:
                pass
            raise RuntimeError(f"Failed to update Google Sheets: {e}") from e
