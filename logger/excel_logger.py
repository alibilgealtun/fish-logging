from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional, List


class ExcelLogger:
    def __init__(self, file_path: Optional[str] = None) -> None:
        default_path = Path("logs/hauls/") / "logs.xlsx"
        self.file_path = Path(file_path).absolute() if file_path else default_path.absolute()
        self.lock = Lock()

    def _ensure_workbook(self) -> None:
        from openpyxl import Workbook  # type: ignore

        with self.lock:
            if not self.file_path.exists():
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                wb = Workbook()
                ws = wb.active
                ws.title = "Logs"
                # New header includes Station ID
                ws.append(["Date", "Time",  "Boat", "Station ID", "Species", "Length (cm)", "Confidence"])
                wb.save(self.file_path)

    def _read_header(self, ws) -> List[str]:
        return [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]

    def log_entry(self, species: str, length_cm: float, confidence: float, boat: str = "", station_id: str = "") -> None:
        self._ensure_workbook()
        from openpyxl import load_workbook  # type: ignore

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        with self.lock:
            wb = load_workbook(self.file_path)
            ws = wb.active
            header = self._read_header(ws)
            if "Station ID" in (header or []):
                ws.append([date_str, time_str, boat, station_id, species, float(length_cm), float(confidence)])
            else:
                # Backward compatibility with older files without Station ID column
                ws.append([date_str, time_str, boat, species, float(length_cm), float(confidence)])
            wb.save(self.file_path)

    def cancel_last(self) -> bool:
        self._ensure_workbook()
        from openpyxl import load_workbook  # type: ignore

        with self.lock:
            wb = load_workbook(self.file_path)
            ws = wb.active
            if ws.max_row <= 1:
                return False
            ws.delete_rows(ws.max_row, 1)
            wb.save(self.file_path)
            return True
