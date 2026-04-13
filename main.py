from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from services.settings_service import load_settings


def main() -> int:
    """Start the HumanSlice desktop application."""
    app = QApplication(sys.argv)
    app.setApplicationName("HumanSlice")
    app.setOrganizationName("HumanSlice")

    config_path = Path("config.json")
    settings = load_settings(config_path if config_path.exists() else None)

    window = MainWindow(settings=settings)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
