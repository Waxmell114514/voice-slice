from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from humanslice.app.main_window import MainWindow
from humanslice.services.settings_service import load_settings


def main() -> int:
    """Run the CLI when arguments are given, otherwise start the desktop application."""
    if len(sys.argv) > 1:
        from humanslice.cli import main as cli_main

        return cli_main(sys.argv[1:])
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
