import sys
from PyQt6.QtWidgets import QApplication

from config import ConfigManager
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    config_manager = ConfigManager()
    window = MainWindow(config_manager)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
