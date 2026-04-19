import os
import sys

from PyQt5.QtWidgets import QApplication

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from frontend.app import TouchlessAssistantApp


def main():
    qt_app = QApplication(sys.argv)
    assistant_gui = TouchlessAssistantApp()
    assistant_gui.show()
    sys.exit(qt_app.exec_())


if __name__ == '__main__':
    main()
