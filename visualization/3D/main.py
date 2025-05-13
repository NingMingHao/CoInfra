import sys
from PyQt5.QtWidgets import QApplication
from visualizer import MainWindow
# pip install open3d PyQt5 numpy opencv-python PyYAML

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
