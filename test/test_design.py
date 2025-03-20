import os

from PyQt5.QtCore import Qt, QSize, QPoint, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
from PyQt5.QtGui import QPixmap, QIcon, QFontDatabase, QFont
from PyQt5.QtWidgets import QMainWindow, QWidget, QGraphicsOpacityEffect
from PyQt5.uic import loadUi

from cfg.cfg import resource_path, ui_dir, css_dir
from utils.styles import read_stylesheet
import res.resources_rc


class TestDesign(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi(resource_path(os.path.join(ui_dir, "sidebar.ui")), self)

        self.icon_only_widget.hide()
        self.page_widget.setCurrentIndex(0)
        self.home_btn2.setChecked(True)
        # add icons on QLabel components
        self.set_pixmap_from_resource(self.home_icon1, ":/icons/img/iclickart_logo.png")
        self.set_pixmap_from_resource(self.home_icon2, ":/icons/img/iclickart_logo.png")
        self.set_icon_from_resource(self.spread_btn, ":/icons/img/icons8-menu-24.png")
        # 2 exit buttons
        self.set_icon_from_resource(self.exit_btn1, ":/icons/img/icons8-x-16.png")
        # load custom fonts from resources.qrc
        self.setup_custom_font()
        self.setStyleSheet(read_stylesheet(resource_path(os.path.join(css_dir, "main.css"))))

    # pixmap set up function(QLabel)
    def set_pixmap_from_resource(self, label, pixmap_path, width=None, height=None):
        pixmap = QPixmap(pixmap_path)
        if width and height:
            pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
    # icon set up function(QPushbutton)
    def set_icon_from_resource(self, btn, icon_path, width=None, height=None):
        icon = QIcon(icon_path)
        if width and height:
            icon = icon.scaled(width, height, Qt.SmoothTransformation)
        btn.setIcon(icon)
    def setup_custom_font(self):
        font_path = ":/fonts/fonts/NotoSansKR-Regular.ttf"
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            self.default_font = QFont(font_family, 9)
            self.setFont(self.default_font)

            button_font = QFont(font_family, 11)
            self.home_btn2.setFont(button_font)
            self.refresh_btn2.setFont(button_font)
            self.search_btn2.setFont(button_font)
            self.settings_btn2.setFont(button_font)
            self.exit_btn2.setFont(button_font)

            # 다른 버튼들에도 같은 방식으로 적용할 수 있습니다.
        else:
            print("Failed to load custom font")