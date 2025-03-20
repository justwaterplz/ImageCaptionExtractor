from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton
#설정 변경 화면에서 "?" 버튼을 눌렀을 때 나오는 도움말 dialog
class HelpDialog(QDialog):
    def __init__(self, title, content, parent=None, position=None):
        super(HelpDialog, self).__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 600)

        layout = QVBoxLayout()

        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        text_browser.setHtml(content)

        close_button = QPushButton("닫기")
        close_button.clicked.connect(self.accept)

        layout.addWidget(text_browser)
        layout.addWidget(close_button)

        self.setLayout(layout)

        if position:
            self.move(position)
