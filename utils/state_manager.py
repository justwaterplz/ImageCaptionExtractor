#전역 변수 같이 사용해야 하는 것들 작성
from PyQt5.QtCore import QSettings

settings = QSettings("iClickArt", "ExtractKeywordApp")

def get_excel_checkbox_state():
    return settings.value("excel_checkbox_state", False, type=bool)

def set_excel_checkbox_state(state):
    settings.setValue("excel_checkbox_state", state)
    print(f"excel_checkbox_state saved: {state}")