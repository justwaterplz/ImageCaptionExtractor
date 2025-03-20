import base64
import logging
import re
import traceback
import json
import os

from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit, QDesktopWidget, QPushButton, QDialogButtonBox, QCheckBox, QVBoxLayout, QGroupBox, QLabel, QLineEdit
from PyQt5.uic import loadUi
from openai import OpenAI

from core.dialog.help_dialog import HelpDialog
import requests
from cfg.cfg import *

from utils.state_manager import get_excel_checkbox_state, set_excel_checkbox_state

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent)
        # config.json 파일 경로를 사용자 홈 디렉토리로 변경
        self.app_dir = os.path.join(os.path.expanduser("~"), ".imagekeywordextractor")
        if not os.path.exists(self.app_dir):
            os.makedirs(self.app_dir, exist_ok=True)
        self.config_file = os.path.join(self.app_dir, "config.json")
        print(f"Config file path: {os.path.abspath(self.config_file)}")  # 디버깅용
        
        self.setWindowTitle("설정")
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)  # 높이 증가
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # API Key 섹션
        api_key_group = QGroupBox("API Key")
        api_key_layout = QVBoxLayout()
        
        self.text_edit_api_key = QTextEdit()
        self.text_edit_api_key.setPlaceholderText("OpenAI API Key를 입력하세요")
        self.text_edit_api_key.setMaximumHeight(60)
        
        self.validate_api_key_button = QPushButton("API Key 확인")
        self.validate_api_key_button.setMinimumHeight(30)
        
        self.reset_api_key_button = QPushButton("재설정")
        self.reset_api_key_button.setMinimumHeight(30)
        
        api_key_layout.addWidget(self.text_edit_api_key)
        api_key_layout.addWidget(self.validate_api_key_button)
        api_key_layout.addWidget(self.reset_api_key_button)
        api_key_group.setLayout(api_key_layout)

        # Assistant ID 섹션 추가
        assistant_id_group = QGroupBox("Assistant ID")
        assistant_id_layout = QVBoxLayout()
        
        self.text_edit_assistant_id = QTextEdit()
        self.text_edit_assistant_id.setPlaceholderText("OpenAI Assistant ID를 입력하세요 (예: asst_abc123...)")
        self.text_edit_assistant_id.setMaximumHeight(60)
        
        self.validate_assistant_id_button = QPushButton("Assistant ID 확인")
        self.validate_assistant_id_button.setMinimumHeight(30)
        
        self.reset_assistant_id_button = QPushButton("재설정")
        self.reset_assistant_id_button.setMinimumHeight(30)
        
        # Assistant ID 도움말 추가
        self.assistant_id_help = QLabel("Assistant ID는 OpenAI 대시보드에서 생성한 Assistant의 ID입니다.\n"
                                       "https://platform.openai.com/assistants 에서 확인할 수 있습니다.")
        self.assistant_id_help.setWordWrap(True)
        assistant_id_layout.addWidget(self.text_edit_assistant_id)
        assistant_id_layout.addWidget(self.validate_assistant_id_button)
        assistant_id_layout.addWidget(self.reset_assistant_id_button)
        assistant_id_layout.addWidget(self.assistant_id_help)
        
        assistant_id_group.setLayout(assistant_id_layout)

        # 체크박스
        self.show_excel_check = QCheckBox("엑셀 파일 표시")
        self.show_excel_check.setChecked(get_excel_checkbox_state())

        # 버튼 박스
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        
        # 메인 레이아웃에 위젯 추가
        main_layout.addWidget(api_key_group)
        main_layout.addWidget(assistant_id_group)  # Assistant ID 그룹 추가
        main_layout.addWidget(self.show_excel_check)
        main_layout.addWidget(self.buttonBox)

        # 시그널 연결
        self.show_excel_check.stateChanged.connect(self.update_checkbox_state)
        self.validate_api_key_button.clicked.connect(self.validate_api_key)
        self.validate_assistant_id_button.clicked.connect(self.validate_assistant_id_button_clicked)
        self.buttonBox.accepted.connect(self.save_settings)
        self.buttonBox.rejected.connect(self.reject)
        self.reset_api_key_button.clicked.connect(self.reset_api_key)
        self.reset_assistant_id_button.clicked.connect(self.reset_assistant_id)

        # 초기 상태 설정
        self.api_key_valid = False
        self.update_buttonbox_state()

        # 기존 설정 로드
        self.load_existing_settings()

    def validate_assistant_id(self):
        """Assistant ID 유효성 검사"""
        assistant_id = self.text_edit_assistant_id.toPlainText().strip()
        
        # 이미 읽기 전용이면 이미 검증된 것으로 간주
        if self.text_edit_assistant_id.isReadOnly():
            return True
        
        # 비어있는 경우 - 선택적으로 허용할 수 있음
        if not assistant_id:
            choice = QMessageBox.question(
                self,
                "Assistant ID 확인",
                "Assistant ID가 비어있습니다. 기본값을 사용하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if choice == QMessageBox.Yes:
                # 비어있는 값을 허용하는 경우 필드와 버튼 비활성화
                self.text_edit_assistant_id.setReadOnly(True)
                self.validate_assistant_id_button.setEnabled(False)
                return True
            return False
        
        # Assistant ID 형식 검사 (asst_로 시작하고 영숫자로 구성)
        pattern = r'^asst_[a-zA-Z0-9]{24,}$'
        if re.match(pattern, assistant_id):
            # 유효성 검사 통과 후 필드와 버튼 비활성화
            self.text_edit_assistant_id.setReadOnly(True)
            self.validate_assistant_id_button.setEnabled(False)
            return True
        else:
            QMessageBox.warning(
                self,
                "Assistant ID 오류",
                "유효하지 않은 Assistant ID 형식입니다.\n'asst_'로 시작하는 올바른 ID를 입력해주세요."
            )
            return False
            
    def validate_assistant_id_button_clicked(self):
        """Assistant ID 확인 버튼 클릭 핸들러"""
        assistant_id = self.text_edit_assistant_id.toPlainText().strip()
        
        # 비어있는 경우 - 선택적으로 허용할 수 있음
        if not assistant_id:
            choice = QMessageBox.question(
                self,
                "Assistant ID 확인",
                "Assistant ID가 비어있습니다. 기본값을 사용하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if choice == QMessageBox.Yes:
                # 비어있는 값을 허용하는 경우 필드와 버튼 비활성화
                self.text_edit_assistant_id.setReadOnly(True)
                self.validate_assistant_id_button.setEnabled(False)
                QMessageBox.information(self, "성공", "기본 Assistant ID를 사용합니다.")
                return True
            return False
        
        # Assistant ID 형식 검사 (asst_로 시작하고 영숫자로 구성)
        pattern = r'^asst_[a-zA-Z0-9]{24,}$'
        if re.match(pattern, assistant_id):
            # 유효성 검사 통과 후 필드와 버튼 비활성화
            self.text_edit_assistant_id.setReadOnly(True)
            self.validate_assistant_id_button.setEnabled(False)
            QMessageBox.information(self, "성공", "Assistant ID 형식이 유효합니다.")
            return True
        else:
            QMessageBox.warning(
                self,
                "Assistant ID 오류",
                "유효하지 않은 Assistant ID 형식입니다.\n'asst_'로 시작하는 올바른 ID를 입력해주세요."
            )
            return False

    def load_existing_settings(self):
        """기존 설정 불러오기"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                    # API 키 설정
                    if 'openai_key' in config and config['openai_key'].strip():
                        self.text_edit_api_key.setText(config['openai_key'])
                        self.api_key_valid = True
                        # 이미 유효한 API 키가 있으면 필드와 버튼 비활성화
                        self.text_edit_api_key.setReadOnly(True)
                        self.validate_api_key_button.setEnabled(False)
                        
                    # Assistant ID 설정
                    if 'assistant_id' in config and config['assistant_id'].strip():
                        self.text_edit_assistant_id.setText(config['assistant_id'])
                        # 이미 유효한 Assistant ID가 있으면 필드와 버튼 비활성화
                        self.text_edit_assistant_id.setReadOnly(True)
                        self.validate_assistant_id_button.setEnabled(False)
                        
                    print(f"기존 설정 불러옴: API Key={bool(config.get('openai_key'))}, Assistant ID={bool(config.get('assistant_id'))}")
                    
                    # 버튼 상태 업데이트
                    self.update_buttonbox_state()
        except Exception as e:
            print(f"기존 설정 불러오기 오류: {str(e)}")
            # 오류 발생 시 빈 설정 사용 (기본값)

    def update_buttonbox_state(self):
        self.buttonBox.button(QDialogButtonBox.Save).setEnabled(self.api_key_valid)

    def update_checkbox_state(self, state):
        set_excel_checkbox_state(bool(state))

    def eventFilter(self, obj, event):
        if event.type() == QEvent.EnterWhatsThisMode:
            self.show_help_dialog()
            return True
        return super().eventFilter(obj, event)

    #입력하는 화면에서 "취소" 누를 시 입력 실패
    def reject(self):
        print("입력 실패")
        super().reject()

    #api key를 정규표현식을 사용해서 유효성 검사함
    def validate_api_key(self):
        api_key = self.text_edit_api_key.toPlainText().strip()
        self.text_edit_api_key.setReadOnly(True)
        self.validate_api_key_button.setEnabled(False)

        logging.debug(f"Validating API key: {api_key[:5]}...{api_key[-5:]}")

        if not api_key:
            QMessageBox.warning(self, "API Key 오류", "API Key를 입력해주세요.")
            self.api_key_valid = False
            self.text_edit_api_key.setReadOnly(False)  # 다시 편집 가능하게
            self.validate_api_key_button.setEnabled(True)  # 버튼 활성화
        else:
            try:
                client = OpenAI(api_key=api_key)
                models = client.models.list()
                logging.debug(f"Models retrieved: {models}")
                self.api_key_valid = True
                QMessageBox.information(self, "성공", "API 키가 유효합니다.")
                
                # 유효성 검사 통과 후 필드와 버튼 비활성화
                self.text_edit_api_key.setReadOnly(True)
                self.validate_api_key_button.setEnabled(False)
            except Exception as e:
                logging.error(f"Error validating API key: {str(e)}")
                logging.error(traceback.format_exc())
                self.api_key_valid = False
                QMessageBox.critical(self, "오류", f"API 키 검증 중 오류 발생:\n{str(e)}")
                
                # 오류 발생 시 다시 편집 가능하게
                self.text_edit_api_key.setReadOnly(False)
                self.validate_api_key_button.setEnabled(True)

        self.update_buttonbox_state()
        logging.debug("API key validation completed")

    def show_help_dialog(self):
        help_content = self.get_help_content()
        help_dialog = HelpDialog("설정 도움말", help_content, self)
        
        #도움말 화면 중앙 배치
        screen = QDesktopWidget().screenNumber(QDesktopWidget().cursor().pos())
        center_point = QDesktopWidget().screenGeometry(screen).center()
        frame_geometry = help_dialog.frameGeometry()
        frame_geometry.moveCenter(center_point)
        help_dialog.move(frame_geometry.topLeft())
        help_dialog.exec_()

    def get_help_content(self):
        return """
            <style>
                body {
                    font-family: 'Malgun Gothic', '맑은 고딕', Arial, sans-serif;
                    font-size: 14px;
                }
                h1 {
                    font-size: 24px;
                    color: #333;
                }
                h2 {
                    font-size: 20px;
                    color: #444;
                }
                p, li {
                    font-size: 16px;
                    line-height: 1.5;
                }
                code {
                    background-color: #f0f0f0;
                    padding: 2px 4px;
                    border-radius: 4px;
                }
                
            </style>
            <h1>설정 도움말</h1>
            <h2>API KEY 확인하는 방법</h2>
            <ol>
                <li>OpenAI 웹사이트(<a href='https://platform.openai.com/api-keys'>https://platform.openai.com/api-keys</a>)에 접속하여 로그인 또는 회원가입을 진행합니다.</li>
                <li>우측 상단 톱니바퀴 버튼을 눌러 설정 창에 진입합니다.</li>
                <li>왼쪽 "billing" 메뉴에서 "payment methods"를 클릭하여 결제수단을 등록합니다.</li>
                <li>"overview"에서 예상되는 API 사용량만큼의 금액을 충전합니다. (최소 5$)</li>
                <li>OpenAI 웹사이트에서 "create new secret key"를 눌러 기본 설정과 함께 api key를 생성합니다.</li>
            </ol>
            <p>주의: API 키는 비밀번호와 같이 중요한 정보이므로 절대 타인과 공유하지 마세요.</p>

            <h2>Assistant ID</h2>
            <p>Assistant ID는 OpenAI의 Assistants API를 사용하기 위한 식별자입니다.</p>
            <ol>
                <li>OpenAI 웹사이트(<a href='https://platform.openai.com/assistants'>https://platform.openai.com/assistants</a>)에서 로그인합니다.</li>
                <li>"Create" 버튼을 클릭하여 새 Assistant를 생성합니다.</li>
                <li>필요한 이름과 설명, 모델, 기능 등을 설정합니다.</li>
                <li>생성 후 Assistant 페이지에서 Assistant ID(asst_로 시작하는 문자열)를 확인할 수 있습니다.</li>
                <li>이 ID를 복사하여 설정에 붙여넣습니다.</li>
            </ol>
            <p>비워두면 기본 Assistant ID가 사용됩니다.</p>
        """
    
    def save_settings(self):
        """설정 저장"""
        try:
            # 설정 파일 디렉토리 확인
            if not os.path.exists(self.app_dir):
                os.makedirs(self.app_dir, exist_ok=True)

            # API 키가 비어있는지 확인
            api_key = self.text_edit_api_key.toPlainText().strip()
            if not api_key:
                choice = QMessageBox.question(
                    self,
                    "설정 경고",
                    "API 키가 없습니다. 계속 진행하시겠습니까?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if choice == QMessageBox.No:
                    return

            # Assistant ID 가져오기 및 유효성 검사
            assistant_id = self.text_edit_assistant_id.toPlainText().strip()
            if assistant_id and not self.validate_assistant_id():
                return  # 유효하지 않은 Assistant ID면 저장하지 않음

            # 기존 설정 불러오기 또는 새 설정 생성
            if os.path.exists(self.config_file):
                try:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        settings = json.load(f)
                except:
                    settings = {}
            else:
                settings = {}

            # 설정 업데이트
            settings['openai_key'] = api_key
            settings['assistant_id'] = assistant_id
            
            # 설정 저장
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
            
            print(f"설정 저장됨: API Key={bool(api_key)}, Assistant ID={bool(assistant_id)}")
            self.accept()
        except Exception as e:
            print(f"설정 저장 중 오류: {str(e)}")
            traceback.print_exc()
            QMessageBox.critical(self, "오류", f"설정을 저장할 수 없습니다: {str(e)}")
            return

    def closeEvent(self, event):
        QMessageBox.warning(self, '경고', '완료되지 않았습니다.', QMessageBox.Ok)
        event.ignore()

    def accept(self):
        # 설정을 저장하고 다이얼로그를 닫을 때 호출되는 메서드
        set_excel_checkbox_state(self.show_excel_check.isChecked())
        print(f"Settings saved. Excel checkbox state: {get_excel_checkbox_state()}")
        super().accept()

    def reset_api_key(self):
        """API Key 필드 재설정"""
        self.text_edit_api_key.setReadOnly(False)
        self.validate_api_key_button.setEnabled(True)
        self.api_key_valid = False
        self.update_buttonbox_state()

    def reset_assistant_id(self):
        """Assistant ID 필드 재설정"""
        self.text_edit_assistant_id.setReadOnly(False)
        self.validate_assistant_id_button.setEnabled(True)
