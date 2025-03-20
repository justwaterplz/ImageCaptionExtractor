import os
import subprocess
import json

from PyQt5.QtCore import Qt, QSize, pyqtSignal, QFileInfo
from PyQt5.QtGui import QPixmap, QIcon, QFontDatabase, QFont, QImage, QColor, QCursor
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpacerItem, QSizePolicy,
    QStackedWidget, QMenuBar, QStatusBar, QGridLayout,
    QTableWidget, QHeaderView, QCheckBox, QMenu, QAction,
    QMessageBox, QAbstractItemView, QTableWidgetSelectionRange, QFileIconProvider, QTableWidgetItem,
    QDialog, QCheckBox, QDesktopWidget
)
from core.dialog.setting_dialog import SettingsDialog
from core.services.file_operations import FileOperations
from core.services.settings_handler import SettingsHandler
from core.services.image_processor import ImageProcessor
from utils.keyword_manager import KeywordManager
from utils.state_manager import get_excel_checkbox_state
from utils.styles import read_stylesheet
from cfg.cfg import *
# import res.resources_rc
import sys

from utils.worker_thread import WorkerThread


class MainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        print("MainUI initialization start")
        
        # settings_handler를 먼저 초기화
        self.settings_handler = SettingsHandler(self, "config.json")
        
        # 설정 확인을 먼저 수행
        self.check_settings()
        
        self.file_operations = FileOperations(self, config_file="config.json")
        
        # ImageProcessor는 settings_handler 초기화 후에 생성
        self.image_processor = ImageProcessor(self, self.settings_handler)
        
        self.init_ui()
        self.setup_signals()
        
        # 디버깅을 위한 정보 출력
        print("MainUI initialization complete")

    def init_ui(self):
        # 기본 창 설정
        screen = QDesktopWidget().screenGeometry()
        width = 1600
        height = 900
        x = (screen.width() - width) // 2
        y = (screen.height() - height) // 2

        self.setGeometry(x, y, 1600, 900)
        self.setWindowTitle("MainWindow")

        # Central Widget 설정
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QGridLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Full Menu Widget (사이드바)
        self.setup_full_menu_widget()

        # Content Widget (메인 콘텐츠)
        self.setup_content_widget()

        # 위젯들을 메인 레이아웃에 추가
        self.main_layout.addWidget(self.full_menu_widget, 0, 0)
        self.main_layout.addWidget(self.content_widget, 0, 1)

        # 열 너비 설정 추가
        self.main_layout.setColumnStretch(0, 0)  # full_menu_widget
        self.main_layout.setColumnStretch(1, 1)  # content_widget

        # 초기 상태 설정
        self.page_widget.setCurrentIndex(0)
        # self.home_btn2.setChecked(True)

        # 스타일 설정
        self.setup_custom_font()
        self.setStyleSheet(read_stylesheet(resource_path(os.path.join(css_dir, "main.css"))))

        # 디버깅을 위한 정보 출력
        print("레이아웃 디버깅 정보:")
        print(f"Content Widget 위치: {self.content_widget.pos()}")
        print(f"Content Widget 크기: {self.content_widget.size()}")
        print(f"Stack Widget 위치: {self.page_widget.pos()}")
        print(f"Stack Widget 크기: {self.page_widget.size()}")

        # 다른 버튼들 초기화 (settings_btn2 제외)
        self.buttons = [
            self.refresh_btn2,
            self.add_btn,
            self.delete_btn,
            self.select_all_btn,
            self.send_data_btn,
            self.exit_btn2
        ]

        # 설정 버튼 연결
        self.settings_btn2.clicked.connect(self.show_settings_dialog)
        
        # 버튼 시그널 연결
        self.add_btn.clicked.connect(self.file_operations.load_files)
        self.delete_btn.clicked.connect(self.delete_selected_items)
        self.select_all_btn.clicked.connect(self.toggle_select_all)
        self.send_data_btn.clicked.connect(self.on_send_data)
        
        # 테이블 시그널 연결
        self.setup_table_signals()

        # 테이블 설정
        self.image_table.setColumnCount(1)
        self.image_table.setHorizontalHeaderLabels(['이미지'])
        
        # 테이블 스타일 설정 호출
        self.setup_table_style()

        # 새로고침 버튼 연결
        self.refresh_btn2.clicked.connect(self.refresh_table)

    def check_settings(self):
        """설정 확인"""
        print("Checking settings...")
        api_key = self.settings_handler.get_setting('openai_key')
        assistant_id = self.settings_handler.get_setting('assistant_id')
        print(f"Initial API Key exists: {bool(api_key)}")
        print(f"Initial Assistant ID exists: {bool(assistant_id)}")
        
        while not api_key or not assistant_id:
            print("API key or Assistant ID not found, opening settings dialog...")
            QMessageBox.warning(self, "설정 필요", "API Key와 Assistant ID를 설정해주세요.")
            dialog_result = self.settings_handler.open_settings_dialog()
            if dialog_result != QDialog.Accepted:
                print("Settings dialog cancelled")
                sys.exit()
            
            # 설정 다이얼로그 종료 후 API 키와 Assistant ID 다시 확인
            api_key = self.settings_handler.get_setting('openai_key')
            assistant_id = self.settings_handler.get_setting('assistant_id')
            print(f"After dialog API Key exists: {bool(api_key)}")
            print(f"After dialog Assistant ID exists: {bool(assistant_id)}")
            
            if not api_key or not assistant_id:
                print("API key or Assistant ID still not set after dialog")
        
        print(f"Final API Key validation successful: {bool(api_key)}")
        print(f"Final Assistant ID validation successful: {bool(assistant_id)}")
        return True

    def disable_buttons(self):
        """설정 버튼을 제외한 모든 버튼 비활성화"""
        print("Disabling buttons...")
        for button in self.buttons:
            button.setEnabled(False)
        # 설정 버튼은 항상 활성화
        self.settings_btn2.setEnabled(True)

    def enable_buttons(self):
        """모든 버튼 활성화"""
        print("Enabling buttons...")
        for button in self.buttons:
            button.setEnabled(True)

    def show_settings_dialog(self):
        """설정 다이얼로그 표시"""
        try:
            settings_dialog = SettingsDialog(self)
            if settings_dialog.exec_() == QDialog.Accepted:
                # 설정이 저장되면 버튼 상태 업데이트
                if self.check_settings():
                    print("Settings saved successfully")
                    self.enable_buttons()
                    QMessageBox.information(self, '설정 완료', '설정이 성공적으로 저장되었습니다.')
                else:
                    print("Settings validation failed")
                    self.disable_buttons()
        except Exception as e:
            print(f"Error in show_settings_dialog: {str(e)}")
            QMessageBox.critical(self, '오류', f'설정 창을 열 수 없습니다: {str(e)}')

    def setup_full_menu_widget(self):
        self.full_menu_widget = QWidget()
        self.full_menu_widget.setStyleSheet("""
            QWidget {
                margin: 0px;
                padding: 0px;
            }
        """)
        self.full_menu_widget.setMinimumWidth(168)
        self.full_menu_widget.setMaximumWidth(168)
        menu_layout = QVBoxLayout(self.full_menu_widget)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        menu_layout.setSpacing(0)

        # 로고
        logo_layout = QHBoxLayout()
        logo_layout.setContentsMargins(5, 20, 0, 20)  # 상단 여백 추가
        self.home_icon2 = QLabel()
        self.home_icon2.setMinimumSize(50, 50)
        self.set_pixmap_from_resource(self.home_icon2, ":/icons/img/iclickart_logo.png")
        logo_layout.addWidget(self.home_icon2, alignment=Qt.AlignHCenter)
        menu_layout.addLayout(logo_layout)

        # 상단 여백 추가
        menu_layout.addSpacing(20)

        # 버튼들
        button_layout = QVBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(5)

        # 버튼 생성 및 설정
        # self.home_btn2 = self.create_text_button("Home")
        self.refresh_btn2 = self.create_text_button("새로고침")
        # self.search_btn2 = self.create_text_button("검색")
        self.settings_btn2 = self.create_text_button("설정")

        # 버튼 스타일 설정
        for btn in [self.refresh_btn2, self.settings_btn2]:
            btn.setFixedWidth(168)
            btn.setMinimumHeight(40)

        # button_layout.addWidget(self.home_btn2)
        button_layout.addWidget(self.refresh_btn2)
        # button_layout.addWidget(self.search_btn2)
        button_layout.addWidget(self.settings_btn2)
        menu_layout.addLayout(button_layout)

        # Spacer
        menu_layout.addItem(QSpacerItem(20, 584, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Exit 버튼
        self.exit_btn2 = QPushButton("종료")
        self.exit_btn2.setMinimumHeight(40)
        self.exit_btn2.setMinimumWidth(165)
        menu_layout.addWidget(self.exit_btn2)

    def setup_content_widget(self):
        self.content_widget = QWidget()
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # 스택 위젯 설정
        self.setup_stack_widget()
        
        # 스택 위젯이 공간을 차지하도록 설정
        self.page_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_layout.addWidget(self.page_widget, stretch=1)

    def setup_stack_widget(self):
        self.page_widget = QStackedWidget()
        
        # Main Page
        self.main_page = QWidget()
        self.main_page.setObjectName("main_page")
        main_page_layout = QHBoxLayout(self.main_page)
        main_page_layout.setContentsMargins(0, 0, 0, 0)

        # 왼쪽 영역 (파일 리스트)
        left_widget = QWidget()
        left_widget.setObjectName("left_widget")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 이미지 테이블 설정
        self.image_table = QTableWidget()
        self.image_table.setObjectName("image_table")
        
        # 테이블 기본 설정
        self.image_table.setColumnCount(1)
        self.image_table.setHorizontalHeaderLabels(['이미지'])
        self.image_table.verticalHeader().setDefaultSectionSize(60)
        self.image_table.setShowGrid(False)
        self.image_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.image_table.verticalHeader().setVisible(False)
        
        # 헤더 선택 방지 및 테이블 선택 모드 설정
        self.image_table.horizontalHeader().setHighlightSections(False)  # 헤더 하이라이트 비활성화
        self.image_table.setSelectionBehavior(QAbstractItemView.SelectRows)  # 행 단위 선택
        self.image_table.setSelectionMode(QAbstractItemView.MultiSelection)  # 다중 선택 가능
        
        # 테이블 크기 정책 설정
        self.image_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_table.setMinimumSize(400, 300)
        
        # 테이블을 레이아웃에 추가
        left_layout.addWidget(self.image_table)
        
        # 오른쪽 영역 (버튼)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # 상단 여백을 위한 spacer
        right_layout.addSpacing(50)  # 위쪽 여백 추가

        # 버튼들을 담을 컨테이너 위젯
        button_container = QWidget()
        button_container_layout = QVBoxLayout(button_container)
        button_container_layout.setContentsMargins(0, 0, 0, 0)
        button_container_layout.setSpacing(10)
        button_container_layout.setAlignment(Qt.AlignHCenter)  # 가운데 정렬

        # 버튼들 생성 및 추가
        self.add_btn = QPushButton("이미지 추가")
        self.delete_btn = QPushButton("선택 삭제")
        self.select_all_btn = QPushButton("전체 선택/해제")
        self.send_data_btn = QPushButton("선택 전송")
        self.send_data_btn.setEnabled(False)

        # 버튼 스타일 및 크기 설정
        for btn in [self.add_btn, self.delete_btn, self.select_all_btn, self.send_data_btn]:
            btn.setMinimumHeight(30)  # 버튼 사이즈 설정
            btn.setMinimumWidth(120)
            button_container_layout.addWidget(btn, 0, Qt.AlignHCenter)  # 가운데 정렬로 추가

        # 버튼 컨테이너를 오른쪽 레이아웃에 추가
        right_layout.addWidget(button_container, 0, Qt.AlignHCenter)
        
        # 나머지 공간을 채우는 신축성 있는 공간 추가
        right_layout.addStretch()

        # 메인 페이지 레이아웃에 위젯 추가
        main_page_layout.addWidget(left_widget, stretch=4)
        main_page_layout.addWidget(right_widget, stretch=1)

        # 스택 위젯에 페이지들 추가
        self.page_widget.addWidget(self.main_page)
        self.page_widget.setCurrentIndex(0)

        # 테이블 시그널 연결
        self.image_table.itemSelectionChanged.connect(self.update_send_button)
        self.image_table.cellDoubleClicked.connect(self.open_image_doubleclick)
        self.image_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_table.customContextMenuRequested.connect(self.show_context_menu)

    def create_icon_button(self, name, icon_path):
        btn = QPushButton()
        btn.setMinimumHeight(36)
        self.set_icon_from_resource(btn, icon_path)
        btn.setIconSize(QSize(16, 16))
        btn.setCheckable(True)
        btn.setAutoExclusive(True)
        return btn

    def create_text_button(self, text):
        btn = QPushButton(text)
        btn.setMinimumHeight(36)
        btn.setCheckable(True)
        btn.setAutoExclusive(True)
        return btn

    def connect_signals(self):
        # 파일 리스트 관련 버튼 시그널 연결
        self.add_btn.clicked.connect(self.load_files)
        self.delete_btn.clicked.connect(self.delete_selected_items)
        self.select_all_btn.clicked.connect(self.toggle_select_all)
        self.send_data_btn.clicked.connect(self.send_selected_images)
        self.exit_btn2.clicked.connect(self.close)

    def set_pixmap_from_resource(self, label, pixmap_path, width=None, height=None):
        pixmap = QPixmap(pixmap_path)
        if width and height:
            pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def set_icon_from_resource(self, btn, icon_path, width=None, height=None):
        icon = QIcon(icon_path)
        if width and height:
            icon = icon.scaled(width, height, Qt.SmoothTransformation)
        btn.setIcon(icon)

    def setup_services(self):
        # 서비스 및 설정 초기화
        self.config_file = cfg_path
        self.file_operations = FileOperations(self, self.config_file)
        self.settings_handler = SettingsHandler(self, self.config_file)
        self.image_processor = ImageProcessor(self)

        # 상태 변수 초기화
        self.progress = None
        self.worker = None
        self.excel_file_path = None
        self.transmitted_files = []

        # 초기 설정 확인 및 설정 다이얼로그 표시
        if not self.settings_handler.check_settings():
            self.settings_handler.open_settings_dialog()

        # 시그널 연결
        self.image_table.itemSelectionChanged.connect(self.update_send_button)
        self.image_processor.progress_updated.connect(self.update_progress)
        self.image_processor.process_finished.connect(self.on_process_finished)
        self.image_processor.error_occurred.connect(self.on_error)

        # Excel 상태 라벨 업데이트
        if hasattr(self, 'label_excel_state'):
            self.update_excel_state_label()

    def update_send_button(self):
        # 선택된 항목이 있는지 확인
        selected_items = self.get_selected_files()
        if hasattr(self, 'send_data_btn'):
            self.send_data_btn.setEnabled(len(selected_items) > 0)

    def update_progress(self, value):
        """진행 상태 업데이트"""
        print(f"Progress: {value}%")
        # 프로그레스바 업데이트 로직

    def on_process_finished(self):
        if self.progress:
            self.progress.close()
        if hasattr(self, 'widget_file_list'):
            self.widget_file_list.clear()

    def on_error(self, error_message):
        """에러 처리"""
        print(f"Error occurred: {error_message}")
        QMessageBox.critical(self, '오류', f'처리 중 오류가 발생했습니다: {error_message}')

    def update_excel_state_label(self):
        excel_state = get_excel_checkbox_state()
        state_text = "비활성화" if excel_state else "활성화"
        if hasattr(self, 'label_excel_state'):
            self.label_excel_state.setText(f"엑셀 자동 열기: {state_text}")

    def add_file_to_list(self, file_path):
        """테이블에 파일 추가"""
        try:
            row = self.image_table.rowCount()
            self.image_table.insertRow(row)
            
            # 체크박스 추가
            checkbox_widget = self.setup_checkbox(row)
            self.image_table.setCellWidget(row, 0, checkbox_widget)
            
            # 파일 정보 추가
            file_info = QFileInfo(file_path)
            file_name = file_info.fileName()
            
            # 파일명 아이템 설정
            name_item = QTableWidgetItem(file_name)
            name_item.setData(Qt.UserRole, file_path)  # 전체 경로 저장
            self.image_table.setItem(row, 1, name_item)
            
            self.update_button_states()
            
        except Exception as e:
            print(f"Error adding file to list: {str(e)}")

    def create_item_widget(self, file_info, custom_icon, file_path):
        """테이블 항목 위젯 생성"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 0, 5, 0)
        layout.setSpacing(10)

        # 체크박스 추가
        checkbox = QCheckBox()
        checkbox.setStyleSheet("margin-left: 5px;")
        layout.addWidget(checkbox)

        # 썸네일 라벨
        icon_label = QLabel()
        try:
            thumbnail = self.create_thumbnail(file_path)
            icon_label.setPixmap(thumbnail)
        except Exception as e:
            print(f"썸네일 생성 실패: {str(e)}")
            if custom_icon and not custom_icon.isNull():
                icon_label.setPixmap(custom_icon.pixmap(24, 24))
            else:
                default_icon = QFileIconProvider().icon(file_info)
                icon_label.setPixmap(default_icon.pixmap(24, 24))
        layout.addWidget(icon_label)

        # 파일명 라벨
        file_name_label = QLabel(file_info.fileName())
        file_name_label.setStyleSheet("padding-left: 5px;")
        layout.addWidget(file_name_label)

        widget.setProperty("file_path", file_path)
        widget.setProperty("checkbox", checkbox)  # 체크박스 참조 저장
        layout.addStretch()

        # 체크박스 상태 변경 시그널 연결
        checkbox.stateChanged.connect(lambda state, row=self.image_table.rowCount(): 
            self.on_checkbox_changed(state, row))

        return widget

    def create_thumbnail(self, file_path):
        """이미지 썸네일 생성"""
        image = QImage(file_path)
        if image.isNull():
            raise Exception("이미지를 불러올 수 없습니다.")
        thumbnail = image.scaled(QSize(50, 50), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QPixmap.fromImage(thumbnail)

    def on_item_clicked(self, row, column):
        widget = self.image_table.cellWidget(row, column)
        if widget:
            checkbox = widget.layout().itemAt(0).widget()
            if isinstance(checkbox, QCheckBox):
                checkbox.setChecked(not checkbox.isChecked())

    def on_checkbox_changed(self, state, row):
        """체크박스 상태 변경 처리"""
        self.image_table.selectRow(row) if state == Qt.Checked else self.image_table.clearSelection()
        self.update_button_states()

    def update_row_color(self, row, color):
        self.image_table.cellWidget(row, 0).setStyleSheet(f"background-color: {color.name()};")

    def get_selected_files(self):
        """선택된 파일 목록 반환"""
        selected_files = []
        for row in range(self.image_table.rowCount()):
            widget = self.image_table.cellWidget(row, 0)
            if widget:
                checkbox = widget.property("checkbox")
                if checkbox and checkbox.isChecked():
                    file_path = widget.property("file_path")
                    if file_path:
                        selected_files.append(file_path)
        return selected_files

    def is_allowed_format(self, file_path):
        _, file_format = os.path.splitext(file_path)
        return file_format.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}

    def get_custom_icon(self, file_path):
        """파일 형식에 따른 아이콘 반환"""
        _, file_format = os.path.splitext(file_path)
        file_format = file_format.lower()

        icon_mapping = {
            '.jpg': 'icon_jpg.png',
            '.jpeg': 'icon_jpeg.png',
            '.png': 'icon_png.png',
            '.bmp': 'icon_bmp.png'
        }

        if file_format in icon_mapping:
            icon_path = resource_path(os.path.join("res", "img", icon_mapping[file_format]))
            if os.path.exists(icon_path):
                icon = QIcon(icon_path)
                if not icon.isNull():
                    return icon
        return QIcon()

    def open_image_doubleclick(self, row, column):
        widget = self.image_table.cellWidget(row, column)
        if widget:
            file_path = widget.property("file_path")
            if file_path:
                try:
                    if os.name == 'nt':
                        os.startfile(file_path)
                    elif os.name == 'posix':
                        opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
                        subprocess.call([opener, file_path])
                except Exception as e:
                    print(f"파일 더블클릭 실패: {str(e)}")

    def show_context_menu(self):
        context_menu = QMenu(self)
        delete_action = QAction("삭제", self)
        delete_action.triggered.connect(self.delete_selected_items)
        context_menu.addAction(delete_action)
        delete_action.setEnabled(len(self.image_table.selectedRanges()) > 0)
        context_menu.exec_(QCursor.pos())

    def toggle_select_all(self):
        """전체 선택/해제 토글"""
        if not hasattr(self, 'all_selected'):
            self.all_selected = False
        
        self.all_selected = not self.all_selected
        
        # 모든 행의 체크박스 상태 변경
        for row in range(self.image_table.rowCount()):
            checkbox = self.get_checkbox_at_row(row)
            if checkbox:
                checkbox.setChecked(self.all_selected)
        
        self.update_button_states()

    def get_checkbox_at_row(self, row):
        """특정 행의 체크박스 가져오기"""
        try:
            widget = self.image_table.cellWidget(row, 0)  # 첫 번째 열의 위젯
            if widget:
                return widget.findChild(QCheckBox)
            return None
        except Exception as e:
            print(f"Error getting checkbox at row {row}: {e}")
            return None

    def setup_checkbox(self, row):
        """체크박스 설정"""
        widget = QWidget()
        checkbox = QCheckBox()
        checkbox.setProperty("row", row)  # 체크박스에 행 번호 저장
        
        # 체크박스 상태 변경 시그널 연결
        checkbox.stateChanged.connect(lambda state, r=row: self.on_checkbox_changed(r, state))
        
        layout = QHBoxLayout(widget)
        layout.addWidget(checkbox)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)
        
        return widget

    def on_checkbox_changed(self, row, state):
        """체크박스 상태 변경 처리"""
        print(f"Checkbox changed at row {row} to state {state}")
        self.update_button_states()

    def send_selected_images(self):
        """선택된 이미지 전송"""
        try:
            print("Starting send_selected_images...")
            selected_files = self.get_selected_files()
            if not selected_files:
                print("No files selected")
                QMessageBox.warning(self, '선택 오류', '전송할 이미지를 선택해주세요.')
                return

            print(f"Selected files: {selected_files}")
            
            # KeywordManager 초기화 확인
            keyword_manager = KeywordManager()
            if not keyword_manager.is_loaded:
                print("KeywordManager failed to load keywords")
                QMessageBox.warning(self, '키워드 오류', '키워드를 로드할 수 없습니다.')
                return

            # 설정 확인
            try:
                with open(cfg_path, 'r') as f:
                    settings = json.load(f)
                    api_key = settings.get('openai_key', '').strip()
                    print(f"API Key exists: {bool(api_key)}")
            except Exception as e:
                print(f"Error loading settings: {str(e)}")
                QMessageBox.warning(self, '설정 오류', '설정을 불러올 수 없습니다.')
                return

            # ImageProcessor를 통해 이미지 처리 시작
            self.image_processor.process_images(selected_files)

        except Exception as e:
            print(f"Error in send_selected_images: {str(e)}")
            QMessageBox.critical(self, "오류", f"이미지 처리 중 오류가 발생했습니다: {str(e)}")

    def handle_result(self, file_path, response):
        """개별 결과 처리"""
        try:
            file_name = os.path.basename(file_path)
            self.logger.info(f"Processing completed for: {file_name}")
            
            # 응답이 완벽한지 확인
            is_valid_response = False
            if isinstance(response, dict):
                required_fields = ["file_type", "image_desc", "theme", "person_info", "keywords", "colors"]
                # 모든 필수 필드가 있고, 키워드가 비어있지 않은지 확인
                if all(field in response for field in required_fields) and response.get("keywords"):
                    is_valid_response = True
            
            # 테이블에서 해당 파일 항목 찾기
            found = False
            for row in range(self.image_table.rowCount()):
                item = self.image_table.item(row, 1)  # 파일명 열
                if item and item.data(Qt.UserRole) == file_path:
                    self.logger.info(f"Found matching file at row {row}")
                    
                    # 응답이 완벽한 경우에만 테이블에서 삭제
                    if is_valid_response:
                        self.logger.info(f"Removing row {row} from table (valid response)")
                        self.image_table.removeRow(row)
                    else:
                        self.logger.info(f"Not removing row {row} (invalid response)")
                        # 응답이 완벽하지 않은 경우 행 색상 변경
                        self.update_row_color(row, QColor(255, 200, 200))  # 연한 빨간색
                    
                    found = True
                    break
            
            if not found:
                self.logger.info(f"File not found in table: {file_name}")
            
            # 버튼 상태 업데이트
            self.update_button_states()
            
            # 모든 이미지가 처리되었는지 확인
            if self.image_table.rowCount() == 0:
                self.logger.info("All images processed, showing save dialog")
                # 저장 다이얼로그 표시
                if hasattr(self, 'image_processor') and self.image_processor.results:
                    self.image_processor.save_results_to_excel()
            
        except Exception as e:
            self.logger.error(f"Error in handle_result: {str(e)}")

    def process_complete(self):
        """모든 처리 완료"""
        print("All images processed")
        QMessageBox.information(self, '완료', '모든 이미지 처리가 완료되었습니다.')
        
        # 테이블이 비어있는지 확인하고 버튼 상태 업데이트
        if self.image_table.rowCount() == 0:
            self.update_button_states()

    def handle_error(self, error_msg):
        """에러 처리"""
        print(f"Error occurred: {error_msg}")
        QMessageBox.critical(self, '오류', f'처리 중 오류가 발생했습니다: {error_msg}')

    def delete_selected_items(self):
        """선택된 항목 삭제"""
        rows_to_delete = []
        
        # 모든 행을 순회하면서 체크된 항목 찾기
        for row in range(self.image_table.rowCount()):
            # 첫 번째 열의 위젯 아이템 가져오기
            item = self.image_table.cellWidget(row, 0)
            if item:
                # 레이아웃에서 체크박스 찾기
                layout = item.layout()
                if layout:
                    for i in range(layout.count()):
                        widget = layout.itemAt(i).widget()
                        if isinstance(widget, QCheckBox) and widget.isChecked():
                            rows_to_delete.append(row)
                            break
        
        # 삭제할 항목이 있는 경우에만 처리
        if rows_to_delete:
            # 역순으로 정렬하여 삭제 (인덱스 변화 방지)
            for row in sorted(rows_to_delete, reverse=True):
                self.image_table.removeRow(row)
            
            # 삭제 완료 메시지 표시
            QMessageBox.information(self, "삭제 완료", f"{len(rows_to_delete)}개의 항목이 삭제되었습니다.")
        
        # 버튼 상태 업데이트
        self.update_button_states()

    def update_button_states(self):
        """버튼 상태 업데이트"""
        try:
            # 테이블에 항목이 있는지 확인
            has_items = self.image_table.rowCount() > 0
            
            # 선택된 항목이 있는지 확인
            has_selected = self.file_operations.has_selected_files()
            
            # 버튼 상태 업데이트
            self.select_all_btn.setEnabled(has_items)  # 항목이 있으면 활성화
            self.send_data_btn.setEnabled(has_selected)  # 선택된 항목이 있으면 활성화
            self.delete_btn.setEnabled(has_selected)  # 선택된 항목이 있으면 활성화
            
        except Exception as e:
            print(f"Error updating button states: {str(e)}")

    def setup_custom_font(self):
        font_path = ":/fonts/fonts/NotoSansKR-Regular.ttf"
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            self.default_font = QFont(font_family, 9)
            self.setFont(self.default_font)

            button_font = QFont(font_family, 11)
            # self.home_btn2.setFont(button_font)
            self.refresh_btn2.setFont(button_font)
            # self.search_btn2.setFont(button_font)
            self.settings_btn2.setFont(button_font)
            self.exit_btn2.setFont(button_font)
        else:
            print("Failed to load custom font")

    def setup_table_signals(self):
        """테이블 시그널 설정"""
        self.image_table.itemSelectionChanged.connect(self.update_button_states)
        self.image_table.cellDoubleClicked.connect(self.open_image_preview)
        self.image_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_table.customContextMenuRequested.connect(self.show_context_menu)

    def open_image_preview(self, row, column):
        """이미지 미리보기 열기"""
        try:
            widget = self.image_table.cellWidget(row, column)
            if widget:
                file_path = widget.property("file_path")
                if file_path and os.path.exists(file_path):
                    if os.name == 'nt':  # Windows
                        os.startfile(file_path)
                    else:  # macOS 및 Linux
                        import subprocess
                        opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
                        subprocess.call([opener, file_path])
        except Exception as e:
            print(f"Error opening image preview: {str(e)}")
            QMessageBox.critical(self, '오류', f'이미지를 열 수 없습니다: {str(e)}')

    def setup_signals(self):
        """시그널 연결 설정"""
        try:
            # 기존 연결 모두 해제
            self.add_btn.clicked.disconnect()
            self.delete_btn.clicked.disconnect()
            self.select_all_btn.clicked.disconnect()
            self.send_data_btn.clicked.disconnect()
            self.settings_btn2.clicked.disconnect()
        except:
            # 연결이 없는 경우 예외 발생하므로 무시
            pass
        
        # 새로운 연결 설정
        self.add_btn.clicked.connect(self.file_operations.load_files)
        self.delete_btn.clicked.connect(self.delete_selected_items)
        self.select_all_btn.clicked.connect(self.file_operations.toggle_select_all)
        self.send_data_btn.clicked.connect(self.on_send_data)
        self.settings_btn2.clicked.connect(self.show_settings_dialog)

    def on_send_data(self):
        """이미지 전송 버튼 클릭 시 호출"""
        try:
            selected_files = self.file_operations.get_selected_files()
            if not selected_files:
                QMessageBox.warning(self, "경고", "선택된 이미지가 없습니다.")
                return

            # 설정에서 API 키 가져오기
            api_key = self.settings_handler.get_setting('openai_key')
            if not api_key:
                QMessageBox.warning(self, "설정 오류", "OpenAI API 키가 설정되지 않았습니다.")
                return

            # ImageProcessor에 API 키 전달
            self.image_processor.set_api_key(api_key)  
            self.image_processor.process_images(selected_files)

        except Exception as e:
            print(f"Error in on_send_data: {str(e)}")
            QMessageBox.critical(self, "오류", f"이미지 처리 중 오류가 발생했습니다: {str(e)}")

    def setup_table_style(self):
        """테이블 스타일 설정"""
        self.image_table.setStyleSheet("""
            QTableWidget {
                border: none;
                background-color: white;
                border-top: 1px solid black;
                border-left: 1px solid black;
                border-right: 1px solid black;
                border-bottom: 0;  

            }
            QTableWidget::item {
                border: 1px solid #dcdcdc;
                border-radius: 4px;
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #e6f3ff;
                border: 1px solid #99d1ff;
            }
        """)
        
        # 열 너비 설정
        header = self.image_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # 열 너비 자동 조정

    def create_table_item(self, file_path):
        """테이블 항목 생성"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 체크박스 추가
        checkbox = QCheckBox()
        checkbox.setStyleSheet("QCheckBox { margin-right: 10px; }")
        layout.addWidget(checkbox)
        
        # 썸네일 레이블
        thumbnail_label = QLabel()
        pixmap = self.create_thumbnail(file_path)
        thumbnail_label.setPixmap(pixmap)
        thumbnail_label.setFixedSize(50, 50)
        layout.addWidget(thumbnail_label)
        
        # 파일 이름 레이블
        file_name_label = QLabel(os.path.basename(file_path))
        file_name_label.setStyleSheet("""
            QLabel {
                margin-left: 10px;
                padding-right: 20px;  /* 오른쪽 여백 추가 */
            }
        """)
        file_name_label.setMinimumWidth(200)  # 최소 너비 설정
        layout.addWidget(file_name_label)
        
        # 남은 공간을 채우는 스페이서 추가
        layout.addStretch()
        
        # 위젯 속성 설정
        widget.setProperty("file_path", file_path)
        
        return widget

    def refresh_table(self):
        """테이블 새로고침"""
        try:
            # 처리 완료된 항목 제거
            rows_to_remove = []
            for row in range(self.image_table.rowCount()):
                widget = self.image_table.cellWidget(row, 0)
                if widget:
                    file_path = widget.property("file_path")
                    # ImageProcessor에서 처리 완료된 파일인지 확인
                    if file_path in self.image_processor.processed_files:
                        rows_to_remove.append(row)
            
            # 역순으로 정렬하여 삭제
            for row in sorted(rows_to_remove, reverse=True):
                self.image_table.removeRow(row)
            
            # 버튼 상태 업데이트
            self.update_button_states()
            
        except Exception as e:
            print(f"Error refreshing table: {str(e)}")