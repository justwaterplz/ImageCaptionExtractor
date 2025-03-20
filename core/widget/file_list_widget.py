# 1단계. 메인 화면에 출력되는 파일 리스트, 버튼 등을 구현한 위젯.
import os.path
import subprocess
import sys

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QHeaderView, QCheckBox, QPushButton, \
    QHBoxLayout, QLabel, QMenu, QAction, QMessageBox, QAbstractItemView, \
    QTableWidgetSelectionRange
from PyQt5.QtGui import QIcon, QPixmap, QColor, QImage, QCursor
from PyQt5.QtCore import Qt, QFileInfo, QSize
from PyQt5.QtWidgets import QFileIconProvider, QSizePolicy
from PyQt5.QtCore import pyqtSignal
from cfg.cfg import *

class FileListWidget(QWidget):
    file_clicked = pyqtSignal(str, str)
    files_selected = pyqtSignal(list)

    def __init__(self, parent=None):
        super(FileListWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.image_table = QTableWidget(self)
        self.image_table.setColumnCount(1)
        self.image_table.setHorizontalHeaderLabels(['이미지'])
        self.image_table.verticalHeader().setDefaultSectionSize(60)
        self.image_table.setShowGrid(False)
        self.image_table.setStyleSheet("""  
                    QTableWidget {
                        background-color: white;
                        selection-background-color: #d0d0d0;
                        border: 2px solid black;  

                    }
                    QHeaderView::section {
                        background-color: #f0f0f0;
                        padding: 1px;
                        border: 1px solid #d0d0d0;
                        border-bottom: 2px solid #a0a0a0;  /* 헤더 아래 경계선 */
                    }
                    QTableWidget::item {
                        border-bottom: 1px solid #d0d0d0;
                    }
                    QTableWidget::item:selected {
                        background-color: #14cee3
                    }
                    QTableWidget::item:selected:!active {
                        background-color: #14cee3;
                        color: black;
                    }
                """)

        self.image_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.image_table.verticalHeader().setVisible(False)
        self.image_table.cellClicked.connect(self.on_item_clicked)
        self.image_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.image_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.layout.addWidget(self.image_table)

        # scroll area 추가하기

        #전체 선택 해제(초기화)
        self.all_selected = False

        # 지원되는 확장자 정의하기
        self.allowed_formats = {'.jpg', '.jpeg', '.png', '.bmp'}

        # icon 폴더 경로 저장
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.icon_folder = resource_path(os.path.join("res", "img"))

        self.thumbnail_size = QSize(100, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.image_table.cellDoubleClicked.connect(self.open_image_doubleclick)

        self.image_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_table.customContextMenuRequested.connect(self.show_context_menu)

        self.image_table.setSelectionMode(QAbstractItemView.MultiSelection)

    def add_file_to_list(self, file_paths):
        if isinstance(file_paths, str):
            file_paths = [file_paths]  # 단일 파일 경로를 리스트로 변환

        if self.image_table.columnCount() == 0:
            self.image_table.setColumnCount(1)  # 또는 필요한 열 수
            self.image_table.horizontalHeader().setVisible(True)
            self.image_table.verticalHeader().setVisible(True)

        for file_path in file_paths:
            if not self.is_allowed_format(file_path):
                print(f"지원되지 않는 파일 형식입니다: {file_path}")
                continue

            row_position = self.image_table.rowCount()
            self.image_table.insertRow(row_position)
            file_info = QFileInfo(file_path)
            custom_icon = self.get_custom_icon(file_path)

            try:
                widget = self.create_item_widget(file_info, custom_icon, file_path)
                self.image_table.setCellWidget(row_position, 0, widget)
            except Exception as e:
                print(f"파일 추가 중 오류 발생: {str(e)}")
                self.image_table.removeRow(row_position)
            else:
                # 파일 추가가 성공한 경우에만 file_list에 추가
                if hasattr(self, 'file_list'):
                    self.file_list.append(file_path)

        self.image_table.viewport().update()

    def create_thumbnail(self, file_path):
        image = QImage(file_path)
        if image.isNull():
            raise Exception("이미지를 불러올 수 없습니다.")

        thumbnail = image.scaled(self.thumbnail_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QPixmap.fromImage(thumbnail)

    def create_item_widget(self, file_info, custom_icon, file_path):
        try:
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(5, 0, 5, 0)
            layout.setSpacing(10)

            checkbox = QCheckBox()
            checkbox.stateChanged.connect(
                lambda state: self.on_checkbox_changed(state, self.image_table.indexAt(widget.pos()).row()))
            layout.addWidget(checkbox)

            icon_label = QLabel()
            try:
                # 썸네일 생성 시도
                thumbnail = self.create_thumbnail(file_path)
                icon_label.setPixmap(thumbnail)
            except Exception as e:
                print(f"썸네일 생성 실패: {str(e)}")
                # 썸네일 생성 실패 시 기존 아이콘 사용
                if custom_icon and not custom_icon.isNull():
                    icon_label.setPixmap(custom_icon.pixmap(24, 24))
                else:
                    default_icon = QFileIconProvider().icon(file_info)
                    icon_label.setPixmap(default_icon.pixmap(24, 24))
            layout.addWidget(icon_label)

            file_name_label = QLabel(file_info.fileName())
            file_name_label.setStyleSheet("padding-left: 5px;")
            layout.addWidget(file_name_label)

            widget.setProperty("file_path", file_path)

            layout.addStretch()
            widget.setLayout(layout)
            return widget
        except Exception as e:
            print(f"Error in create_item_widget: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def on_item_clicked(self, row, column):
        widget = self.image_table.cellWidget(row, column)
        if widget:
            checkbox = widget.layout().itemAt(0).widget()
            if isinstance(checkbox, QCheckBox):
                checkbox.setChecked(not checkbox.isChecked())

            file_path = widget.property("file_path")
            if file_path:
                self.file_clicked.emit(file_path, "preview")

    def create_checkbox_from_item(self, item):
        checkbox = QCheckBox()
        checkbox.setChecked(item.checkState() == Qt.Checked)
        checkbox.stateChanged.connect(
            lambda state, row=self.image_table.rowCount(): self.on_checkbox_changed(state, row))
        return checkbox

    def on_checkbox_changed(self, state, row):
        self.image_table.setRangeSelected(QTableWidgetSelectionRange(row, 0, row, 0), state == Qt.Checked)
        color = QColor('#d0d0d0') if state == Qt.Checked else QColor('white')
        self.update_row_color(row, color)
        self.files_selected.emit(self.get_selected_files())  # 선택된 파일 목록 전송

    def update_row_color(self, row, color):
        self.image_table.cellWidget(row, 0).setStyleSheet(f"background-color: {color.name()};")

    def get_selected_files(self):
        selected_files = []
        for row in range(self.image_table.rowCount()):
            widget = self.image_table.cellWidget(row, 0)
            if widget:
                checkbox = widget.layout().itemAt(0).widget()
                if checkbox.isChecked():
                    file_path = widget.property("file_path")
                    selected_files.append(file_path)
        return selected_files

    # 파일 확장자에 따라 res/img에 있는 아이콘 이미지 사용
    def get_custom_icon(self, file_path):
        _, file_format = os.path.splitext(file_path)
        file_format = file_format.lower()

        # 현재 지원되는 파일 확장자: jpg, jpeg, png, bmp
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
                else:
                    print(f"아이콘 불러오기 실패: {icon_path}")
            else:
                print(f"아이콘 파일 찾기 실패: {icon_path}")
        else:
            print(f"지원되지 않는 형식입니다: {file_format}")

        return QIcon()
    def is_allowed_format(self, file_path):
        _, file_format = os.path.splitext(file_path)
        return file_format.lower() in self.allowed_formats

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

    # 항목 삭제를 위한 컨텍스트 메뉴 구현
    def show_context_menu(self):
        context_menu = QMenu(self)
        delete_action = QAction("삭제", self)
        delete_action.triggered.connect(self.delete_selected_items)
        context_menu.addAction(delete_action)

        delete_action.setEnabled(len(self.image_table.selectedRanges()) > 0)
        context_menu.exec_(QCursor.pos())

    def delete_selected_items(self):
        selected_rows = set(index.row() for index in self.image_table.selectedIndexes())
        if not selected_rows:
            QMessageBox.information(self, "알림", "삭제할 항목을 선택해주세요.")
            return

        total_rows = len(selected_rows)
        reply = QMessageBox.question(self, "삭제 확인",
                                     f"선택한 {total_rows}개의 항목을 삭제하시겠습니까?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        # 역순으로 정렬하여 삭제 (인덱스 변화 방지)
        for row in sorted(selected_rows, reverse=True):
            self.image_table.removeRow(row)

        self.image_table.clearSelection()
        QMessageBox.information(self, "삭제 완료", f"{total_rows}개의 항목이 삭제되었습니다.")
        print(f"삭제 후 테이블 총 행 수: {self.image_table.rowCount()}")

    def clear(self):
        self.image_table.setRowCount(0)

        # 선택 모델 초기화
        self.image_table.clearSelection()

        # 정렬 초기화 (필요한 경우)
        self.image_table.setSortingEnabled(False)
        self.image_table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)

        # 스크롤바 위치 초기화
        self.image_table.verticalScrollBar().setValue(0)
        self.image_table.horizontalScrollBar().setValue(0)

        # 내부 데이터 구조 초기화 (만약 있다면)
        if hasattr(self, 'file_list'):
            self.file_list.clear()

        # 테이블 새로고침
        self.image_table.viewport().update()

