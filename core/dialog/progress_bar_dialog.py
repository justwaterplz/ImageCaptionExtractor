from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                             QProgressBar, QTextEdit, QPushButton, QLabel)
from PyQt5.QtCore import Qt


class ProgressBarDialog(QDialog):
    def __init__(self, total_images, parent=None):
        super().__init__(parent)
        self.total_images = total_images
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("이미지 처리 진행 상황")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        layout = QVBoxLayout()

        # 진행 상황 표시 영역
        progress_layout = QVBoxLayout()
        self.progress_label = QLabel("진행 상황:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)

        # 현재 처리 중인 파일 표시
        self.current_file_label = QLabel("처리 중인 파일: ")
        progress_layout.addWidget(self.current_file_label)

        layout.addLayout(progress_layout)

        # 로그 표시 영역
        log_layout = QVBoxLayout()
        log_label = QLabel("처리 로그:")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log_text)

        layout.addLayout(log_layout)

        # 버튼 영역
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("취소")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def update_progress(self, processed, total):
        """진행률 업데이트"""
        if total > 0:
            percentage = int((processed / total) * 100)
            self.progress_bar.setValue(percentage)
            self.progress_label.setText(f"진행 상황: {processed}/{total} ({percentage}%)")
            
            # 모든 처리가 완료되면 UI 업데이트
            if processed == total:
                self.progress_label.setText(f"처리 완료! ({total}개 이미지 처리됨)")
                self.cancel_button.setText("닫기")  # "취소" 대신 "닫기"로 변경
                self.cancel_button.setStyleSheet("background-color: #4CAF50; color: white;")  # 초록색 배경으로 변경
                
                try:
                    self.cancel_button.clicked.disconnect()  # 기존 연결 해제
                except TypeError:
                    pass  # 연결이 없는 경우 무시
                
                self.cancel_button.clicked.connect(self.accept)  # accept로 변경
                
                # 완료 메시지를 로그에 추가
                self.add_log("\n모든 이미지 처리가 완료되었습니다!")

    def update_current_file(self, file_path):
        """현재 처리 중인 파일 정보 업데이트"""
        if file_path:
            self.current_file_label.setText(f"처리 중인 파일: {file_path}")
            self.add_log(f"\n새로운 파일 처리 시작: {file_path}")
        else:
            self.current_file_label.setText("처리 완료")

    def add_log(self, message):
        """로그 메시지 추가"""
        self.log_text.append(message)
        # 스크롤을 항상 맨 아래로
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def clear_log(self):
        """로그 초기화"""
        self.log_text.clear()