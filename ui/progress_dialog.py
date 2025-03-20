import sys
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QHBoxLayout, QTextEdit, QApplication, QFrame, QPlainTextEdit
from PyQt5.QtCore import Qt, pyqtSlot, QTimer, QMetaType, QDateTime
from PyQt5.QtGui import QFont, QTextCursor

# QTextCursor 메타타입 등록 시도
try:
    QMetaType.registerType("QTextCursor", QTextCursor.__hash__)
    print("QTextCursor 메타타입 등록 성공")
except Exception as e:
    print(f"QTextCursor 메타타입 등록 실패: {e}")

class ProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("처리 진행 상황")
        self.resize(600, 400)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # 현재 처리 중인 파일 레이블
        self.current_file_label = QLabel("처리 중인 파일: ")
        main_layout.addWidget(self.current_file_label)
        
        # 로그 영역
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)
        
        # 진행 바 영역
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("0%")
        progress_layout.addWidget(self.progress_label)
        main_layout.addLayout(progress_layout)
        
        # 버튼 영역
        button_layout = QHBoxLayout()
        
        # 일시정지/재개 버튼
        self.pause_button = QPushButton("일시정지")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.paused = False
        button_layout.addWidget(self.pause_button)
        
        # 취소 버튼
        self.cancel_button = QPushButton("취소")
        button_layout.addWidget(self.cancel_button)
        
        main_layout.addLayout(button_layout)
        
        # 닫기 버튼 비활성화 (작업 중에는 닫을 수 없음)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        
        # 초기 상태 설정
        self.add_log("처리 준비 중...")
        
        # 상태 변수
        self.processed_count = 0
        self.total_count = 0
        
        # 작업자 스레드 참조
        self.worker_thread = None
        
    def set_worker_thread(self, worker):
        """작업자 스레드 설정 및 시그널 연결"""
        self.worker_thread = worker
        
        # 일시 정지/재개 버튼 연결
        if self.worker_thread:
            self.pause_button.clicked.disconnect(self.toggle_pause)
            self.pause_button.clicked.connect(self.toggle_worker_pause)
    
    def toggle_worker_pause(self):
        """작업자 스레드 일시 정지/재개 토글"""
        if not self.worker_thread:
            return
            
        self.paused = not self.paused
        button_text = "재개" if self.paused else "일시 정지"
        self.pause_button.setText(button_text)
        
        if self.paused:
            self.worker_thread.pause()
        else:
            self.worker_thread.resume()
        
    def toggle_pause(self):
        """일시정지/재개 버튼 토글"""
        self.paused = not self.paused
        button_text = "재개" if self.paused else "일시정지"
        self.pause_button.setText(button_text)
    
    def add_log(self, message):
        """로그 메시지 추가"""
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        log_message = f"[{timestamp}] {message}"
        
        # QTimer를 사용하여 메인 스레드에서 실행
        QTimer.singleShot(0, lambda: self._append_log_safe(log_message))
    
    def _append_log_safe(self, message):
        """메인 스레드에서 안전하게 로그 추가"""
        self.log_text.appendPlainText(message)
        # 스크롤을 항상 맨 아래로 유지
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    @pyqtSlot(int, int)
    def update_progress(self, current, total):
        """진행 상태 업데이트 (스레드 안전)"""
        # QTimer를 사용하여 메인 스레드에서 실행
        QTimer.singleShot(0, lambda: self._update_progress_safe(current, total))
    
    def _update_progress_safe(self, current, total):
        """메인 스레드에서 안전하게 진행 상태 업데이트"""
        if total > 0:
            percent = int(current * 100 / total)
            self.progress_bar.setValue(percent)
            self.progress_label.setText(f"{percent}% ({current}/{total})")
    
    @pyqtSlot(str)
    def update_current_file(self, file_path):
        """현재 처리 중인 파일 업데이트 (스레드 안전)"""
        # QTimer를 사용하여 메인 스레드에서 실행
        QTimer.singleShot(0, lambda: self._update_current_file_safe(file_path))
    
    def _update_current_file_safe(self, file_path):
        """메인 스레드에서 안전하게 현재 파일 업데이트"""
        self.current_file_label.setText(f"처리 중인 파일: {file_path}")
    
    def enable_close_button(self):
        """작업 완료 후 닫기 버튼 활성화"""
        # QTimer를 사용하여 메인 스레드에서 실행
        QTimer.singleShot(0, self._enable_close_button_safe)
    
    def _enable_close_button_safe(self):
        """메인 스레드에서 안전하게 닫기 버튼 활성화"""
        self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)
        self.show()  # 윈도우 플래그가 변경되었으므로 다시 표시
        
        # 일시정지 버튼 비활성화
        self.pause_button.setEnabled(False)
        self.cancel_button.setText("닫기")
        
        # 취소 버튼의 기능을 닫기로 변경
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.close)

    def closeEvent(self, event):
        """다이얼로그가 닫힐 때 호출되는 이벤트 핸들러"""
        try:
            # 직접 worker_thread에 접근하지 않고 취소 버튼 클릭으로 처리
            if hasattr(self, 'cancel_button'):
                self.cancel_button.click()
                
            # 다이얼로그 종료 허용
            event.accept()
        except Exception as e:
            print(f"다이얼로그 닫기 중 오류: {e}")
            event.accept()  # 오류가 발생해도 닫기 허용
            
    def update_progress_value(self, value, maximum=None):
        """진행률 업데이트"""
        def _update_progress():
            try:
                if maximum is not None:
                    self.progress_bar.setMaximum(maximum)
                self.progress_bar.setValue(value)
                
                # 진행률 계산 및 표시
                percent = int(value / self.progress_bar.maximum() * 100) if self.progress_bar.maximum() > 0 else 0
                self.progress_bar.setFormat(f"{value}/{self.progress_bar.maximum()} ({percent}%)")
            except Exception as e:
                print(f"진행률 업데이트 중 오류: {e}")
        
        QTimer.singleShot(0, _update_progress) 