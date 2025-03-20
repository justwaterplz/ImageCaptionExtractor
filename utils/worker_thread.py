import os
import sys
import time
import json
import traceback
import re
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal, QSettings
from PyQt5.QtWidgets import QFileDialog
from openai import OpenAI
from openai.types.beta import Assistant
from openai.types.beta.threads import Run
from queue import Queue, Empty
from PyQt5.QtWidgets import QApplication

class WorkerThread(QThread):
    progress = pyqtSignal(int, int)  # 처리된 이미지 수, 전체 이미지 수
    progress_signal = pyqtSignal(int, int)  # 처리된 이미지 수, 전체 이미지 수 (별칭)
    result = pyqtSignal(str, str)  # file_path, response
    error = pyqtSignal(str)
    finished = pyqtSignal(list)  # [(file_path, response), ...]
    current_file = pyqtSignal(str)  # 현재 처리 중인 파일 경로
    remove_from_table = pyqtSignal(str)  # 테이블에서 제거할 파일 경로
    result_signal = pyqtSignal(str, dict)
    error_signal = pyqtSignal(str, str)
    status_signal = pyqtSignal(str)
    increment_progress_signal = pyqtSignal()
    completed_signal = pyqtSignal(str)

    def __init__(self, queue=None, settings_handler=None, image_processor=None, image_paths=None, api_key=None):
        super().__init__()
        self.queue = queue if queue is not None else Queue()
        self.image_queue = self.queue  # 호환성을 위한 별칭
        self.settings_handler = settings_handler
        self.image_processor = image_processor
        self.stopped = False
        self.is_paused = False  # 일시 정지 상태 추가
        self.client = None      # 초기화는 load_api_settings() 또는 run() 메서드에서 수행
        self.thread = None
        self.run_id = None
        self.tool_calls = None
        self.api_key = api_key  # API 키 직접 설정
        self.assistant_id = None
        self.jsonl_file_path = None  # JSONL 파일 경로 추가
        self.moveToThread(self)  # 자신의 스레드로 이동
        
        # 마지막 저장 위치 가져오기
        self.last_save_directory = os.path.expanduser('~')  # 기본값
        if self.settings_handler:
            save_dir = self.settings_handler.get_setting('last_save_directory')
            if save_dir and os.path.exists(save_dir):
                self.last_save_directory = save_dir
                print(f"마지막 저장 위치 설정: {self.last_save_directory}")
        
        # image_paths가 있으면 큐에 추가
        if image_paths:
            for image_path in image_paths:
                self.queue.put(image_path)
        
        # API 키와 Assistant ID 로드
        self.load_api_settings()
        
        # 필수 객체 확인 로직 추가
        if self.settings_handler is None:
            print("경고: settings_handler가 초기화되지 않았습니다.")
        
        if self.image_processor is None:
            print("경고: image_processor가 초기화되지 않았습니다.")
        
        print("\n=== WorkerThread Initialization ===")
        print(f"API Key exists: {bool(self.api_key)}")
        
        # 클라이언트 헤더 확인
        if self.client:
            beta_header = self.client._custom_headers.get("OpenAI-Beta") if hasattr(self.client, "_custom_headers") else None
            print(f"OpenAI client initialized with beta header: {beta_header}")
        else:
            print("OpenAI client not initialized yet")
        
        # 응답 저장을 위한 리스트 초기화
        self.responses = []
        self.is_running = True
        
        # config.json 확인
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                print("\n=== Config File Contents ===")
                print(f"Config keys: {list(config.keys())}")
                print(f"Assistant ID in config: {config.get('assistant_id')}")
        except Exception as e:
            print(f"Error reading config.json: {e}")
        
        # assistant_id 가져오기 또는 새로 생성
        self.assistant_id = self.get_assistant_id()
        print(f"Using Assistant ID: {self.assistant_id}")
        
        # JSONL 파일 초기화 - 아직 초기화하지 않음 (사용자 경로 선택 후 초기화)

    def load_api_settings(self):
        """API 키와 Assistant ID 설정 로드"""
        try:
            # 이미 API 키가 설정되어 있지 않은 경우에만 settings_handler에서 로드
            if not self.api_key and self.settings_handler:
                self.api_key = self.settings_handler.get_setting('openai_key', '')
            
            # settings_handler가 있을 경우에만 assistant_id 로드
            if self.settings_handler:
                self.assistant_id = self.settings_handler.get_setting('assistant_id', '')
            
            if not self.api_key:
                self.error_signal.emit("설정 오류", "API 키가 설정되지 않았습니다.")
                return False
            
            # OpenAI 클라이언트 초기화 - v2 beta 헤더 명시
            self.client = OpenAI(
                api_key=self.api_key,
                default_headers={"OpenAI-Beta": "assistants=v2"}  # v2 API 사용
            )
            print("OpenAI 클라이언트 초기화 완료 (assistants=v2)")
            return True
        except Exception as e:
            self.error_signal.emit("API 설정 오류", str(e))
            return False

    def initialize_thread(self):
        """OpenAI API 스레드 초기화"""
        try:
            # 새 클라이언트 인스턴스 생성
            client = OpenAI(
                api_key=self.api_key,
                default_headers={"OpenAI-Beta": "assistants=v2"}
            )
            
            # 스레드 생성
            self.thread = client.beta.threads.create()
            print(f"Thread created with ID: {self.thread.id}")
            return True
        except Exception as e:
            self.error_signal.emit("OpenAI 스레드 초기화 오류", str(e))
            print(f"OpenAI 스레드 초기화 오류: {e}")
            return False

    def get_vector_store_id(self):
        """Retrieve an existing vector store or create one if none exists."""
        try:
            vector_stores = self.client.beta.vector_stores.list()
            if vector_stores.data and len(vector_stores.data) > 0:
                vs_id = vector_stores.data[0].id
                print(f"Using existing vector store: {vs_id}")
                return vs_id
            else:
                vs = self.client.beta.vector_stores.create(name="Financial Statements")
                print(f"Created new vector store with ID: {vs.id}")
                return vs.id
        except Exception as e:
            print(f"Error listing vector stores: {e}")
            vs = self.client.beta.vector_stores.create(name="Financial Statements")
            print(f"Created new vector store with ID: {vs.id}")
            return vs.id

    def get_assistant_id(self):
        """config.json에서 assistant_id 가져오기"""
        try:
            # config.json에서 assistant_id 읽기
            config_path = os.path.join(os.path.expanduser("~"), ".imagekeywordextractor", "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # config에 assistant_id가 있으면 사용
                    if 'assistant_id' in config and config['assistant_id'].strip():
                        assistant_id = config['assistant_id']
                        print(f"Using assistant_id from config: {assistant_id}")
                        return assistant_id
            
            # 설정 파일에 assistant_id가 없으면 빈 문자열 반환
            print("No valid assistant_id in config")
            return ""
                
        except Exception as e:
            print(f"Error in get_assistant_id: {e}")
            # 오류 발생 시 빈 문자열 반환
            return ""

    def upload_image(self, image_path):
        """이미지 파일 업로드"""
        try:
            # v2 API 클라이언트 생성
            upload_client = OpenAI(
                api_key=self.api_key,
                default_headers={"OpenAI-Beta": "assistants=v2"}
            )
            
            with open(image_path, "rb") as file:
                response = upload_client.files.create(
                    file=file,
                    purpose="assistants"
                )
                return response.id
        except Exception as e:
            print(f"Error uploading image: {e}")
            raise

    def initialize_jsonl_file(self):
        """JSONL 파일 초기화 - 사용자가 파일 위치 선택 가능"""
        try:
            # 타임스탬프를 이용한 기본 파일명 생성
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_filename = f"captions_{timestamp}.jsonl"
            
            # 파일 저장 다이얼로그 표시 (PyQt5 필요)
            from PyQt5.QtWidgets import QFileDialog, QApplication
            
            # 저장 다이얼로그 표시
            file_path, _ = QFileDialog.getSaveFileName(
                None,  # 부모 위젯 없음
                "JSONL 파일 저장 위치 선택",
                os.path.join(self.last_save_directory, default_filename),
                "JSONL Files (*.jsonl);;All Files (*)"
            )
            
            # 사용자가 취소한 경우
            if not file_path:
                print("사용자가 파일 저장을 취소했습니다. 기본 위치에 저장합니다.")
                self.jsonl_file_path = os.path.join(self.last_save_directory, default_filename)
            else:
                # 확장자 확인 및 추가
                if not file_path.lower().endswith('.jsonl'):
                    file_path += '.jsonl'
                    print(f"확장자 .jsonl 추가: {file_path}")
                
                self.jsonl_file_path = file_path
                
                # 선택된 디렉토리 저장
                save_directory = os.path.dirname(file_path)
                if self.settings_handler:
                    self.settings_handler.save_setting('last_save_directory', save_directory)
                    self.last_save_directory = save_directory
                    print(f"저장 위치 업데이트: {save_directory}")
            
            # 빈 JSONL 파일 생성
            with open(self.jsonl_file_path, 'w', encoding='utf-8') as f:
                pass  # 빈 파일 생성
            
            # 상태 메시지 표시
            print(f"JSONL 파일 초기화 완료: {self.jsonl_file_path}")
            self.status_signal.emit(f"JSONL 파일 생성 완료: {self.jsonl_file_path}")
            self.status_signal.emit("API 설정 로드 중...")
            
            return True
        except Exception as e:
            print(f"JSONL 파일 초기화 오류: {e}")
            traceback.print_exc()
            self.error_signal.emit("파일 오류", f"JSONL 파일 초기화 실패: {e}")
            
            # 오류 발생 시 기본 경로에 저장
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                default_filename = f"captions_{timestamp}.jsonl"
                self.jsonl_file_path = os.path.join(os.path.expanduser('~'), default_filename)
                with open(self.jsonl_file_path, 'w', encoding='utf-8') as f:
                    pass
                print(f"오류 발생으로 기본 위치에 파일 생성: {self.jsonl_file_path}")
                return True
            except:
                return False

    def request_extract_keyword(self, image_path):
        """단일 이미지에 대한 캡션 추출 요청"""
        max_retries = 3
        retry_delay = 2
        
        file_name = os.path.basename(image_path)
        self.status_signal.emit(f"처리 시작: {file_name}")
        print(f"\n=== Processing Image: {file_name} ===")
        
        for attempt in range(max_retries):
            image_file = None
            thread = None
            file = None

            try:
                # 이미지 파일 크기 확인
                file_size = os.path.getsize(image_path)
                print(f"File size: {file_size} bytes")
                size_mb = file_size / (1024*1024)
                self.status_signal.emit(f"{file_name} - 파일 크기: {size_mb:.2f} MB")
                
                # 이미지 파일이 너무 크면 경고 로그 추가
                if file_size > 20 * 1024 * 1024:  # 20MB 이상
                    print(f"경고: 이미지 파일이 매우 큽니다 ({size_mb:.2f} MB). 처리 시간이 오래 걸릴 수 있습니다.")
                    self.status_signal.emit(f"경고: {file_name}의 크기가 매우 큽니다. 처리 시간이 오래 걸릴 수 있습니다.")
                
                # 이미지 파일 준비
                self.status_signal.emit(f"{file_name} - 파일 업로드 준비 중...")
                image_file = open(image_path, "rb")
                
                # 파일 업로드 (간소화된 버전)
                self.status_signal.emit(f"{file_name} - OpenAI 서버에 업로드 중...")
                try:
                    # 새 클라이언트 인스턴스 생성
                    client = OpenAI(
                        api_key=self.api_key,
                        default_headers={"OpenAI-Beta": "assistants=v2"}
                    )
                    
                    file = client.files.create(
                        file=image_file,
                        purpose="assistants"
                    )
                    print(f"File uploaded with ID: {file.id}")
                    self.status_signal.emit(f"{file_name} - 업로드 성공 (ID: {file.id})")
                except Exception as upload_error:
                    self.status_signal.emit(f"{file_name} - 업로드 실패: {str(upload_error)}")
                    raise
                
                if not file:
                    self.status_signal.emit(f"{file_name} - 파일 업로드 실패")
                    raise Exception("파일 업로드 실패")

                # 스레드 생성 - messages 매개변수를 사용하여 처음부터 메시지 포함
                self.status_signal.emit(f"{file_name} - OpenAI 스레드 및 메시지 생성 중...")
                
                # 분석 요청 텍스트 - 최소화된 버전
                analysis_prompt = "이미지를 분석하여 상세한 설명을 작성해주세요."

                try:
                    # 스레드 생성
                    thread = client.beta.threads.create()
                    print(f"Thread created with ID: {thread.id}")
                    
                    # 메시지 생성 - 이미지 파일 포함
                    message = client.beta.threads.messages.create(
                        thread_id=thread.id,
                        role="user",
                        content=[
                            {
                                "type": "text",
                                "text": analysis_prompt
                            },
                            {
                                "type": "image_file",
                                "image_file": {
                                    "file_id": file.id,
                                    "detail": "high"
                                }
                            }
                        ]
                    )
                    
                    print(f"Message created with ID: {message.id}")
                    self.status_signal.emit(f"{file_name} - 스레드 및 메시지 생성됨 (ID: {thread.id})")
                
                except Exception as thread_error:
                    print(f"스레드 생성 오류: {thread_error}")
                    self.status_signal.emit(f"{file_name} - 스레드 생성 실패: {str(thread_error)}")
                    raise thread_error

                # 실행 생성
                self.status_signal.emit(f"{file_name} - 분석 작업 실행 중... (Assistant ID: {self.assistant_id[:8]}...)")
                try:
                    run = client.beta.threads.runs.create(
                        thread_id=thread.id,
                        assistant_id=self.assistant_id
                    )
                    print(f"Run created with ID: {run.id}")
                    
                    # 취소 요청에서 사용할 수 있도록 현재 thread와 run_id 저장
                    self.thread = thread
                    self.run_id = run.id
                    
                    self.status_signal.emit(f"{file_name} - 작업 시작됨 (Run ID: {run.id})")
                except Exception as run_error:
                    print(f"실행 생성 오류: {run_error}")
                    self.status_signal.emit(f"{file_name} - 실행 생성 실패: {str(run_error)}")
                    raise run_error

                # 응답 대기 (wait_for_completion 메서드 호출 방식 유지)
                self.status_signal.emit(f"{file_name} - 분석 결과 대기 중...")
                response_json = self.wait_for_completion(thread.id, run.id, image_path)
                
                # JSON 응답 처리
                if response_json:
                    try:
                        self.status_signal.emit(f"{file_name} - 응답 데이터 처리 중...")
                        print(f"API 응답 데이터: {response_json}")
                        
                        # 응답 검증 및 결과 포맷팅
                        if isinstance(response_json, dict):
                            result = response_json
                        else:
                            try:
                                result = json.loads(response_json)
                            except:
                                self.status_signal.emit(f"{file_name} - 유효하지 않은 JSON 응답")
                                return None

                        # API 응답에서 필요한 필드만 추출
                        try:
                            # API 응답에서 text 부분만 가져오기
                            if not result.get("text") or \
                               not result["text"].get("english_caption") or \
                               not result["text"].get("korean_caption"):
                                self.status_signal.emit(f"{file_name} - 필수 필드가 누락됨")
                                print(f"Missing required fields in response: {result}")
                                return None

                            # 캡션 내용 가져오기
                            text_content = {
                                "english_caption": result["text"]["english_caption"].strip(),
                                "korean_caption": result["text"]["korean_caption"].strip()
                            }

                            # 문장 수 검증 함수
                            def count_sentences(text):
                                # 영어 문장 구분: .!? 뒤에 공백이나 문장 끝
                                if any(ord(c) < 128 for c in text):  # 영어 텍스트
                                    sentences = [s.strip() for s in re.split(r'[.!?](?=\s|$)', text) if s.strip()]
                                # 한글 문장 구분: .!?。！？ 뒤에 공백이나 문장 끝
                                else:  # 한글 텍스트
                                    sentences = [s.strip() for s in re.split(r'[.!?。！？](?=\s|$)', text) if s.strip()]
                                return sentences

                            # 캡션 문장 수 검증 및 처리
                            eng_sentences = count_sentences(text_content["english_caption"])
                            kor_sentences = count_sentences(text_content["korean_caption"])

                            if len(eng_sentences) > 3:
                                text_content["english_caption"] = '. '.join(eng_sentences[:3]) + '.'
                                print(f"English caption truncated to 3 sentences")
                            
                            if len(kor_sentences) > 3:
                                text_content["korean_caption"] = '. '.join(kor_sentences[:3]) + '.'
                                print(f"Korean caption truncated to 3 sentences")

                            # 문장 수가 3개 미만인 경우 로그 출력
                            if len(eng_sentences) < 3 or len(kor_sentences) < 3:
                                print(f"Warning: Caption has fewer than 3 sentences. English: {len(eng_sentences)}, Korean: {len(kor_sentences)}")

                            # 최종 결과 객체 생성 (파일 정보 추가)
                            formatted_result = {
                                "content": os.path.basename(image_path),  # 파일명만 추출
                                "image_path": image_path.replace("\\", "/"),  # 경로 구분자 통일
                                "text": text_content
                            }

                            print(f"처리된 결과: {json.dumps(formatted_result, ensure_ascii=False, indent=2)}")
                            self.status_signal.emit(f"{file_name} - 처리 완료")
                            return formatted_result

                        except Exception as e:
                            self.status_signal.emit(f"{file_name} - 결과 처리 중 오류: {str(e)}")
                            print(f"Error processing result: {str(e)}")
                            return None
                    except json.JSONDecodeError as e:
                        print(f"JSON 파싱 오류: {e}")
                        print(f"원본 응답: {response_json}")
                        self.status_signal.emit(f"{file_name} - JSON 파싱 오류: {e}")
                        raise Exception(f"JSON 파싱 오류: {e}")
                
                self.status_signal.emit(f"{file_name} - 응답이 없거나 처리할 수 없는 형식입니다")
                return None

            except Exception as e:
                error_detail = str(e)
                print(f"Error in request: {error_detail}")
                self.status_signal.emit(f"{file_name} - 오류 발생: {error_detail}")
                
                if "timeout" in error_detail.lower() and attempt < max_retries - 1:
                    retry_msg = f"{file_name} - 타임아웃 오류 감지. 재시도 중... (Attempt {attempt + 1}/{max_retries})"
                    print(retry_msg)
                    self.status_signal.emit(retry_msg)
                    time.sleep(retry_delay * (attempt + 1))  # 지수 백오프 적용
                    continue
                elif attempt < max_retries - 1:
                    retry_msg = f"{file_name} - 오류 발생. 재시도 중... (Attempt {attempt + 1}/{max_retries})"
                    print(retry_msg)
                    self.status_signal.emit(retry_msg)
                    time.sleep(retry_delay)
                    continue
                
                self.status_signal.emit(f"{file_name} - 최대 재시도 횟수 초과. 처리 실패")
                raise

            finally:
                try:
                    if file:
                        self.status_signal.emit(f"{file_name} - 임시 파일 정리 중...")
                        # 동일한 클라이언트 인스턴스 사용
                        client.files.delete(file.id)
                        print(f"File {file.id} deleted")
                except Exception as cleanup_error:
                    print(f"Error deleting file: {cleanup_error}")
                    self.status_signal.emit(f"{file_name} - 임시 파일 정리 오류: {cleanup_error}")
                
                try:
                    if image_file:
                        image_file.close()
                        print("Image file closed")
                except Exception as close_error:
                    print(f"Error closing file: {close_error}")
                    self.status_signal.emit(f"{file_name} - 파일 닫기 오류: {close_error}")

    def append_to_jsonl(self, result):
        """결과를 JSONL 파일에 추가"""
        try:
            with open(self.jsonl_file_path, 'a', encoding='utf-8') as f:
                json_line = json.dumps(result, ensure_ascii=False)
                f.write(json_line + '\n')
            print(f"결과가 JSONL 파일에 추가됨: {os.path.basename(result.get('image_path', 'unknown'))}")
            return True
        except Exception as e:
            print(f"JSONL 파일 기록 오류: {e}")
            self.error_signal.emit("파일 오류", f"JSONL 파일 기록 실패: {e}")
            return False

    def wait_for_completion(self, thread_id, run_id, image_path=None):
        """실행이 완료될 때까지 대기하고 응답을 반환"""
        max_wait_time = 600  # 최대 대기 시간 (초) - 10분
        start_time = time.time()
        file_name = os.path.basename(image_path) if image_path else "Unknown"
        
        # 초기 상태 로그
        self.status_signal.emit(f"{file_name} - OpenAI API에 요청 전송됨. 처리 대기 중...")
        print(f"Waiting for completion of thread_id: {thread_id}, run_id: {run_id}")
        
        # 단일 클라이언트 인스턴스 생성
        client = OpenAI(
            api_key=self.api_key,
            default_headers={"OpenAI-Beta": "assistants=v2"}
        )
        
        while time.time() - start_time < max_wait_time:
            # 중단 여부 확인
            if self.stopped:
                self.status_signal.emit(f"{file_name} - 사용자에 의해 처리가 취소되었습니다.")
                return None
                
            try:
                # 실행 상태 확인
                run = client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run_id
                )
                
                status = run.status
                print(f"Run status: {status}")
                self.status_signal.emit(f"{file_name} - 처리 상태: {status}")
                
                if status == "completed":
                    self.status_signal.emit(f"{file_name} - 분석 완료! (소요 시간: {int(time.time() - start_time)}초)")
                    
                    # 메시지 검색
                    messages = client.beta.threads.messages.list(
                        thread_id=thread_id,
                        order="desc",
                        limit=1
                    )
                    
                    if messages.data:
                        message = messages.data[0]
                        print(f"Found message (ID: {message.id})")
                        
                        # 텍스트 추출
                        text_content = None
                        for content_item in message.content:
                            if content_item.type == "text":
                                text_content = content_item.text.value
                                break
                        
                        if text_content:
                            # JSON 추출 시도
                            return self.extract_json_from_text(text_content, file_name)
                    
                    self.status_signal.emit(f"{file_name} - 응답 메시지가 없습니다.")
                    return None
                
                elif status == "failed":
                    error_msg = f"실행 실패: {run.last_error.message if hasattr(run, 'last_error') else '알 수 없는 오류'}"
                    print(error_msg)
                    self.status_signal.emit(f"{file_name} - {error_msg}")
                    return None
                
                elif status == "cancelled":
                    self.status_signal.emit(f"{file_name} - 처리가 취소되었습니다.")
                    return None
                
                elif status == "expired":
                    self.status_signal.emit(f"{file_name} - 처리 시간이 만료되었습니다.")
                    return None
                
                # 아직 완료되지 않았으면 대기
                time.sleep(1)
                
            except Exception as e:
                error_detail = str(e)
                print(f"상태 확인 중 오류 발생: {error_detail}")
                self.status_signal.emit(f"{file_name} - 상태 확인 중 오류: {error_detail}")
                time.sleep(2)
        
        # 시간 초과
        self.status_signal.emit(f"{file_name} - 처리 시간 초과 (제한: {max_wait_time}초)")
        print(f"API 응답 대기 시간 초과: 경과 시간: {int(time.time() - start_time)}초")
        return None

    def run(self):
        """스레드 실행"""
        if not self.image_queue or self.image_queue.empty():
            self.emit_status_signal("오류: 처리할 이미지가 없습니다.")
            self.emit_status_signal("처리 완료")
            return
            
        try:
            # JSONL 파일 초기화
            self.initialize_jsonl_file()
            
            # OpenAI API 설정 로드
            self.emit_status_signal("OpenAI API 설정 로드 중...")
            
            # API 키 확인
            if not self.api_key:
                self.emit_status_signal("오류: API 키가 설정되지 않았습니다.")
                return
                
            # 어시스턴트 ID 확인
            if not self.assistant_id:
                self.emit_status_signal("오류: Assistant ID가 설정되지 않았습니다.")
                return
                
            # v2 API 클라이언트 보장
            if not self.ensure_v2_client():
                self.emit_status_signal("OpenAI 클라이언트 초기화 실패")
                return
                
            self.emit_status_signal("OpenAI 클라이언트 초기화 성공 (assistants=v2)")
                
            # 이미지 처리 시작
            total_images = self.image_queue.qsize()
            processed_count = 0
            
            self.emit_status_signal(f"이미지 처리 시작 (총 {total_images}개)...")
            
            while not self.image_queue.empty():
                # 취소 요청 확인
                if self.stopped:
                    # 결과 파일 경로 알림
                    msg = f"작업이 취소되었습니다. 결과 파일: {self.jsonl_file_path}"
                    self.emit_status_signal(msg)
                    self.completed_signal.emit(msg)
                    break
                    
                # 일시정지 요청 확인
                if self.is_paused:
                    self.emit_status_signal("처리가 일시정지되었습니다. 재개하려면 '재개' 버튼을 클릭하세요.")
                    while self.is_paused and not self.stopped:
                        # QThread.msleep 사용하여 스레드 안전하게 대기
                        QThread.msleep(500)
                    
                    if not self.is_paused:
                        self.emit_status_signal("처리가 재개되었습니다.")
                    
                    if self.stopped:
                        # 결과 파일 경로 알림
                        msg = f"작업이 취소되었습니다. 결과 파일: {self.jsonl_file_path}"
                        self.emit_status_signal(msg)
                        self.completed_signal.emit(msg)
                        break
                
                # 이미지 추출
                image_path = self.image_queue.get()
                file_name = os.path.basename(image_path)
                
                self.current_file.emit(file_name)
                self.emit_status_signal(f"처리 중: {file_name}")
                
                try:
                    # 이미지 처리 - thread.run, initialize_thread 없이 직접 요청
                    result = self.request_extract_keyword(image_path)
                    
                    if result and 'content' in result:
                        # 결과 저장
                        self.append_to_jsonl(result)
                        processed_count += 1
                        self.emit_status_signal(f"처리 완료: {file_name}")
                        self.result_signal.emit(image_path, result)
                    else:
                        self.emit_status_signal(f"처리 실패: {file_name} (결과 없음)")
                except Exception as e:
                    self.emit_status_signal(f"이미지 처리 오류: {file_name} - {str(e)}")
                
                # 진행 상황 업데이트
                self.progress.emit(processed_count, total_images)
            
            # 취소되지 않았을 경우 완료 메시지 표시
            if not self.stopped:
                # 결과 파일 경로 알림 (완료 시)
                msg = f"모든 이미지 처리가 완료되었습니다. 결과 파일: {self.jsonl_file_path}"
                self.emit_status_signal(msg)
                self.completed_signal.emit(msg)
        
        except Exception as e:
            error_msg = f"처리 오류: {str(e)}"
            self.emit_status_signal(error_msg)
            traceback.print_exc()

    def stop(self):
        """스레드 중지"""
        self.stopped = True
        
    def exit(self):
        """안전한 스레드 종료 - 모든 리소스 정리"""
        self.stop()
        try:
            # 진행 중인 API 작업 취소
            if hasattr(self, 'thread') and self.thread and hasattr(self, 'run_id') and self.run_id:
                try:
                    # v2 API 클라이언트 생성
                    exit_client = OpenAI(
                        api_key=self.api_key,
                        default_headers={"OpenAI-Beta": "assistants=v2"}
                    )
                    
                    # 실행중인 작업 취소
                    exit_client.beta.threads.runs.cancel(
                        thread_id=self.thread.id,
                        run_id=self.run_id
                    )
                except Exception as e:
                    print(f"API 작업 취소 중 오류: {e}")
                    
            # 큐 비우기
            while not self.image_queue.empty():
                try:
                    self.image_queue.get(block=False)
                except Empty:
                    break
                    
            # 스레드 종료
            self.quit()
            self.wait()
        except Exception as e:
            print(f"스레드 종료 중 오류: {e}")
            
    def __del__(self):
        """소멸자 - 객체 삭제 시 호출됨"""
        try:
            self.exit()  # 안전하게 종료
        except:
            pass

    def extract_field(self, text, field_name):
        """텍스트에서 특정 필드 값을 추출"""
        # 정규식 패턴 수정: 콜론 뒤 공백 및 여러 줄에 걸친 내용 고려
        field_pattern = rf"{field_name}:?\s*(.*?)(?:\n\w+:|$)"
        match = re.search(field_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            value = match.group(1).strip()
            # 여러 줄 값에서 불필요한 줄바꿈 제거
            value = re.sub(r'\n\s*', ' ', value)
            return value
        return ""

    def save_result(self, file_path, result):
        """결과를 파일에 저장"""
        try:
            # 확장자가 .xlsx가 아니면 강제로 변경
            if not file_path.lower().endswith('.xlsx'):
                file_path = os.path.splitext(file_path)[0] + '.xlsx'
                print(f"Changed file extension to .xlsx: {file_path}")
            
            # 결과를 파일에 저장
            with open(file_path, 'w') as f:
                json.dump(result, f)
            
            self.status_signal.emit(f"결과를 파일에 저장했습니다: {file_path}")
            print(f"결과를 파일에 저장했습니다: {file_path}")
        except Exception as e:
            self.status_signal.emit(f"결과 저장 중 오류: {str(e)}")
            print(f"결과 저장 중 오류: {str(e)}")

    def clean_text_value(self, text):
        """텍스트 값을 정리하는 메서드"""
        if not isinstance(text, str):
            return text
        
        # 앞뒤 공백 제거
        clean_text = text.strip()
        
        # 줄바꿈 및 과도한 공백 제거
        clean_text = ' '.join(clean_text.split())
        
        # "키워드:" 또는 "주요 색상:" 문자열로 시작하는 경우 제거
        clean_text = re.sub(r'^키워드:?\s*', '', clean_text)
        clean_text = re.sub(r'^주요\s*색상:?\s*', '', clean_text)
        
        return clean_text

    def pause(self):
        """작업 일시 정지"""
        self.is_paused = True
        self.status_signal.emit("작업이 일시 정지되었습니다.")
    
    def resume(self):
        """작업 재개"""
        self.is_paused = False
        self.status_signal.emit("작업이 재개되었습니다.")
    
    def cancel(self):
        """작업 취소"""
        self.stopped = True
        self.status_signal.emit("작업 취소 요청이 접수되었습니다. 처리 중인 요청을 중단 중...")
        
        # 현재 활성화된, 처리 중인 run이 있으면 취소 요청
        if hasattr(self, 'thread') and self.thread and hasattr(self, 'run_id') and self.run_id:
            try:
                self.status_signal.emit(f"OpenAI API 실행 취소 요청 중... (Run ID: {self.run_id})")
                # v2 API 클라이언트 생성
                cancel_client = OpenAI(
                    api_key=self.api_key,
                    default_headers={"OpenAI-Beta": "assistants=v2"}
                )
                
                # v2 API로 실행 취소
                cancel_client.beta.threads.runs.cancel(
                    thread_id=self.thread.id,
                    run_id=self.run_id
                )
                self.status_signal.emit("API 실행 취소 완료")
            except Exception as e:
                print(f"API 실행 취소 중 오류: {e}")
        
        # 작업 큐를 비워 추가 처리 방지
        while not self.image_queue.empty():
            try:
                self.image_queue.get(block=False)
            except Empty:
                break
                
        self.status_signal.emit("모든 작업이 취소되었습니다.")

    def add_image(self, image_path):
        """이미지 경로를 큐에 추가"""
        try:
            # 상대 경로를 절대 경로로 변환
            abs_path = os.path.abspath(image_path)
            
            # 파일 존재 여부 확인
            if not os.path.exists(abs_path):
                print(f"경고: 파일이 존재하지 않습니다: {abs_path}")
                self.error_signal.emit("파일 오류", f"파일이 존재하지 않습니다: {image_path}")
                return False
            
            self.image_queue.put(abs_path)
            return True
        except Exception as e:
            print(f"이미지 추가 오류: {str(e)}")
            return False

    def set_credentials(self, api_key, assistant_id):
        """API 키와 어시스턴트 ID 설정"""
        self.api_key = api_key
        self.assistant_id = assistant_id
        print(f"API 키와 어시스턴트 ID 설정 완료: {bool(api_key)}, {bool(assistant_id)}")
        return True

    def extract_json_from_text(self, text, file_name):
        """텍스트에서 JSON 데이터 추출"""
        if not text:
            return None
            
        # 디버깅을 위한 응답 내용 출력
        print(f"응답 텍스트: {text[:200]}..." if len(text) > 200 else text)
        self.status_signal.emit(f"{file_name} - 응답 데이터 수신 완료")
        
        try:
            # 1. 마크다운 코드 블록 내 JSON 추출 시도
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if json_match:
                json_str = json_match.group(1).strip()
                try:
                    json_obj = json.loads(json_str)
                    self.status_signal.emit(f"{file_name} - JSON 형식 응답 발견 (코드 블록)")
                    return json_obj
                except:
                    pass  # 파싱 실패하면 다음 방법 시도
            
            # 2. 중괄호로 둘러싸인 JSON 객체 추출 시도
            # 가장 바깥쪽 중괄호 쌍을 찾음
            json_pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
            json_match = re.search(json_pattern, text)
            if json_match:
                json_str = json_match.group(1).strip()
                try:
                    json_obj = json.loads(json_str)
                    self.status_signal.emit(f"{file_name} - JSON 객체 발견")
                    return json_obj
                except:
                    pass  # 파싱 실패하면 다음 방법 시도
            
            # 3. 전체 텍스트가 JSON인지 확인
            try:
                json_obj = json.loads(text)
                self.status_signal.emit(f"{file_name} - 유효한 JSON 응답 수신")
                return json_obj
            except:
                # 파싱에 실패하면 원본 텍스트 반환
                self.status_signal.emit(f"{file_name} - JSON 파싱 실패, 텍스트 응답으로 처리")
                return text
                
        except Exception as e:
            print(f"JSON 추출 오류: {e}")
            self.status_signal.emit(f"{file_name} - JSON 추출 실패: {e}")
            return text  # 오류 발생시 원본 텍스트 반환

    def emit_status_signal(self, message):
        """상태 메시지를 전송하는 편의 메서드"""
        try:
            self.status_signal.emit(message)
        except Exception as e:
            print(f"상태 메시지 전송 오류: {e}")

    def get_openai_client(self):
        """OpenAI 클라이언트 객체 반환"""
        try:
            # API 키가 없으면 설정에서 로드
            if not self.api_key and self.settings_handler:
                self.api_key = self.settings_handler.get_setting('openai_key', '')
            
            if not self.api_key:
                self.error_signal.emit("설정 오류", "API 키가 설정되지 않았습니다.")
                return None
            
            # OpenAI 클라이언트 초기화 - v2 beta 헤더 명시
            client = OpenAI(
                api_key=self.api_key,
                default_headers={"OpenAI-Beta": "assistants=v2"}  # v2 API 사용
            )
            print("OpenAI 클라이언트 객체 생성 완료 (assistants=v2)")
            return client
        except Exception as e:
            print(f"OpenAI 클라이언트 초기화 오류: {e}")
            self.error_signal.emit("API 설정 오류", str(e))
            return None

    def ensure_v2_client(self):
        """OpenAI 클라이언트가 v2 API를 사용하도록 보장"""
        try:
            if not self.client:
                # 클라이언트가 없으면 새로 생성
                self.client = OpenAI(
                    api_key=self.api_key,
                    default_headers={"OpenAI-Beta": "assistants=v2"}
                )
                print("OpenAI 클라이언트 새로 초기화 (assistants=v2)")
                return True
                
            # 기존 클라이언트의 헤더 확인
            if not hasattr(self.client, '_custom_headers') or self.client._custom_headers.get("OpenAI-Beta") != "assistants=v2":
                # 헤더가 없거나 올바르지 않으면 클라이언트 재생성
                print("기존 클라이언트의 베타 헤더가 잘못되었습니다. 클라이언트 재초기화...")
                self.client = OpenAI(
                    api_key=self.api_key,
                    default_headers={"OpenAI-Beta": "assistants=v2"}
                )
                print("OpenAI 클라이언트 재초기화 완료 (assistants=v2)")
            
            return True
        except Exception as e:
            print(f"v2 클라이언트 보장 오류: {e}")
            traceback.print_exc()
            return False