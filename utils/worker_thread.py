from PyQt5.QtCore import QThread, pyqtSignal
import requests
import base64
from openai import OpenAI
from openai.types.beta import Assistant
from openai.types.beta.threads import Run
import time
import pandas as pd
import os
import json
import re
import traceback
from queue import Queue, Empty

class WorkerThread(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(str, str)  # file_path, response
    error = pyqtSignal(str)
    finished = pyqtSignal(list)  # [(file_path, response), ...]
    current_file = pyqtSignal(str)  # 현재 처리 중인 파일 경로
    remove_from_table = pyqtSignal(str)  # 테이블에서 제거할 파일 경로
    result_signal = pyqtSignal(str, dict)
    error_signal = pyqtSignal(str, str)
    status_signal = pyqtSignal(str)
    increment_progress_signal = pyqtSignal()

    def __init__(self, queue=None, settings_handler=None, image_processor=None, image_paths=None, api_key=None):
        super().__init__()
        self.queue = queue if queue is not None else Queue()
        self.settings_handler = settings_handler
        self.image_processor = image_processor
        self.stopped = False
        self.client = None
        self.thread = None
        self.run_id = None
        self.tool_calls = None
        self.api_key = api_key  # API 키 직접 설정
        self.assistant_id = None
        
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
        
        print("OpenAI client initialized")
        
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
            
            # OpenAI 클라이언트 초기화
            self.client = OpenAI(
                api_key=self.api_key,
                default_headers={"OpenAI-Beta": "assistants=v2"}
            )
            return True
        except Exception as e:
            self.error_signal.emit("API 설정 오류", str(e))
            return False

    def initialize_thread(self):
        """OpenAI API 스레드 초기화"""
        try:
            self.thread = self.client.beta.threads.create()
            return True
        except Exception as e:
            self.error_signal.emit("OpenAI 스레드 초기화 오류", str(e))
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
            with open(image_path, "rb") as file:
                response = self.client.files.create(
                    file=file,
                    purpose="assistants"
                )
                return response.id
        except Exception as e:
            print(f"Error uploading image: {e}")
            raise

    def request_extract_keyword(self, image_path):
        """단일 이미지에 대한 키워드 추출 요청"""
        max_retries = 3
        retry_delay = 2
        
        print(f"\n=== Processing Image: {os.path.basename(image_path)} ===")
        
        for attempt in range(max_retries):
            image_file = None
            thread = None
            file = None

            try:
                # 이미지 파일 크기 확인
                file_size = os.path.getsize(image_path)
                print(f"File size: {file_size} bytes")
                
                # 이미지 파일이 너무 크면 경고 로그 추가
                if file_size > 20 * 1024 * 1024:  # 20MB 이상
                    print(f"경고: 이미지 파일이 매우 큽니다 ({file_size / (1024*1024):.2f} MB). 처리 시간이 오래 걸릴 수 있습니다.")
                    self.status_signal.emit(f"경고: 이미지 파일이 매우 큽니다. 처리 시간이 오래 걸릴 수 있습니다.")
                
                # 이미지 파일 준비
                image_file = open(image_path, "rb")
                
                # 파일 업로드 (재시도 로직 추가)
                upload_attempts = 3
                for upload_attempt in range(upload_attempts):
                    try:
                        file = self.client.files.create(
                            file=image_file,
                            purpose="assistants"
                        )
                        print(f"File uploaded with ID: {file.id}")
                        break  # 성공하면 루프 종료
                    except Exception as upload_error:
                        if "timeout" in str(upload_error).lower() and upload_attempt < upload_attempts - 1:
                            print(f"파일 업로드 타임아웃. 재시도 중... ({upload_attempt+1}/{upload_attempts})")
                            self.status_signal.emit(f"파일 업로드 타임아웃. 재시도 중... ({upload_attempt+1}/{upload_attempts})")
                            time.sleep(retry_delay)
                            # 파일 포인터 위치 초기화
                            image_file.seek(0)
                        else:
                            raise  # 다른 오류이거나 최대 재시도 횟수 초과 시 예외 발생
                
                if not file:
                    raise Exception("파일 업로드 실패")

                # 스레드 생성
                thread = self.client.beta.threads.create()
                print(f"Thread created with ID: {thread.id}")

                # 메시지 추가
                message = self.client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=[
                        {
                            "type": "text",
                            "text": "이미지를 분석하고 vector store에서 관련 키워드를 찾아 다음 형식으로 응답해주세요:\n포토|이미지설명|주제및컨셉|인물정보|키워드|주요색상\n\n반드시 vector store에서 키워드를 참조해야 하며, 키워드는 한국어로만 제공해야 합니다. 키워드는 '키워드:점수' 형식으로 제공하세요. 예: '사람:0.95, 자연:0.85'"
                        },
                        {
                            "type": "image_file",
                            "image_file": {"file_id": file.id}
                        }
                    ]
                )
                print(f"Message added with ID: {message.id}")

                # 실행 생성
                run = self.client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=self.assistant_id
                )
                print(f"Run created with ID: {run.id}")

                # 응답 대기 (타임아웃 처리 로직이 추가된 메서드 호출)
                response = self.wait_for_completion(thread.id, run.id, image_path)
                
                if response:
                    # 응답의 줄바꿈 제거 및 공백 정리
                    response = ' '.join(response.split())
                    
                    # 응답을 파이프로 분리하고 앞뒤 공백 제거
                    parts = [part.strip() for part in response.split('|') if part.strip()]
                    
                    if len(parts) >= 6:
                        file_type = parts[0]  # AI가 분류한 파일 타입 사용
                        image_desc = parts[1]
                        theme = parts[2]
                        person_info = parts[3]
                        keywords = parts[4]
                        colors = parts[5]
                        
                        standardized_response = (
                            f"{file_type}|"  # AI가 분류한 타입 그대로 사용
                            f"{image_desc}|"
                            f"{theme}|"
                            f"{person_info}|"
                            f"{keywords}|"
                            f"{colors}"
                        ).strip()
                        
                        return standardized_response
                
                return response

            except Exception as e:
                print(f"Error in request: {str(e)}")
                if "timeout" in str(e).lower() and attempt < max_retries - 1:
                    print(f"타임아웃 오류 감지. 재시도 중... (Attempt {attempt + 1}/{max_retries})")
                    self.status_signal.emit(f"타임아웃 오류 감지. 재시도 중... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay * (attempt + 1))  # 지수 백오프 적용
                    continue
                elif attempt < max_retries - 1:
                    print(f"오류 발생. 재시도 중... (Attempt {attempt + 1}/{max_retries})")
                    self.status_signal.emit(f"오류 발생. 재시도 중... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                raise

            finally:
                if file:
                    try:
                        self.client.files.delete(file.id)
                        print(f"File {file.id} deleted")
                    except Exception as e:
                        print(f"Error deleting file: {e}")
                if image_file:
                    image_file.close()
                    print("Image file closed")

    def wait_for_completion(self, thread_id, run_id, image_path=None, timeout=600):
        """스레드 실행이 완료될 때까지 대기"""
        start_time = time.time()
        max_retries = 3  # 타임아웃 오류에 대한 최대 재시도 횟수
        retry_count = 0
        
        while time.time() - start_time < timeout:
            try:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run_id
                )
                
                status = run.status
                print(f"Run status: {status}")
                
                if status == "completed":
                    # 실행이 완료되면 메시지 검색
                    print("Run completed, retrieving messages...")
                    
                    messages = self.client.beta.threads.messages.list(
                        thread_id=thread_id,
                        order="desc",
                        limit=1
                    )
                    
                    if messages.data:
                        message = messages.data[0]
                        print(f"Found message from assistant (ID: {message.id})")
                        
                        # 텍스트 추출
                        text_content = None
                        for content in message.content:
                            if content.type == "text":
                                text_content = content.text.value
                                print(f"=== 텍스트 내용 ===\n{text_content}\n===================")
                                break
                        
                        if text_content:
                            return text_content
                        else:
                            print("No text content found in assistant message")
                            return "No text content found"
                    else:
                        print("No messages found")
                        return "No messages found"
                
                elif status == "failed":
                    error_message = f"Run failed: {run.last_error}"
                    print(error_message)
                    
                    # 타임아웃 오류 확인 및 재시도
                    if "Timeout while downloading" in str(run.last_error) and retry_count < max_retries:
                        retry_count += 1
                        print(f"이미지 다운로드 타임아웃 감지. 재시도 중... ({retry_count}/{max_retries})")
                        self.status_signal.emit(f"이미지 다운로드 타임아웃. 재시도 중... ({retry_count}/{max_retries})")
                        
                        # 새로운 실행 생성
                        try:
                            # 이전 실행 취소
                            self.client.beta.threads.runs.cancel(
                                thread_id=thread_id,
                                run_id=run_id
                            )
                            print(f"이전 실행 취소됨: {run_id}")
                            
                            # 새 실행 생성
                            new_run = self.client.beta.threads.runs.create(
                                thread_id=thread_id,
                                assistant_id=self.assistant_id
                            )
                            print(f"새 실행 생성됨: {new_run.id}")
                            
                            # 재귀적으로 새 실행 대기
                            return self.wait_for_completion(thread_id, new_run.id, image_path, timeout - (time.time() - start_time))
                        except Exception as retry_error:
                            print(f"재시도 중 오류 발생: {retry_error}")
                            return f"처리 실패: {error_message} (재시도 중 오류: {retry_error})"
                    
                    return f"처리 실패: {error_message}"
                
                elif status == "cancelled":
                    print("Run was cancelled")
                    return "처리가 취소되었습니다."
                
                else:
                    # 진행 중일 경우 잠시 대기
                    time.sleep(1)
            
            except Exception as e:
                error_message = f"Error checking run status: {e}"
                print(error_message)
                return f"오류: {error_message}"
        
        # 타임아웃 발생
        print(f"Timeout after {timeout} seconds")
        return "처리 시간이 초과되었습니다."

    def run(self):
        """스레드 실행 메서드"""
        try:
            # API 설정 로드
            if not self.load_api_settings():
                self.error_signal.emit("API 설정 오류", "API 키 또는 Assistant ID 로드 실패")
                return
            
            # 스레드 초기화
            if not self.initialize_thread():
                self.error_signal.emit("초기화 오류", "OpenAI 스레드 초기화 실패")
                return
            
            # 이미지 처리 완료 여부 추적 변수
            image_processed = False
            
            # 이미지 처리 루프
            while not self.stopped:
                try:
                    # 큐에서 이미지 경로 가져오기
                    if not self.queue.empty():
                        image_processed = True
                        image_path = self.queue.get(block=False)
                        
                        # 이미지 파일 확인
                        if not os.path.exists(image_path):
                            self.error_signal.emit("파일 오류", f"파일을 찾을 수 없습니다: {image_path}")
                            self.queue.task_done()
                            continue
                        
                        # 이미지 처리 요청
                        response = self.request_extract_keyword(image_path)
                        
                        # 응답이 문자열인 경우 딕셔너리로 변환
                        if isinstance(response, str) and '|' in response:
                            print(f"파이프 구분 응답 변환 중: {response}")
                            parts = [part.strip() for part in response.split('|') if part.strip()]
                            
                            if len(parts) >= 6:
                                result_dict = {
                                    "file_type": parts[0],
                                    "image_desc": parts[1],
                                    "theme": parts[2],
                                    "person_info": parts[3],
                                    "keywords": "",
                                    "keywords_vector": "",
                                    "keywords_ai": "",
                                    "colors": parts[5]
                                }
                                
                                # 키워드에 점수 형식이 있는지 확인 (예: 키워드:0.95)
                                if ':' in parts[4]:
                                    keywords_with_scores = parts[4].split(',')
                                    keywords_only = []
                                    vector_keywords = []
                                    ai_keywords = []
                                    
                                    # 영어 키워드 감지 플래그
                                    has_english_keywords = False
                                    
                                    for kw in keywords_with_scores:
                                        kw = kw.strip()
                                        if ':' in kw:
                                            # 벡터 스토어에서 가져온 키워드 (점수 있음)
                                            keyword_parts = kw.split(':')
                                            keyword_part = keyword_parts[0].strip()
                                            score = float(keyword_parts[1].strip()) if len(keyword_parts) > 1 else 0
                                            
                                            # 영어 키워드 감지 (영어 알파벳이 50% 이상인 경우)
                                            english_char_count = sum(1 for c in keyword_part if 'a' <= c.lower() <= 'z')
                                            if english_char_count > len(keyword_part) * 0.5:
                                                has_english_keywords = True
                                                print(f"영어 키워드 감지됨: {keyword_part}")
                                            
                                            keywords_only.append(f"{keyword_part} ({score:.2f})")
                                            vector_keywords.append(f"{keyword_part} ({score:.2f})")
                                        else:
                                            # AI가 생성한 키워드 (점수 없음)
                                            keywords_only.append(kw)
                                            ai_keywords.append(kw)
                                    
                                    # 영어 키워드가 감지된 경우 경고 로그 추가
                                    if has_english_keywords:
                                        print("경고: 영어 키워드가 감지되었습니다. 한국어 키워드만 사용해야 합니다.")
                                        self.status_signal.emit("경고: 영어 키워드가 감지되었습니다. 한국어 키워드만 사용해야 합니다.")
                                    
                                    # 벡터 스토어 키워드가 없는 경우 오류 처리
                                    if not vector_keywords:
                                        print("오류: 벡터 스토어 참조 키워드가 없습니다. 기본 키워드를 사용합니다.")
                                        self.status_signal.emit("오류: 벡터 스토어 참조 키워드가 없습니다. 기본 키워드를 사용합니다.")
                                        # 기본 키워드 추가 (예시)
                                        default_keywords = ["이미지 (0.90)", "사진 (0.85)", "콘텐츠 (0.80)"]
                                        vector_keywords = default_keywords
                                        keywords_only.extend(default_keywords)
                                    
                                    # 키워드 형식 통일 및 개행 추가
                                    result_dict["keywords"] = ', '.join(keywords_only)
                                    result_dict["keywords_vector"] = ', '.join(vector_keywords)
                                    result_dict["keywords_ai"] = ', '.join(ai_keywords) if ai_keywords else "AI 생성 키워드 없음"
                                else:
                                    # 점수 형식이 없는 경우 모든 키워드를 AI 생성으로 간주하고 벡터 스토어 키워드 추가
                                    print("오류: 키워드에 점수 형식이 없습니다. 기본 벡터 스토어 키워드를 추가합니다.")
                                    self.status_signal.emit("오류: 키워드에 점수 형식이 없습니다. 기본 벡터 스토어 키워드를 추가합니다.")
                                    
                                    # 쉼표로 구분된 키워드를 개행으로 변환
                                    keywords = parts[4].split(',')
                                    formatted_keywords = ', '.join([kw.strip() for kw in keywords])
                                    
                                    # 기본 벡터 스토어 키워드 추가
                                    default_vector_keywords = ["이미지 (0.90)", "사진 (0.85)", "콘텐츠 (0.80)"]
                                    vector_keywords_str = ', '.join(default_vector_keywords)
                                    
                                    # 결과 저장
                                    result_dict["keywords"] = formatted_keywords + ", " + vector_keywords_str
                                    result_dict["keywords_vector"] = vector_keywords_str
                                    result_dict["keywords_ai"] = formatted_keywords
                                
                                # 결과 전송
                                self.status_signal.emit(f"파일 처리 완료: {os.path.basename(image_path)}")
                                self.result_signal.emit(image_path, result_dict)
                            else:
                                self.error_signal.emit("응답 오류", f"응답 형식이 올바르지 않습니다: {response}")
                        else:
                            self.error_signal.emit("응답 오류", f"응답이 파이프로 구분되지 않았습니다: {response}")
                        
                        # 진행률 증가 시그널 발생
                        print(f"DEBUG - Emitting increment_progress_signal for file: {os.path.basename(image_path)}")
                        self.increment_progress_signal.emit()
                        self.queue.task_done()
                    else:
                        # 큐가 비어있고 이미지를 처리한 적이 있으면 처리 완료로 간주
                        if image_processed and self.queue.empty():
                            print("모든 이미지 처리 완료. 완료 시그널 발생")
                            break  # 루프 종료
                        else:
                            # 큐가 비어있으면 잠시 대기
                            time.sleep(0.5)
                
                except Empty:
                    # 큐가 비었을 때 예외 처리
                    time.sleep(0.5)
                
                except Exception as e:
                    # 기타 예외 처리
                    print(f"이미지 처리 중 오류: {str(e)}")
                    self.error_signal.emit("처리 오류", str(e))
            
            # 모든 처리가 완료되면 finished 시그널 발생
            if not self.stopped and image_processed:
                print("처리 완료 신호 전송")
                self.finished.emit([])  # 빈 리스트 전달
        
        except Exception as e:
            print(f"WorkerThread.run 오류: {str(e)}")
            self.error_signal.emit("스레드 오류", str(e))

    def stop(self):
        """스레드 중지"""
        self.stopped = True

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