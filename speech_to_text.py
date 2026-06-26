import sys
import time
import queue
import re
# Wrap third-party imports to catch missing dependencies before the console closes
try:
    import numpy as np
    import soundcard as sc
    import whisper
    from deep_translator import GoogleTranslator
    from PySide6.QtCore import Qt, QThread, Signal, Slot, QPoint, QTimer, QVariantAnimation
    from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QScrollArea
except ImportError as e:
    print("\n" + "="*50)
    print("DEPENDENCY ERROR: A required library is missing!")
    print("="*50)
    print(f"Details: {e}")
    print("="*50)
    print("\nPlease run the following command to install dependencies:")
    print("pip install numpy PySide6 soundcard openai-whisper deep-translator")
    print("="*50)
    
    input("\nPress Enter to close the console...")
    sys.exit(1)

# --- Configuration ---
SAMPLE_RATE = 16000
CHANNELS = 2  
BLOCK_SIZE = int(SAMPLE_RATE * 1)  # 1-second audio chunks
SILENCE_THRESHOLD = 0.015          # Volume below this is considered silence

AUDIO_QUEUE = queue.Queue()


class AudioRecorderWorker(QThread):
    def __init__(self):
        super().__init__()
        self.is_running = True

    def run(self):
        try:
            default_speaker = sc.default_speaker()
            print(f"Active System Speaker: {default_speaker.name}")
            
            # FIX: Use soundcard's built-in string/substring matching by speaker name
            # This safely targets the correct system loopback device without crashing.
            loopback_mic = sc.get_microphone(default_speaker.name, include_loopback=True)
                
            print(f"Successfully hooked loopback onto: {loopback_mic.name}")

            with loopback_mic.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS) as recorder:
                while self.is_running:
                    data = recorder.record(numframes=BLOCK_SIZE)
                    if not self.is_running:
                        break
                    if data is None or data.size == 0:
                        continue
                        
                    if data.ndim > 1 and data.shape[1] > 1:
                        audio_data = np.mean(data, axis=1).astype(np.float32)
                    else:
                        audio_data = data.flatten().astype(np.float32)
                    
                    if audio_data.size > 0:
                        AUDIO_QUEUE.put((audio_data, time.time()))
                        
        except Exception as e:
            print(f"Recorder Error: {e}")

    def stop(self):
        self.is_running = False
        self.wait()


class TranslationWorker(QThread):
    """Dedicated thread for network translation to prevent audio blocking."""
    translation_ready = Signal(str, str, bool)

    def __init__(self):
        super().__init__()
        self.translator = GoogleTranslator(source='en', target='zh-CN')
        self.text_queue = queue.Queue()
        self.is_running = True
        self.last_translated_en = ""

    @Slot(str, bool)
    def queue_translation(self, en_text, is_final):
        if not is_final:
            with self.text_queue.mutex:
                self.text_queue.queue.clear()
        self.text_queue.put((en_text, is_final))

    def run(self):
        print("Translator ready...")
        while self.is_running:
            try:
                en_text, is_final = self.text_queue.get(timeout=0.1)
                
                if en_text == self.last_translated_en and not is_final:
                    self.text_queue.task_done()
                    continue
                    
                try:
                    zh_text = self.translator.translate(en_text) if en_text.strip() else ""
                    self.last_translated_en = en_text
                except Exception:
                    zh_text = "[Translation Error]"
                
                self.translation_ready.emit(en_text, zh_text, is_final)
                self.text_queue.task_done()
                
                if is_final:
                    self.last_translated_en = ""
                    
            except queue.Empty:
                continue

    def stop(self):
        self.is_running = False
        self.wait()


class TranscriptionWorker(QThread):
    text_ready = Signal(str, bool)

    def __init__(self):
        super().__init__()
        print("Loading Whisper model...")
        self.model = whisper.load_model("tiny.en")
        self.is_running = True
        self.phrase_buffer = []  
        self.last_en_text = ""

    def run(self):
        print("Transcriber ready...")
        while self.is_running:
            try:
                payload = AUDIO_QUEUE.get(timeout=1)
                chunk, record_time = payload
                max_amplitude = np.max(np.abs(chunk))
                
                if max_amplitude >= SILENCE_THRESHOLD:
                    self.phrase_buffer.append(chunk)
                    combined_audio = np.concatenate(self.phrase_buffer)
                    
                    en_text = self.transcribe_audio(combined_audio)
                    if en_text:
                        processing_lag = time.time() - record_time
                        
                        if processing_lag > 2.0:
                            print(f"Lag Alert ({processing_lag:.2f}s). Flushing backlog...")
                            self.phrase_buffer.clear()
                            
                            while not AUDIO_QUEUE.empty():
                                try:
                                    AUDIO_QUEUE.get_nowait()
                                    AUDIO_QUEUE.task_done()
                                except queue.Empty:
                                    break
                                    
                            self.text_ready.emit(en_text, True)  # FIX: Treat as final if forced-flushed
                        else:
                            if en_text != self.last_en_text:
                                self.last_en_text = en_text
                                self.text_ready.emit(en_text, False)
                        
                else:
                    if self.phrase_buffer:
                        combined_audio = np.concatenate(self.phrase_buffer)
                        final_en = self.transcribe_audio(combined_audio)
                        if final_en:
                            self.text_ready.emit(final_en, True)
                        
                        self.phrase_buffer.clear()
                        self.last_en_text = ""
                
                AUDIO_QUEUE.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription Error: {e}")

    def transcribe_audio(self, audio_data):
        try:
            result = self.model.transcribe(
                audio_data, fp16=False, language="en",
                task="transcribe", beam_size=1, temperature=0.0
            )
            return result["text"].strip()
        except Exception:
            return ""

    def stop(self):
        self.is_running = False
        self.wait()


class OverlayCaptionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.display_en = ""
        self.display_zh = ""
        self.drag_position = QPoint()
        
        self.clear_timer = QTimer(self)
        self.clear_timer.setSingleShot(True)
        self.clear_timer.timeout.connect(self.clear_captions)
        
        self.en_animation = QVariantAnimation(self)
        self.en_animation.setDuration(200)
        self.zh_animation = QVariantAnimation(self)
        self.zh_animation.setDuration(200)
        
        self.init_ui()
        
        self.recorder = AudioRecorderWorker()
        self.transcriber = TranscriptionWorker()
        self.translator = TranslationWorker()
        
        self.transcriber.text_ready.connect(self.translator.queue_translation)
        self.translator.translation_ready.connect(self.update_captions)
        
        self.recorder.start()
        self.transcriber.start()
        self.translator.start()

    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        panel_layout = QVBoxLayout()
        panel_layout.setSpacing(10)  
        panel_layout.setContentsMargins(35, 12, 35, 12)

        # English Section
        self.en_label = QLabel("Listening...")
        self.en_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.en_label.setWordWrap(True)  
        self.en_label.setStyleSheet("color: #FFFFFF; font-size: 32px; font-weight: bold; font-family: 'Segoe UI';")

        en_container = QWidget()
        en_container.setStyleSheet("background: transparent; border: none;")
        en_sub_layout = QVBoxLayout(en_container)
        en_sub_layout.setContentsMargins(0, 0, 0, 0)
        en_sub_layout.addWidget(self.en_label)

        self.en_scroll = QScrollArea()
        self.en_scroll.setWidget(en_container)
        self.en_scroll.setWidgetResizable(True)
        self.en_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.en_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.en_scroll.setStyleSheet("background: transparent; border: none;")

        # Chinese Section
        self.zh_label = QLabel("正在等待音频...")
        self.zh_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.zh_label.setWordWrap(True)  
        self.zh_label.setStyleSheet("color: #FFCC00; font-size: 28px; font-weight: bold; font-family: 'Microsoft YaHei';")

        zh_container = QWidget()
        zh_container.setStyleSheet("background: transparent; border: none;")
        zh_sub_layout = QVBoxLayout(zh_container)
        zh_sub_layout.setContentsMargins(0, 0, 0, 0)
        zh_sub_layout.addWidget(self.zh_label)

        self.zh_scroll = QScrollArea()
        self.zh_scroll.setWidget(zh_container)
        self.zh_scroll.setWidgetResizable(True)
        self.zh_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.zh_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.zh_scroll.setStyleSheet("background: transparent; border: none;")

        panel_layout.addWidget(self.en_scroll, stretch=1)
        panel_layout.addWidget(self.zh_scroll, stretch=1)

        self.container = QWidget(self)
        self.container.setObjectName("MainContainer") # FIX: Set object name to stop global style cascade
        self.container.setLayout(panel_layout)
        self.container.setCursor(Qt.OpenHandCursor)
        
        # FIX: Changed generic QWidget rule to targeted #MainContainer rule
        self.container.setStyleSheet("""
            QWidget#MainContainer {
                background-color: rgba(15, 15, 15, 0.85);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.12);
            }
            QLabel {
                background: transparent;
                border: none;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.container)
        
        self.en_animation.valueChanged.connect(lambda val: self.en_scroll.verticalScrollBar().setValue(val))
        self.zh_animation.valueChanged.connect(lambda val: self.zh_scroll.verticalScrollBar().setValue(val))
        
        self.resize_and_center()

    def resize_and_center(self):
        width = 1200
        height = 180  
        self.setFixedSize(width, height)
        
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - width) // 2
        y = screen.height() - height - 60
        self.move(x, y)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self.container.setCursor(Qt.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.container.setCursor(Qt.OpenHandCursor)

    @Slot(str, str, bool)
    def update_captions(self, en_text, zh_text, is_final):
        if not is_final:
            self.clear_timer.stop()

        self.display_en = en_text
        self.display_zh = zh_text
            
        self.render_history_text()

        if is_final and (en_text or zh_text):
            self.clear_timer.start(4000)  # FIX: 500ms was too fast. Upped to 4 seconds.

    @Slot()
    def clear_captions(self):
        self.display_en = ""
        self.display_zh = ""
        self.render_history_text()

    def render_history_text(self):
        def format_text(text):
            if not text:
                return text
            return re.sub(r'([.!?])\s+', r'\1\n', text)

        en_display = format_text(self.display_en) if self.display_en else "Listening..."
        zh_display = format_text(self.display_zh) if self.display_zh else "正在等待音频..."
        
        self.en_label.setText(en_display)
        self.zh_label.setText(zh_display)

        # FIX: Explicitly force UI engine updates before setting scroll layout metrics
        self.en_label.adjustSize()
        self.zh_label.adjustSize()
        QTimer.singleShot(30, self.animate_scroll_bars)

    def animate_scroll_bars(self):
        en_bar = self.en_scroll.verticalScrollBar()
        if en_bar.value() != en_bar.maximum():
            self.en_animation.stop()
            self.en_animation.setStartValue(en_bar.value())
            self.en_animation.setEndValue(en_bar.maximum())
            self.en_animation.start()

        zh_bar = self.zh_scroll.verticalScrollBar()
        if zh_bar.value() != zh_bar.maximum():
            self.zh_animation.stop()
            self.zh_animation.setStartValue(zh_bar.value())
            self.zh_animation.setEndValue(zh_bar.maximum())
            self.zh_animation.start()

    def closeEvent(self, event):
        self.clear_timer.stop()
        self.en_animation.stop()
        self.zh_animation.stop()
        self.recorder.stop()
        self.transcriber.stop()
        self.translator.stop()
        event.accept()


if __name__ == "__main__":
    # Define a custom exception handler to keep the console open on a crash
    def console_exception_hook(exctype, value, tb):
        import traceback
        print("\n" + "="*50)
        print("CRITICAL APPLICATION ERROR:")
        print("="*50)
        # Print the full error stack trace to the console
        traceback.print_exception(exctype, value, tb)
        print("="*50)
        
        # This keeps the terminal window open until you manually press Enter
        input("\nPress Enter to close the console...")
        sys.exit(1)

    # Register the hook
    sys.excepthook = console_exception_hook

    # Start the application
    app = QApplication(sys.argv)
    window = OverlayCaptionWindow()
    window.show()
    sys.exit(app.exec())