import sys
import queue
import numpy as np
import soundcard as sc
import whisper
from deep_translator import GoogleTranslator
from PySide6.QtCore import Qt, QThread, Signal, Slot, QPoint, QTimer, QVariantAnimation
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QScrollArea

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
            
            loopback_mic = sc.get_microphone(
                id=default_speaker.id, 
                include_loopback=True
            )
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
                        AUDIO_QUEUE.put(audio_data)
                        
        except Exception as e:
            print(f"Recorder Error: {e}")

    def stop(self):
        self.is_running = False
        self.wait()


class TranscriptionWorker(QThread):
    caption_ready = Signal(str, str, bool)

    def __init__(self):
        super().__init__()
        print("Loading Whisper model...")
        self.model = whisper.load_model("tiny.en")
        self.translator = GoogleTranslator(source='en', target='zh-CN')
        self.is_running = True
        self.phrase_buffer = []  
        self.reset_requested = False 

    def request_buffer_reset(self):
        self.reset_requested = True

    def run(self):
        print("Transcriber and Translator ready...")
        while self.is_running:
            try:
                if self.reset_requested:
                    self.phrase_buffer.clear()
                    self.reset_requested = False

                chunk = AUDIO_QUEUE.get(timeout=1)
                max_amplitude = np.max(np.abs(chunk))
                
                if max_amplitude >= SILENCE_THRESHOLD:
                    self.phrase_buffer.append(chunk)
                    
                    # ADDED: Check if buffer exceeds 20 blocks (20 seconds).
                    # If so, drop old context to safeguard memory and stop model lag.
                    if len(self.phrase_buffer) > 20:
                        print("Buffer exceeded 20 seconds. Clearing old segments...")
                        self.phrase_buffer = self.phrase_buffer[-1:] # Keep only the newest 1-second block
                    
                    combined_audio = np.concatenate(self.phrase_buffer)
                    
                    en_text = self.transcribe_audio(combined_audio)
                    if en_text:
                        zh_text = self.translate_text(en_text)
                        self.caption_ready.emit(en_text, zh_text, False)
                        
                else:
                    if self.phrase_buffer:
                        combined_audio = np.concatenate(self.phrase_buffer)
                        final_en = self.transcribe_audio(combined_audio)
                        if final_en:
                            final_zh = self.translate_text(final_en)
                            self.caption_ready.emit(final_en, final_zh, True)
                        
                        self.phrase_buffer.clear()
                
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

    def translate_text(self, text):
        try:
            return self.translator.translate(text)
        except Exception:
            return "[Translation Error]"

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
        self.transcriber.caption_ready.connect(self.update_captions)
        
        self.recorder.start()
        self.transcriber.start()

    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)

        panel_layout = QVBoxLayout()
        panel_layout.setSpacing(10)  
        panel_layout.setContentsMargins(35, 12, 35, 12)

        # English Section
        self.en_label = QLabel("Listening...")
        self.en_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.en_label.setWordWrap(True)  
        self.en_label.setStyleSheet("color: #FFFFFF; font-size: 32px; font-weight: bold; font-family: 'Segoe UI';")

        en_container = QWidget()
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
        self.container.setLayout(panel_layout)
        self.container.setCursor(Qt.OpenHandCursor)
        self.container.setStyleSheet("""
            QWidget {
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
            self.clear_timer.start(500)

    @Slot()
    def clear_captions(self):
        self.display_en = ""
        self.display_zh = ""
        self.render_history_text()

    def render_history_text(self):
        en_display = self.display_en if self.display_en else "Listening..."
        zh_display = self.display_zh if self.display_zh else "正在等待音频..."
        
        self.en_label.setText(en_display)
        self.zh_label.setText(zh_display)

        QTimer.singleShot(20, self.animate_scroll_bars)

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
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OverlayCaptionWindow()
    window.show()
    sys.exit(app.exec())