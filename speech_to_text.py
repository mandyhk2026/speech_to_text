import sys
import queue
import time
import numpy as np
import soundcard as sc
import whisper
from deep_translator import GoogleTranslator
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

# --- Configuration ---
SAMPLE_RATE = 16000
CHANNELS = 2  
BLOCK_SIZE = int(SAMPLE_RATE * 1)  # 1-second audio chunks
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
                    
                    if data is None or data.size == 0:
                        continue
                        
                    if data.ndim > 1 and data.shape[1] > 1:
                        audio_data = np.mean(data, axis=1).astype(np.float32)
                    else:
                        audio_data = data.flatten().astype(np.float32)
                    
                    if audio_data.size > 0 and np.max(np.abs(audio_data)) >= 0.01:
                        AUDIO_QUEUE.put(audio_data)
                        
        except Exception as e:
            print(f"Recorder Error: {e}")

    def stop(self):
        self.is_running = False
        self.wait()


class TranscriptionWorker(QThread):
    caption_ready = Signal(str, str)

    def __init__(self):
        super().__init__()
        print("Loading Whisper model...")
        self.model = whisper.load_model("tiny.en")
        self.translator = GoogleTranslator(source='en', target='zh-CN')
        self.is_running = True

    def run(self):
        print("Transcriber and Translator ready...")
        while self.is_running:
            try:
                audio_data = AUDIO_QUEUE.get(timeout=1)
                
                if audio_data is None or audio_data.size == 0:
                    AUDIO_QUEUE.task_done()
                    continue

                en_result = self.model.transcribe(
                    audio_data, 
                    fp16=False, 
                    language="en",
                    task="transcribe",
                    beam_size=1,
                    best_of=1,
                    temperature=0.0
                )
                en_text = en_result["text"].strip()

                if en_text:
                    try:
                        zh_text = self.translator.translate(en_text)
                    except Exception as tx_err:
                        print(f"Translation Error: {tx_err}")
                        zh_text = "[Translation Error]"
                    
                    self.caption_ready.emit(en_text, zh_text)
                
                AUDIO_QUEUE.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription Error: {e}")

    def stop(self):
        self.is_running = False
        self.wait()


class OverlayCaptionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.en_history = []
        self.zh_history = []
        
        self.init_ui()
        
        self.recorder = AudioRecorderWorker()
        self.transcriber = TranscriptionWorker()
        
        self.transcriber.caption_ready.connect(self.update_captions)
        
        self.recorder.start()
        self.transcriber.start()

    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        panel_layout = QVBoxLayout()
        panel_layout.setSpacing(4)  
        panel_layout.setContentsMargins(35, 10, 35, 10)

        # Row 1: English Text
        self.en_label = QLabel("Listening...")
        self.en_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.en_label.setWordWrap(True)
        self.en_label.setStyleSheet("color: #FFFFFF; font-size: 32px; font-weight: bold;")

        # Row 2: Chinese Text
        self.zh_label = QLabel("正在等待音频...")
        self.zh_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.zh_label.setWordWrap(True)
        self.zh_label.setStyleSheet("color: #FFCC00; font-size: 28px; font-weight: bold;")

        panel_layout.addWidget(self.en_label)
        panel_layout.addWidget(self.zh_label)

        self.container = QWidget(self)
        self.container.setLayout(panel_layout)
        self.container.setStyleSheet("""
            QWidget {
                background-color: rgba(15, 15, 15, 0.85);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.12);
            }
            QLabel {
                background: transparent;
                border: none;
                font-family: 'Segoe UI', Arial, 'Microsoft YaHei', sans-serif;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.container)
        
        self.resize_and_center()

    def resize_and_center(self):
        width = 1200
        height = 160
        
        # FIX: Enforce absolute maximum sizes so text wrap cannot aggressively push the box larger
        self.setFixedSize(width, height)
        
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - width) // 2
        y = screen.height() - height - 60
        self.move(x, y)

    @Slot(str, str)
    def update_captions(self, en_text, zh_text):
        self.en_history.append(en_text)
        self.zh_history.append(zh_text)
        
        if len(self.en_history) > 5:
            self.en_history = [self.en_history[-1]]
            self.zh_history = [self.zh_history[-1]]
            
        self.render_history_text()

    def render_history_text(self):
        en_display = " ".join(self.en_history)
        zh_display = " ".join(self.zh_history)
        
        self.en_label.setText(en_display if en_display else "Listening...")
        self.zh_label.setText(zh_display if zh_display else "正在等待音频...")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OverlayCaptionWindow()
    window.show()
    sys.exit(app.exec())