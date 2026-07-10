# pyrefly: ignore [missing-import]
import whisper

model = whisper.load_model("base")
result = model.transcribe('idioms.wav', fp16=False)