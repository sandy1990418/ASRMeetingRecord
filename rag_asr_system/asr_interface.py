import whisper


class ASRInterface:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_file):
        result = self.model.transcribe(audio_file)
        return result["text"]