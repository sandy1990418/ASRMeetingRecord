from asr_interface import ASRInterface

asr = ASRInterface()


def transcribe_audio(audio_file):
    return asr.transcribe(audio_file)
