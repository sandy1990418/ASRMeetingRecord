from asr_interface import ASRInterface
from document_loader import DocumentLoader
from vector_store import MilvusVectorStore
from rag_chain import RAGChain
from utils import transcribe_audio
import os 


with open('./api_key/openai_key.txt', 'r') as f: 
    os.environ['OPENAI_API_KEY'] = f.read()


class RAGASR:
    def __init__(self):
        self.asr = ASRInterface()
        self.document_loader = DocumentLoader()
        self.vector_store = MilvusVectorStore()
        self.rag_chain = RAGChain(self.vector_store)

    def index_data(self, url):
        documents = self.document_loader.load(url)
        self.vector_store.add_documents(documents)

    def process_audio(self, audio_file):
        transcription = transcribe_audio(audio_file)
        return self.rag_chain.run(transcription)


if __name__ == "__main__":
    rag_asr = RAGASR()
    rag_asr.index_data("https://lilianweng.github.io/posts/2023-06-23-agent/")

    while True:
        audio_file = input("Enter the path to your audio file (or 'quit' to exit): ")
        if audio_file.lower() == "quit":
            break
        result = rag_asr.process_audio(audio_file)
        print(f"Result: {result}")
