from dotenv import load_dotenv
from asr_interface import ASRInterface
from document_loader import DocumentLoader
from vector_store import MilvusVectorStore
from rag_chain import RAGChain
from utils import transcribe_audio

load_dotenv()


class RAGASR:
    def __init__(self, milvus_host="localhost", milvus_port=19530):
        self.asr = ASRInterface()
        self.document_loader = DocumentLoader()
        self.vector_store = MilvusVectorStore(host=milvus_host, port=milvus_port)
        self.rag_chain = None  # We'll initialize this after adding documents

    def index_data(self, url):
        documents = self.document_loader.load(url)
        self.vector_store.add_documents(documents)
        # Initialize RAGChain after adding documents
        self.rag_chain = RAGChain(self.vector_store)

    def process_audio(self, audio_file):
        if self.rag_chain is None:
            print("Please index data before processing audio.")
            return None
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
        if result:
            print(f"Result: {result}")
