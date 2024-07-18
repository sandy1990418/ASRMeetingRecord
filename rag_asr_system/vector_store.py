from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from pymilvus import connections


class MilvusVectorStore:
    def __init__(self, host="localhost", port=19530):
        self.embeddings = OpenAIEmbeddings()
        self.host = host
        self.port = port
        self.collection_name = "rag_collection"
        self.vector_store = None
        self.connect()

    def connect(self):
        connections.connect(alias="default", host=self.host, port=self.port)
        print(f"Connected to Milvus at {self.host}:{self.port}")

    def add_documents(self, documents):
        if not documents:
            print("No documents to add.")
            return

        if self.vector_store is None:
            self.vector_store = Milvus.from_documents(
                documents,
                self.embeddings,
                collection_name=self.collection_name,
                connection_args={"host": self.host, "port": self.port},
            )
        else:
            self.vector_store.add_documents(documents)

        print(f"Added {len(documents)} documents to Milvus.")

    def similarity_search(self, query, k=4):
        if self.vector_store is None:
            print("Vector store is not initialized. Please add documents first.")
            return []
        return self.vector_store.similarity_search(query, k=k)
