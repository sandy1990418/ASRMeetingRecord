from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings


class MilvusVectorStore:
    def __init__(self, host="localhost", port=19530):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            connection_args={"host": host, "port": port},
            collection_name="rag_collection",
        )

    def add_documents(self, documents):
        self.vector_store.add_documents(documents)

    def similarity_search(self, query, k=4):
        return self.vector_store.similarity_search(query, k=k)
