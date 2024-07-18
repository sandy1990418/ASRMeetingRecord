from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class RAGChain:
    def __init__(self, vector_store):
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.retriever = vector_store.vector_store.as_retriever(search_kwargs={"k": 4})

        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt},
        )

    def run(self, query):
        return self.chain.run(query)
