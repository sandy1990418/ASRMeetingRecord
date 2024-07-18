# RAG-ASR Integration Project

## Overview

This project integrates Retrieval Augmented Generation (RAG) with Automatic Speech Recognition (ASR) for real-time transcription and information retrieval. It uses OpenAI's Whisper for speech recognition, Milvus as a vector database for efficient similarity search, and LangChain for the RAG pipeline.

## Features

- Real-time speech transcription using Whisper
- RAG-based information retrieval and question answering
- Integration with Milvus for scalable vector storage
- Flexible document loading and text splitting
- Easy-to-use interface for processing audio queries

## Prerequisites

- Python 3.8+
- Docker (for running Milvus)
- OpenAI API key
- Milvus server running locally or remotely

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rag-asr-integration.git
   cd rag-asr-integration
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up Milvus:
   Follow the [official Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md) to set up Milvus using Docker.

5. Set your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

## Project Structure

```
rag_asr_system/
├── rag_asr_app.py
├── asr_interface.py
├── document_loader.py
├── vector_store.py
├── rag_chain.py
├── utils.py
├── test_rag_asr_app.py
└── requirements.txt
```

## Usage

1. Run the main application:
   ```
   python rag_asr_app.py
   ```

2. When prompted, enter the path to your audio file. The system will transcribe the audio, process it through the RAG pipeline, and return the result.

3. To exit the application, type 'quit' when prompted for an audio file path.

## Customization

- To use a different ASR model, modify the `ASRInterface` class in `asr_interface.py`.
- To change the document source, update the URL in `rag_asr_app.py` or modify the `DocumentLoader` class in `document_loader.py`.
- To adjust the RAG chain parameters, modify the `RAGChain` class in `rag_chain.py`.

## Testing

Run the unit tests with:
```
python -m unittest test_rag_asr_app.py
```

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) for ASR
- [Milvus](https://milvus.io/) for vector similarity search
- [LangChain](https://github.com/hwchase17/langchain) for the RAG pipeline
- [OpenAI](https://openai.com/) for the language model and embeddings
