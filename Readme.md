# RAG-ASR Integration Project

## Overview

This project integrates Automatic Speech Recognition (ASR) with Retrieval-Augmented Generation (RAG) for real-time meeting transcription and information retrieval. It uses OpenAI's Whisper for speech recognition, Milvus as a vector database for efficient similarity search, and Gradio for the user interface.

## Features

- Real-time speech transcription
- Vector-based information retrieval
- Interactive web interface
- Scalable vector storage with Milvus

## Prerequisites

- Python 3.8+
- Docker (for running Milvus)
- A CSV file containing your knowledge base (with a 'text' column)

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

4. Install and start Milvus:
   Follow the [official Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md) to set up Milvus using Docker.

## Project Structure

```
rag_asr_integration/
├── asr/
│   └── asr_interface.py
├── rag/
│   ├── data_preparation.py
│   ├── vector_database.py
│   └── retrieval_system.py
├── pipeline/
│   └── integration_pipeline.py
├── ui/
│   └── display_interface.py
├── utils/
│   ├── audio_processing.py
│   └── evaluation.py
├── main.py
├── requirements.txt
└── README.md
```

## Usage

1. Prepare your knowledge base:
   - Create a CSV file with a 'text' column containing the relevant information for your meetings.
   - Update the path in `main.py` to point to your CSV file.

2. Run the application:
   ```
   python main.py
   ```

3. Open the Gradio interface in your web browser (the URL will be displayed in the console).

4. Speak into your microphone. The application will transcribe your speech and display relevant information retrieved from your knowledge base.

## Customization

- To modify the ASR model, update the `ASRInterface` class in `asr/asr_interface.py`.
- To change the vector embedding model, update the `SentenceTransformer` model in `rag/data_preparation.py` and `rag/retrieval_system.py`.
- To adjust the Milvus settings, modify the `VectorDatabase` class in `rag/vector_database.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) for ASR
- [Milvus](https://milvus.io/) for vector similarity search
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [Gradio](https://gradio.app/) for the user interface