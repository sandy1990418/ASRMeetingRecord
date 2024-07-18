import unittest
from unittest.mock import patch, MagicMock
from rag_asr_app import RAGASR


class TestRAGASR(unittest.TestCase):
    @patch("rag_asr_app.ASRInterface")
    @patch("rag_asr_app.DocumentLoader")
    @patch("rag_asr_app.MilvusVectorStore")
    @patch("rag_asr_app.RAGChain")
    def setUp(self, mock_rag_chain, mock_vector_store, mock_document_loader, mock_asr):
        self.rag_asr = RAGASR()
        self.mock_asr = mock_asr.return_value
        self.mock_document_loader = mock_document_loader.return_value
        self.mock_vector_store = mock_vector_store.return_value
        self.mock_rag_chain = mock_rag_chain.return_value

    def test_index_data(self):
        url = "https://example.com"
        mock_documents = [MagicMock(), MagicMock()]
        self.mock_document_loader.load.return_value = mock_documents

        self.rag_asr.index_data(url)

        self.mock_document_loader.load.assert_called_once_with(url)
        self.mock_vector_store.add_documents.assert_called_once_with(mock_documents)

    @patch("rag_asr_app.transcribe_audio")
    def test_process_audio(self, mock_transcribe_audio):
        audio_file = "test_audio.wav"
        mock_transcription = "This is a test transcription."
        mock_rag_result = "This is the RAG result."

        mock_transcribe_audio.return_value = mock_transcription
        self.mock_rag_chain.run.return_value = mock_rag_result

        result = self.rag_asr.process_audio(audio_file)

        mock_transcribe_audio.assert_called_once_with(audio_file)
        self.mock_rag_chain.run.assert_called_once_with(mock_transcription)
        self.assertEqual(result, mock_rag_result)


if __name__ == "__main__":
    unittest.main()
