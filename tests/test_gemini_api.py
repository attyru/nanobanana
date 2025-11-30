import pytest
from unittest.mock import MagicMock, call, ANY
from gemini_api import GeminiClient
from google.genai import types
from PIL import Image
import io

class TestGeminiClient:
    
    def test_init(self, mock_genai_client):
        client = GeminiClient("fake_key")
        mock_genai_client.assert_called_with(api_key="fake_key")
        assert client.history == []
        assert client.system_instruction is not None

    def test_image_to_part_success(self, mock_genai_client):
        client = GeminiClient("key")
        img = Image.new('RGB', (10, 10), color='red')
        
        part = client._image_to_part(img)
        
        assert isinstance(part, types.Part)
        assert part.inline_data.mime_type == "image/png"
        assert len(part.inline_data.data) > 0
        
    def test_create_user_content(self, mock_genai_client):
        client = GeminiClient("key")
        img = Image.new('RGB', (10, 10))
        
        content = client._create_user_content("hello", [img])
        
        assert content.role == "user"
        assert len(content.parts) == 2
        # First part: Image (Blob)
        assert content.parts[0].inline_data is not None
        assert content.parts[0].inline_data.mime_type == "image/png"
        # Second part: Text
        assert content.parts[1].text == "hello"

    def test_create_model_content(self, mock_genai_client):
        client = GeminiClient("key")
        content = client._create_model_content("response text")
        assert content.role == "model"
        assert len(content.parts) == 1
        assert content.parts[0].text == "response text"

    def test_get_config(self, mock_genai_client):
        client = GeminiClient("key")
        
        # Test with Aspect Ratio
        config = client._get_config(123, "16:9 (Landscape)")
        assert config.seed == 123
        assert config.response_modalities == ["TEXT", "IMAGE"]
        assert config.image_config.aspect_ratio == "16:9"
        assert config.temperature == 0.7
        assert len(config.safety_settings) == 4
        
        # Test without Aspect Ratio
        config_no_ar = client._get_config(456, None)
        assert config_no_ar.response_modalities == ["TEXT"]
        assert not hasattr(config_no_ar, 'image_config') or config_no_ar.image_config is None

    def test_stream_handler_text_and_image(self, mock_genai_client):
        client = GeminiClient("key")
        
        # Mock response chunks
        chunk1 = MagicMock()
        chunk1.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text="Hello", inline_data=None)]))]
        
        img_bytes = io.BytesIO()
        Image.new('RGB', (10, 10)).save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        chunk2 = MagicMock()
        inline_data = MagicMock()
        inline_data.data = img_bytes.read()
        chunk2.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text=None, inline_data=inline_data)]))]
        
        stream = [chunk1, chunk2]
        results = list(client._stream_handler(stream))
        
        assert len(results) == 2
        assert results[0][0] == "Hello"
        assert isinstance(results[1][1], Image.Image)

    def test_stream_handler_error_503(self, mock_genai_client):
        client = GeminiClient("key")
        
        def error_gen():
            yield MagicMock() # Setup ok
            raise Exception("503 Service Unavailable: The model is overloaded.")
            
        results = list(client._stream_handler(error_gen()))
        
        # Should yield a friendly error message
        assert "503" in results[-1][0]
        assert "overloaded" in results[-1][0].lower()

    def test_stream_handler_generic_error(self, mock_genai_client):
        client = GeminiClient("key")
        
        def error_gen():
            raise Exception("Network connection closed")
            yield MagicMock() 
            
        results = list(client._stream_handler(error_gen()))
        
        assert "Network Error" in results[0][0]

    def test_send_prompt_success_updates_history(self, mock_genai_client):
        client = GeminiClient("key")
        mock_client_instance = mock_genai_client.return_value
        
        # Mock stream to yield text "World"
        chunk = MagicMock()
        chunk.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text="World", inline_data=None)]))]
        mock_client_instance.models.generate_content_stream.return_value = [chunk]

        results = list(client.send_prompt("Hello", [], 123, "model", None))
        
        # Check yield
        assert results[0][0] == "World"
        
        # Check History: User + Model
        assert len(client.history) == 2
        assert client.history[0].parts[0].text == "Hello"
        assert client.history[1].parts[0].text == "World"

    def test_send_prompt_api_failure(self, mock_genai_client):
        client = GeminiClient("key")
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content_stream.side_effect = Exception("Auth Error")
        
        results = list(client.send_prompt("Hello", [], 123, "model", None))
        
        assert "Setup Error" in results[0][0]
        assert "Auth Error" in results[0][0]
        assert len(client.history) == 0 # No history update on fail

    def test_generate_variation_no_history_update(self, mock_genai_client):
        client = GeminiClient("key")
        client.history = [MagicMock()] # Existing history
        mock_client_instance = mock_genai_client.return_value
        mock_client_instance.models.generate_content_stream.return_value = []
        
        list(client.generate_variation("Var", [], 123, "model", None))
        
        # History should remain same length
        assert len(client.history) == 1
        
        # Call verification
        args, kwargs = mock_client_instance.models.generate_content_stream.call_args
        # Request contents should be [History, UserNew]
        assert len(kwargs['contents']) == 2 

    def test_undo(self, mock_genai_client):
        client = GeminiClient("key")
        client.history = [1, 2, 3, 4] 
        
        assert client.undo_last_turn() is True
        assert len(client.history) == 2
        
        assert client.undo_last_turn() is True
        assert len(client.history) == 0
        
        assert client.undo_last_turn() is False

    def test_reset_session(self, mock_genai_client):
        client = GeminiClient("key")
        client.history = [1, 2]
        client.reset_session()
        assert len(client.history) == 0