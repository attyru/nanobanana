# Nanobanana AI Chat

**Nanobanana AI Chat** is a specialized Krita plugin acting as an AI Copilot for digital artists. It integrates Google's **Gemini 2.5 Flash Image** model directly into Krita, allowing for conversational image generation, editing, and rapid variation creation.

## Features

*   **Copilot Interface**: Compact UI designed to sit in a dock without taking up screen space.
*   **Chat Session (History)**: The model remembers context (e.g., "Make it blue" works because it remembers the previous image).
*   **Batch Generation**: Generate 1 to 4 variations in a single click.
    *   Variation 1: Context-aware (updates chat history).
    *   Variations 2-4: Stateless (parallel-like exploration without polluting history).
*   **Img2Img Support**: Sends the current canvas visualization to the AI as reference.
*   **Real-time Streaming**: Text appears instantly; images load as layers as soon as they are ready.

## Architecture

The project uses a thread-safe, asynchronous architecture to ensure Krita's UI remains responsive during AI generation.

*   **`nanobanana.py`**: The main UI controller using PyQt5.
*   **`gemini_api.py`**: Handles communication with Google GenAI (Gemini 2.5/3.0).
*   **`krita_api.py`**: Bridge between Krita and Python.
*   **`utils.py`**: Manages persistent settings.

## Technology Stack

*   **Python 3.10+** (Embedded in Krita)
*   **PyQt5**: UI and Threading.
*   **Google GenAI SDK (`google-genai`)**: Interaction with Gemini models.
*   **Pillow (PIL)**: Image processing.

## Setup

1.  **Install**: Place `nanobanana` folder in Krita's `pykrita` directory.
2.  **Configure**: Open Settings (⚙️) in the docker and paste your Google AI Studio API Key.
3.  **Usage**:
    *   Select **Aspect Ratio**.
    *   Set **Batch Count** (x1 - x4).
    *   Check **Img2Img** if you want to edit the current drawing.
    *   Type prompt and hit **Generate**.

## Development

*   **Tests**: Run `pytest tests/ -v` to verify API logic and Utils.
