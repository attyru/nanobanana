import logging
import io
from typing import Optional, Generator, Tuple, List, Union, Dict, Any
from PIL import Image
import json

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    logging.error("Google GenAI library not found. Please install 'google-genai'.")

class GeminiClient:
    def __init__(self, api_key: str):
        if not HAS_GENAI:
            raise ImportError("Library 'google-genai' not installed.")
        
        if not api_key:
            logging.warning("GeminiClient initialized without API Key")
        
        self.client = genai.Client(api_key=api_key)
        self.history: List[types.Content] = []
        
        self.system_instruction = (
            "You are an expert digital art assistant integrated into Krita. "
            "Help the user generate images, variations, and provide creative advice."
        )
        logging.info("Gemini API client initialized (v2 SDK)")

    def _image_to_part(self, img: Image.Image) -> types.Part:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return types.Part(
            inline_data=types.Blob(
                mime_type="image/png",
                data=buf.getvalue()
            )
        )

    def _create_user_content(self, prompt_text: str, images: List[Image.Image]) -> types.Content:
        parts = []
        for img in images:
            try:
                parts.append(self._image_to_part(img))
            except Exception as e:
                logging.error(f"Failed to serialize image: {e}")
        
        if prompt_text:
            parts.append(types.Part(text=prompt_text))
            
        return types.Content(role="user", parts=parts)

    def _create_model_content(self, response_text: str) -> types.Content:
        return types.Content(role="model", parts=[types.Part(text=response_text)])

    def _get_config(self, seed: int, aspect_ratio: Optional[str]) -> types.GenerateContentConfig:
        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
        ]

        config_args = {
            "seed": seed,
            "safety_settings": safety_settings,
            "temperature": 0.7,
            "system_instruction": self.system_instruction,
            "response_modalities": ["TEXT"]
        }

        if aspect_ratio:
            config_args["response_modalities"].append("IMAGE")
            ar_val = aspect_ratio.split(" ")[0]
            if ":" in ar_val:
                config_args["image_config"] = types.ImageConfig(aspect_ratio=ar_val)

        return types.GenerateContentConfig(**config_args)

    def _stream_handler(self, stream) -> Generator[Tuple[Optional[str], Optional[Image.Image]], None, None]:
        try:
            for chunk in stream:
                if not chunk.candidates:
                    if chunk.prompt_feedback:
                        logging.warning(f"Prompt Feedback: {chunk.prompt_feedback}")
                    continue
                
                candidate = chunk.candidates[0]
                
                # Log safety/finish reasons if content is empty
                if not candidate.content or not candidate.content.parts:
                    logging.warning(f"Empty candidate. Finish Reason: {candidate.finish_reason}")
                    if candidate.safety_ratings:
                        logging.warning(f"Safety Ratings: {candidate.safety_ratings}")
                    continue

                for part in candidate.content.parts:
                    if part.text:
                        yield (part.text, None)
                    
                    if part.inline_data:
                        try:
                            img_data = part.inline_data.data
                            image = Image.open(io.BytesIO(img_data))
                            yield (None, image)
                        except Exception as e:
                            logging.error(f"Image decode error: {e}")
                            yield ("[System: Failed to decode image]", None)
                            
        except Exception as e:
            logging.error(f"Stream iteration error: {e}")
            if "503" in str(e) or "overloaded" in str(e).lower():
                yield ("\n[System: Google AI Model is overloaded (503). Please wait a moment and try again.]", None)
            else:
                yield (f"\n[System: Network Error - {str(e)}]", None)

    def send_prompt(self, prompt: str, images: List[Image.Image], seed: int, model: str, aspect_ratio: str) -> Generator[Tuple[Optional[str], Optional[Image.Image]], None, None]:
        user_content = self._create_user_content(prompt, images)
        request_history = self.history + [user_content]
        
        config = self._get_config(seed, aspect_ratio)
        logging.info(f"Sending Prompt. History: {len(self.history)} turns.")

        full_text = ""
        try:
            stream = self.client.models.generate_content_stream(
                model=model,
                contents=request_history,
                config=config
            )
            
            for text, img in self._stream_handler(stream):
                if text: full_text += text
                yield (text, img)
            
            if full_text or len(self.history) == 0:
                model_content = self._create_model_content(full_text)
                self.history.append(user_content)
                self.history.append(model_content)
                
        except Exception as e:
            logging.error(f"Generation Failed: {e}")
            yield (f"[System: Setup Error - {str(e)}]", None)

    def generate_variation(self, prompt: str, images: List[Image.Image], seed: int, model: str, aspect_ratio: str) -> Generator[Tuple[Optional[str], Optional[Image.Image]], None, None]:
        user_content = self._create_user_content(prompt, images)
        request_history = self.history + [user_content]
        
        config = self._get_config(seed, aspect_ratio)
        logging.info(f"Generating Variation. Seed: {seed}")

        try:
            stream = self.client.models.generate_content_stream(
                model=model,
                contents=request_history,
                config=config
            )
            yield from self._stream_handler(stream)
        except Exception as e:
            logging.error(f"Variation Failed: {e}")
            yield (f"[System: Variation Error - {str(e)}]", None)

    def enhance_prompt(self, user_prompt: str, images: List[Image.Image]) -> Dict[str, Any]:
        """
        Generates a structured, professional prompt based on user text and optional reference images.
        Uses gemini-3-pro-preview (or newer) and structured JSON output.
        """
        # Use modern reasoning model
        reasoning_model = "gemini-3-pro-preview" 
        
        # Full Schema definition (UniversalImagePrompt)
        schema = {
            "type": "object",
            "title": "UniversalImagePrompt",
            "description": "A structured, model-agnostic prompt description for text-to-image generation.",
            "properties": {
                "concept": {
                    "type": "string",
                    "description": "Very short summary of the main idea (1 short phrase)."
                },
                "scene": {
                    "type": "object",
                    "description": "What is happening, where, and with what key elements.",
                    "properties": {
                        "setting": {
                            "type": "string",
                            "description": "Environment or location (city street, forest, spaceship bridge, fantasy castle hall, abstract void, etc.)."
                        },
                        "time_of_day": {
                            "type": "string",
                            "description": "Time and ambience: dawn, golden hour, night, neon-lit, overcast, etc."
                        },
                        "key_elements": {
                            "type": "array",
                            "description": "Important objects, characters, creatures, or landmarks that must appear.",
                            "items": {
                                "type": "string"
                            }
                        },
                        "action": {
                            "type": "string",
                            "description": "Short description of what is happening in the scene. Leave empty if static."
                        }
                    }
                },
                "style": {
                    "type": "object",
                    "description": "Visual style and mood.",
                    "properties": {
                        "mood_keywords": {
                            "type": "array",
                            "description": "Mood/adjectives: dreamy, dark, epic, cozy, surreal, cyberpunk, whimsical, etc.",
                            "items": {
                                "type": "string"
                            }
                        },
                        "art_style": {
                            "type": "string",
                            "description": "Overall style: photorealistic, cinematic, digital painting, anime, pixel art, lowpoly, oil painting, concept art, etc."
                        },
                        "color_palette": {
                            "type": "string",
                            "description": "Dominant colors/palette: warm oranges and reds, cold blues, neon, pastel, monochrome, etc."
                        },
                        "influences": {
                            "type": "array",
                            "description": "Optional references: 'Studio Ghibli-like', 'dark fantasy', 'sci-fi concept art', etc. No direct copyrighted names if avoidable.",
                            "items": {
                                "type": "string"
                            }
                        }
                    }
                },
                "composition": {
                    "type": "object",
                    "description": "How the image is framed and composed.",
                    "properties": {
                        "framing": {
                            "type": "string",
                            "description": "Wide shot, mid shot, close-up, top-down, isometric, etc."
                        },
                        "focus": {
                            "type": "string",
                            "description": "What should be in focus or emphasized."
                        },
                        "depth_of_field": {
                            "type": "string",
                            "description": "Shallow DOF with blurred background, deep focus, etc."
                        },
                        "additional_notes": {
                            "type": "string",
                            "description": "Extra composition hints: symmetry, rule of thirds, leading lines, minimalism, cluttered detail, etc."
                        }
                    }
                },
                "technical": {
                    "type": "object",
                    "description": "Technical preferences that are model-agnostic.",
                    "properties": {
                        "render_quality": {
                            "type": "string",
                            "description": "Overall quality: high detail, ultra-detailed, painterly, sketchy, etc."
                        },
                        "aspect_ratio_hint": {
                            "type": "string",
                            "description": "Desired aspect ratio or orientation, e.g. '3:4 vertical', '16:9 wide', '1:1 square'. This is a hint, not a command."
                        },
                        "avoid": {
                            "type": "array",
                            "description": "What should NOT appear (e.g., 'no text', 'no watermark', 'no frame', etc.).",
                            "items": {
                                "type": "string"
                            }
                        }
                    }
                },
                "final_prompt": {
                    "type": "string",
                    "description": "Single coherent English description combining all important details from this JSON, ready to be used as a text prompt. No JSON/meta references."
                }
            },
            "required": [
                "concept",
                "final_prompt"
            ]
        }

        system_instr = (
            "You are an expert prompt engineer for generative AI. "
            "Your task is to convert a user's raw input (and optional reference image) into a structured "
            "image generation prompt using the provided JSON schema. "
            "The downstream model is optimized to understand this specific JSON structure directly. "
            "Do NOT simplify or summarize. Fill all relevant fields based on the input and your creative inference."
        )
        
        parts = []
        if user_prompt:
            parts.append(types.Part(text=f"User Request: {user_prompt}"))
        
        for img in images:
             parts.append(self._image_to_part(img))
             
        if not parts:
             return {"final_prompt": "No input provided."}

        content = types.Content(role="user", parts=parts)
        
        config = types.GenerateContentConfig(
            system_instruction=system_instr,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=1.0,
            top_p=0.95
        )

        try:
            response = self.client.models.generate_content(
                model=reasoning_model,
                contents=[content],
                config=config
            )
            
            if response.text:
                return json.loads(response.text)
            return {"final_prompt": "Error: Empty response from Magic Prompt."}
            
        except Exception as e:
            logging.error(f"Magic Prompt Failed: {e}")
            return {"final_prompt": f"Magic Prompt Error: {str(e)}"}

    def undo_last_turn(self) -> bool:
        if len(self.history) >= 2:
            self.history.pop()
            self.history.pop()
            return True
        return False

    def reset_session(self) -> None:
        self.history = []
