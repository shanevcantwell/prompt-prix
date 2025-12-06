"""
FaraService - Visual UI element location using Microsoft Fara-7B.

Fara-7B is Microsoft's agentic vision model for computer use, built on
Qwen2.5-VL-7B. It outputs Playwright actions directly (click, type, etc).

Ref: https://github.com/microsoft/fara
     https://huggingface.co/microsoft/Fara-7B

Usage:
    fara = FaraService(server_url="http://localhost:1234")

    # Take screenshot, locate element
    result = await fara.locate("The send button", screenshot_b64)
    if result["found"]:
        await page.mouse.click(result["x"], result["y"])

Prerequisites:
    - Fara-7B GGUF loaded in LM Studio
    - LM Studio server running with vision API enabled
"""

import asyncio
import base64
import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


# Default native resolutions for vision models
DEFAULT_NATIVE_RESOLUTIONS = {
    "square": (1024, 1024),
    "landscape": (1428, 896),
    "portrait": (896, 1428),
}


@dataclass
class FaraService:
    """
    Visual UI element location service using vision models.

    Handles resolution scaling internally - callers work in original
    screenshot coordinates.

    Configuration:
        Set FARA_SERVER_URL and FARA_MODEL_ID in .env, or pass explicitly.
    """
    server_url: Optional[str] = None
    model_id: Optional[str] = None
    timeout: float = 30.0
    native_resolutions: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: DEFAULT_NATIVE_RESOLUTIONS.copy()
    )

    def __post_init__(self):
        """Resolve server_url and model_id from config if not provided."""
        from prompt_prix.config import get_fara_config
        default_server, default_model = get_fara_config()
        if self.server_url is None:
            self.server_url = default_server
        if self.model_id is None:
            self.model_id = default_model

    async def locate(
        self,
        description: str,
        screenshot_b64: str,
    ) -> Dict[str, Any]:
        """
        Locate a UI element by description.

        Args:
            description: Natural language description of element
                        e.g., "The send button", "Text input area"
            screenshot_b64: Base64-encoded screenshot

        Returns:
            Dict with keys:
                found: bool
                x: int (in original screenshot coordinates)
                y: int (in original screenshot coordinates)
                confidence: float (0-1)
        """
        try:
            # Get original dimensions for coordinate scaling
            original_size = self._get_image_dimensions(screenshot_b64)
            native_size = self._select_best_resolution(*original_size)

            # Scale down for model
            scaled_b64 = self._scale_to_native(screenshot_b64, native_size)

            # Build vision request
            prompt = f"Locate the UI element: {description}. Return coordinates as JSON: {{\"found\": true/false, \"x\": pixels, \"y\": pixels}}"

            result = await self._call_vision_model(prompt, scaled_b64)

            # Scale coordinates back to original resolution
            if result.get("found") and result.get("x") is not None:
                native_x, native_y = result["x"], result["y"]
                result["x"], result["y"] = self._scale_coordinates(
                    native_x, native_y, original_size, native_size
                )
                logger.debug(f"Fara: Scaled ({native_x}, {native_y}) -> ({result['x']}, {result['y']})")

            return result

        except Exception as e:
            logger.error(f"Fara.locate failed: {e}")
            return {"found": False, "error": str(e)}

    async def verify(
        self,
        description: str,
        screenshot_b64: str,
    ) -> Dict[str, Any]:
        """
        Verify a UI element exists.

        Args:
            description: Natural language description
            screenshot_b64: Base64-encoded screenshot

        Returns:
            Dict with keys: exists (bool), confidence (float)
        """
        try:
            original_size = self._get_image_dimensions(screenshot_b64)
            native_size = self._select_best_resolution(*original_size)
            scaled_b64 = self._scale_to_native(screenshot_b64, native_size)

            prompt = f"Does this UI contain: {description}? Answer with JSON: {{\"found\": true/false, \"confidence\": 0.0-1.0}}"

            result = await self._call_vision_model(prompt, scaled_b64)

            return {
                "exists": result.get("found", False),
                "confidence": result.get("confidence", 0.0),
            }

        except Exception as e:
            logger.error(f"Fara.verify failed: {e}")
            return {"exists": False, "error": str(e)}

    async def _call_vision_model(
        self,
        prompt: str,
        image_b64: str,
    ) -> Dict[str, Any]:
        """Call the vision model via LM Studio API."""

        # Ensure proper data URL format
        if not image_b64.startswith("data:"):
            image_b64 = f"data:image/png;base64,{image_b64}"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a vision model for computer use. "
                    "Analyze screenshots and locate UI elements. "
                    "Respond with JSON only, no explanation."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_b64}}
                ]
            }
        ]

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.server_url}/v1/chat/completions",
                json={
                    "model": self.model_id,
                    "messages": messages,
                    "max_tokens": 256,
                    "temperature": 0.0,
                }
            )
            response.raise_for_status()
            data = response.json()

        text = data["choices"][0]["message"]["content"]
        logger.info(f"Fara raw response: {text}")

        parsed = self._parse_response(text)
        logger.info(f"Fara parsed: {parsed}")
        return parsed

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse model response, handling various formats."""

        # Try <tool_call> XML format (Fara native)
        tool_match = re.search(r"<tool_call>\s*({.*?})\s*</tool_call>", text, re.DOTALL)
        if tool_match:
            try:
                tool_json = json.loads(tool_match.group(1))
                return self._normalize_tool_call(tool_json)
            except json.JSONDecodeError:
                pass

        # Try markdown code block
        md_match = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
        if md_match:
            try:
                return json.loads(md_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try raw JSON
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass

        logger.warning(f"Fara: Could not parse response: {text[:100]}")
        return {"found": False, "error": "Could not parse response"}

    def _normalize_tool_call(self, tool_json: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Fara's tool_call format to Playwright action.

        Fara outputs Playwright actions directly:
            {"name": "computer", "arguments": {"action": "left_click", "coordinate": [x, y]}}
            {"name": "computer", "arguments": {"action": "type", "text": "hello"}}
            {"name": "computer", "arguments": {"action": "scroll", "coordinate": [x, y], "direction": "down"}}

        Returns dict with:
            found: bool - whether element was located
            action: str - Playwright action (left_click, right_click, type, scroll, key, etc.)
            x, y: int - coordinates (if applicable)
            text: str - text to type (if applicable)
            direction: str - scroll direction (if applicable)
        """
        args = tool_json.get("arguments", {})
        action = args.get("action", "left_click")
        coord = args.get("coordinate", [])

        result = {
            "found": len(coord) >= 2 or action in ("type", "key"),
            "action": action,
        }

        if len(coord) >= 2:
            result["x"] = coord[0]
            result["y"] = coord[1]

        if "text" in args:
            result["text"] = args["text"]

        if "direction" in args:
            result["direction"] = args["direction"]

        return result

    def _get_image_dimensions(self, b64_image: str) -> Tuple[int, int]:
        """Get dimensions of base64-encoded image."""
        try:
            from PIL import Image
        except ImportError:
            # Fallback: assume standard resolution
            logger.warning("PIL not available, assuming 1920x1080")
            return (1920, 1080)

        if "," in b64_image:
            b64_image = b64_image.split(",", 1)[1]

        img_bytes = base64.b64decode(b64_image)
        img = Image.open(io.BytesIO(img_bytes))
        return img.size

    def _select_best_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """Select native resolution based on aspect ratio."""
        ratio = width / height

        if ratio > 1.2:
            return self.native_resolutions.get("landscape", (1428, 896))
        elif ratio < 0.8:
            return self.native_resolutions.get("portrait", (896, 1428))
        else:
            return self.native_resolutions.get("square", (1024, 1024))

    def _scale_to_native(self, b64_image: str, native_res: Tuple[int, int]) -> str:
        """Scale image to native resolution."""
        try:
            from PIL import Image
        except ImportError:
            # Can't scale without PIL, return as-is
            return b64_image

        prefix = ""
        if "," in b64_image:
            prefix, b64_image = b64_image.split(",", 1)
            prefix += ","

        img_bytes = base64.b64decode(b64_image)
        img = Image.open(io.BytesIO(img_bytes))

        scaled = img.resize(native_res, Image.LANCZOS)

        buffer = io.BytesIO()
        scaled.save(buffer, format="PNG")
        scaled_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return prefix + scaled_b64

    def _scale_coordinates(
        self,
        x: int,
        y: int,
        original_size: Tuple[int, int],
        native_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Scale coordinates from native back to original resolution."""
        scale_x = original_size[0] / native_size[0]
        scale_y = original_size[1] / native_size[1]
        return (int(x * scale_x), int(y * scale_y))
