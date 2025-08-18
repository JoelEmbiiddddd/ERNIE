# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Optimized Chatbot architecture supporting text, multimodal, and thought models
with enhanced debugging capabilities
"""
import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

import gradio as gr
import openai

from erniekit.webui.common import config


@dataclass
class ChatRequest:
    """Chat request configuration"""

    message: str
    model_name: str
    history: List[Dict[str, str]]
    file_input: Optional[List[str]] = None  # Added file input parameter
    role_setting: Optional[str] = None
    system_prompt: Optional[str] = None
    max_length: int = 1000
    top_p: float = 0.8
    temperature: float = 0.7
    port: int = 8188
    enable_thinking: bool = False  # Added for thought models


@dataclass
class DebugInfo:
    """Debug information for each chat turn"""

    timestamp: str
    session_id: str
    model_type: str
    request_data: Dict[str, Any]
    processed_messages: List[Dict[str, str]]
    response_content: str
    thought_content: str = ""
    generation_time: float = 0.0
    token_count: int = 0
    error_info: Optional[str] = None


class DebugLogger:
    """Debug logging and output system"""

    def __init__(self, enabled: bool = False, log_level: str = "INFO"):
        self.enabled = enabled
        self.session_logs: Dict[str, List[DebugInfo]] = {}
        self.current_session_id: Optional[str] = None

        # Setup logging
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        self.logger = logging.getLogger(__name__)

    def enable_debug(self, session_id: Optional[str] = None):
        """Enable debug mode"""
        self.enabled = True
        if session_id:
            self.current_session_id = session_id
        else:
            self.current_session_id = f"session_{int(time.time())}"

        if self.current_session_id not in self.session_logs:
            self.session_logs[self.current_session_id] = []

        self.logger.info(f"Debug mode enabled for session: {self.current_session_id}")

    def disable_debug(self):
        """Disable debug mode"""
        self.enabled = False
        self.logger.info("Debug mode disabled")

    def log_debug_info(self, debug_info: DebugInfo):
        """Log debug information"""
        if not self.enabled or not self.current_session_id:
            return

        # Add to session logs
        self.session_logs[self.current_session_id].append(debug_info)

        # Print debug info
        self._print_debug_info(debug_info)

    def _print_debug_info(self, debug_info: DebugInfo):
        """Print formatted debug information"""
        print("\n" + "=" * 80)
        print(f"🐛 DEBUG INFO - {debug_info.timestamp}")
        print("=" * 80)
        print(f"📋 Session ID: {debug_info.session_id}")
        print(f"🤖 Model Type: {debug_info.model_type}")
        print(f"⏱️  Generation Time: {debug_info.generation_time:.2f}s")
        print(f"🔢 Token Count: {debug_info.token_count}")

        print("\n📥 REQUEST DATA:")
        print("-" * 40)
        print(f"Message: {debug_info.request_data.get('message', 'N/A')}")
        print(f"File Input: {debug_info.request_data.get('file_input', 'N/A')}")
        print(f"Role Setting: {debug_info.request_data.get('role_setting', 'N/A')}")
        print(f"System Prompt: {debug_info.request_data.get('system_prompt', 'N/A')}")
        print(f"Temperature: {debug_info.request_data.get('temperature', 'N/A')}")
        print(f"Top P: {debug_info.request_data.get('top_p', 'N/A')}")
        print(f"Max Length: {debug_info.request_data.get('max_length', 'N/A')}")
        print(f"Port: {debug_info.request_data.get('port', 'N/A')}")
        print(
            f"Enable Thinking: {debug_info.request_data.get('enable_thinking', 'N/A')}"
        )

        print("\n📨 PROCESSED MESSAGES:")
        print("-" * 40)
        for i, msg in enumerate(debug_info.processed_messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content_preview = f"[Multimodal content with {len(content)} parts]"
            else:
                content_preview = content[:200] + ("..." if len(content) > 200 else "")
            print(f"  {i + 1}. [{role.upper()}]: {content_preview}")

        if debug_info.thought_content:
            print("\n🤔 THOUGHT PROCESS:")
            print("-" * 40)
            thought_preview = debug_info.thought_content[:300] + (
                "..." if len(debug_info.thought_content) > 300 else ""
            )
            print(thought_preview)

        print("\n💬 RESPONSE CONTENT:")
        print("-" * 40)
        response_preview = debug_info.response_content[:300] + (
            "..." if len(debug_info.response_content) > 300 else ""
        )
        print(response_preview)

        if debug_info.error_info:
            print("\n❌ ERROR INFO:")
            print("-" * 40)
            print(debug_info.error_info)

        print("=" * 80 + "\n")

    def get_session_logs(self, session_id: Optional[str] = None) -> List[DebugInfo]:
        """Get logs for a specific session"""
        target_session = session_id or self.current_session_id
        return self.session_logs.get(target_session, [])

    def export_session_logs(
        self, session_id: Optional[str] = None, format: str = "json"
    ) -> str:
        """Export session logs in specified format"""
        logs = self.get_session_logs(session_id)

        if format.lower() == "json":
            return json.dumps(
                [asdict(log) for log in logs], ensure_ascii=False, indent=2
            )
        elif format.lower() == "txt":
            output = []
            for log in logs:
                output.append(f"Timestamp: {log.timestamp}")
                output.append(f"Model Type: {log.model_type}")
                output.append(f"Request: {log.request_data.get('message', '')}")
                output.append(f"Response: {log.response_content}")
                output.append("-" * 50)
            return "\n".join(output)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def clear_session_logs(self, session_id: Optional[str] = None):
        """Clear logs for a specific session"""
        target_session = session_id or self.current_session_id
        if target_session in self.session_logs:
            del self.session_logs[target_session]
            self.logger.info(f"Cleared logs for session: {target_session}")


class ModelType:
    """Model type constants"""

    TEXT = "text"
    MULTIMODAL = "multimodal"
    THOUGHT = "thought"
    MULTIMODAL_THOUGHT = "multimodal_thought"


class MessageProcessor:
    """Handle message history processing and formatting"""

    @staticmethod
    async def build_message_history(
        message: str,
        history: List[Union[Dict, List, Tuple]],
        role_setting: Optional[str] = None,
        system_prompt: Optional[str] = None,
        is_multimodal: bool = False,
        file_input: Optional[List[str]] = None,  # Added file_input parameter
        chatbot_instance=None,  # Added to access _classifie_file_by_ext method
    ) -> List[Dict[str, Any]]:
        """
        Build standardized message history from various input formats

        Args:
            message: Current user message
            history: Conversation history in various formats
            role_setting: Role configuration
            system_prompt: System prompt
            is_multimodal: Whether this is for multimodal model
            file_input: List of file paths
            chatbot_instance: ChatBotGenerator instance for file classification

        Returns:
            Standardized message list
        """
        messages = []

        system_content = MessageProcessor._build_system_content(
            role_setting, system_prompt
        )
        if system_content:
            messages.append({"role": "system", "content": system_content})

        if history:
            messages.extend(MessageProcessor._parse_history(history, is_multimodal))

        if is_multimodal:
            user_content = MessageProcessor._parse_multimodal_content(
                message, file_input, chatbot_instance
            )
        else:
            user_content = message

        messages.append({"role": "user", "content": user_content})
        return messages

    @staticmethod
    def _build_system_content(
        role_setting: Optional[str], system_prompt: Optional[str]
    ) -> str:
        """Build system content from role setting and system prompt"""
        content_parts = []
        if role_setting:
            content_parts.append(f"你现在扮演: {role_setting}")
        if system_prompt:
            content_parts.append(system_prompt)
        return "".join(content_parts)

    @staticmethod
    def _parse_history(
        history: List[Union[Dict, List, Tuple]], is_multimodal: bool = False
    ) -> List[Dict[str, Any]]:
        """Parse various history formats into standardized format"""
        messages = []

        for entry in history:
            if isinstance(entry, dict) and "role" in entry:
                role = entry["role"]
                content = entry.get("content", "")
                if role in ["user", "assistant"]:
                    if is_multimodal and role == "user" and isinstance(content, str):
                        content = MessageProcessor._parse_multimodal_content(content)
                    messages.append({"role": role, "content": content})
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                user_msg, bot_msg = entry
                user_content = user_msg
                if is_multimodal:
                    user_content = MessageProcessor._parse_multimodal_content(user_msg)
                messages.append({"role": "user", "content": user_content})
                if bot_msg:
                    messages.append({"role": "assistant", "content": bot_msg})
            else:
                print(f"Warning: Unresolvable history format: {entry}")

        return messages

    @staticmethod
    def _parse_multimodal_content(
        message: str, file_input: Optional[List[str]] = None, chatbot_instance=None
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Parse message for multimodal content (local images, videos) and text

        Args:
            message: Text message
            file_input: List of local file paths
            chatbot_instance: ChatBotGenerator instance for file classification

        Returns:
            Either text string or list of content dictionaries
        """
        content_list = []

        # Process local file inputs if provided
        if file_input and chatbot_instance:
            classified_files = chatbot_instance._classifie_file_by_ext(file_input)

            # Add image files
            for img_path in classified_files.get("image_url", []):
                content_list.append(
                    {"type": "image_url", "image_url": {"url": img_path}}
                )

            # Add video files
            for vid_path in classified_files.get("video_url", []):
                content_list.append(
                    {"type": "video_url", "video_url": {"url": vid_path}}
                )

        # Add text content if there's any text
        if message.strip():
            content_list.append({"type": "text", "text": message.strip()})

        # Return content list if we have multimodal content, otherwise return plain text
        if (
            content_list
            and len(content_list) > 1
            or (content_list and content_list[0]["type"] != "text")
        ):
            return content_list

        # If only text content, return as simple string
        return message


class ResponseFormatter:
    """Format different types of responses"""

    @staticmethod
    def format_thought_response(thought_content: str, response_content: str) -> str:
        """Format response with thought process"""
        return (
            f"<details open><summary>思考过程</summary>\n"
            f"<div class='thought-container' style='font-size: 13px;opacity: 0.85;"
            f"padding-left:20px;border-left:3px solid #ddd;"
            f"margin-bottom: 1em;'>{thought_content}</div>\n"
            f"</details>\n"
            f"<div class='response-container' style='line-height: 1.5;'>{response_content}</div>"
        )


class BaseResponseGenerator(ABC):
    """Base class for response generators"""

    def __init__(self, client_factory, debug_logger: DebugLogger):
        self.client_factory = client_factory
        self.debug_logger = debug_logger
        self.stop_generation = False

    def stop(self):
        """Set stop flag to interrupt generation"""
        self.stop_generation = True

    def reset(self):
        """Reset stop flag"""
        self.stop_generation = False

    @abstractmethod
    async def generate_response(
        self, request: ChatRequest, chatbot_instance=None
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """Generate response for the given request"""
        pass

    async def _create_chat_completion(
        self,
        client,
        messages: List[Dict],
        request: ChatRequest,
        enable_thinking: bool = False,
    ):
        """Create chat completion with common parameters"""
        completion_params = {
            "model": request.model_name,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_length,
            "stream": True,
        }

        if enable_thinking:
            completion_params["chat_template_kwargs"] = {"enable_thinking": True}

        return client.chat.completions.create(**completion_params)

    def _create_history_with_response(
        self, request: ChatRequest
    ) -> Tuple[List[Dict], Dict]:
        """Create new history with user message and empty assistant response"""
        new_history = list(request.history) if request.history else []

        user_message = {"role": "user", "content": request.message}
        new_history.append(user_message)

        assistant_response = {"role": "assistant", "content": ""}
        new_history.append(assistant_response)

        return new_history, assistant_response

    def _create_debug_info(
        self, request: ChatRequest, messages: List[Dict], model_type: str
    ) -> DebugInfo:
        """Create debug info object"""
        return DebugInfo(
            timestamp=datetime.now().isoformat(),
            session_id=self.debug_logger.current_session_id or "unknown",
            model_type=model_type,
            request_data=asdict(request),
            processed_messages=messages,
            response_content="",
            thought_content="",
            generation_time=0.0,
            token_count=0,
        )


class TextResponseGenerator(BaseResponseGenerator):
    """Generator for text-only models"""

    async def generate_response(
        self, request: ChatRequest, chatbot_instance=None
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """Generate text response"""
        if not request.message:
            yield [], gr.update(value="")
            return

        self.reset()
        start_time = time.time()
        debug_info = None

        try:
            client = self.client_factory(request.port)
            messages = await MessageProcessor.build_message_history(
                request.message,
                request.history,
                request.role_setting,
                request.system_prompt,
                is_multimodal=False,
                file_input=request.file_input,
                chatbot_instance=chatbot_instance,
            )

            print("messages:", messages)

            # Create debug info
            debug_info = self._create_debug_info(request, messages, ModelType.TEXT)

            response = await self._create_chat_completion(client, messages, request)
            new_history, assistant_response = self._create_history_with_response(
                request
            )

            token_count = 0
            async for chunk in self._stream_response(response, assistant_response):
                if self.stop_generation:
                    break
                token_count += 1
                yield new_history, gr.update(value="")
                await asyncio.sleep(0.01)

            # Update debug info
            debug_info.response_content = assistant_response["content"]
            debug_info.generation_time = time.time() - start_time
            debug_info.token_count = token_count

            print("new_history:", new_history)

            yield new_history, gr.update(value="")

        except Exception as e:
            error_result = await self._handle_error(e, request.message)
            if debug_info:
                debug_info.error_info = str(e)
                debug_info.generation_time = time.time() - start_time
            yield error_result
        finally:
            if debug_info:
                self.debug_logger.log_debug_info(debug_info)
            self.reset()

    async def _stream_response(self, response, assistant_response):
        """Stream response content"""
        for chunk in response:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                assistant_response["content"] += chunk.choices[0].delta.content
                yield chunk

    async def _handle_error(self, error: Exception, message: str):
        """Handle API errors"""
        print(f"Text response error: {error}")
        error_msg = {"role": "assistant", "content": f"API调用失败: {error!s}"}
        return [{"role": "user", "content": message}, error_msg], gr.update(value="")


class MultimodalResponseGenerator(BaseResponseGenerator):
    """Generator for multimodal models"""

    async def generate_response(
        self, request: ChatRequest, chatbot_instance=None
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """Generate multimodal response"""
        if not request.message:
            yield [], gr.update(value="")
            return

        self.reset()
        start_time = time.time()
        debug_info = None

        try:
            client = self.client_factory(request.port)
            messages = await MessageProcessor.build_message_history(
                request.message,
                request.history,
                request.role_setting,
                request.system_prompt,
                is_multimodal=True,
                file_input=request.file_input,
                chatbot_instance=chatbot_instance,
            )

            # Create debug info
            debug_info = self._create_debug_info(
                request, messages, ModelType.MULTIMODAL
            )

            response = await self._create_chat_completion(client, messages, request)
            new_history, assistant_response = self._create_history_with_response(
                request
            )

            token_count = 0
            async for chunk in self._stream_response(response, assistant_response):
                if self.stop_generation:
                    break
                token_count += 1
                yield new_history, gr.update(value="")
                await asyncio.sleep(0.01)

            # Update debug info
            debug_info.response_content = assistant_response["content"]
            debug_info.generation_time = time.time() - start_time
            debug_info.token_count = token_count

            yield new_history, gr.update(value="")

        except Exception as e:
            error_result = await self._handle_error(e, request.message)
            if debug_info:
                debug_info.error_info = str(e)
                debug_info.generation_time = time.time() - start_time
            yield error_result
        finally:
            if debug_info:
                self.debug_logger.log_debug_info(debug_info)
            self.reset()

    async def _stream_response(self, response, assistant_response):
        """Stream multimodal response content"""
        for chunk in response:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                assistant_response["content"] += chunk.choices[0].delta.content
                yield chunk

    async def _handle_error(self, error: Exception, message: str):
        """Handle multimodal API errors"""
        print(f"Multimodal response error: {error}")
        error_msg = {"role": "assistant", "content": f"多模态API调用失败: {error!s}"}
        return [{"role": "user", "content": message}, error_msg], gr.update(value="")


class ThoughtResponseGenerator(BaseResponseGenerator):
    """Generator for models with thought process"""

    async def generate_response(
        self, request: ChatRequest, chatbot_instance=None
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """Generate response with thought process"""
        if not request.message:
            yield [], gr.update(value="")
            return

        self.reset()
        start_time = time.time()
        debug_info = None

        try:
            client = self.client_factory(request.port)
            messages = await MessageProcessor.build_message_history(
                request.message,
                request.history,
                request.role_setting,
                request.system_prompt,
                is_multimodal=False,
                file_input=request.file_input,
                chatbot_instance=chatbot_instance,
            )

            # Create debug info
            debug_info = self._create_debug_info(request, messages, ModelType.THOUGHT)

            response = await self._create_chat_completion(
                client, messages, request, enable_thinking=True
            )
            new_history, assistant_response = self._create_history_with_response(
                request
            )

            current_thought = ""
            current_response = ""
            token_count = 0

            for chunk in response:
                if self.stop_generation:
                    break

                if chunk.choices[0].delta:
                    # Extract thought and response content
                    thought_part = (
                        getattr(chunk.choices[0].delta, "reasoning_content", "") or ""
                    )
                    answer_part = getattr(chunk.choices[0].delta, "content", "") or ""

                    current_thought += thought_part
                    current_response += answer_part
                    token_count += 1

                    # Format response with thought process
                    formatted_response = ResponseFormatter.format_thought_response(
                        current_thought, current_response
                    )

                    assistant_response["content"] = formatted_response
                    yield new_history, gr.update(value="")
                    await asyncio.sleep(0.01)

            # Update debug info
            debug_info.thought_content = current_thought
            debug_info.response_content = current_response
            debug_info.generation_time = time.time() - start_time
            debug_info.token_count = token_count

            yield new_history, gr.update(value="")

        except Exception as e:
            error_result = await self._handle_error(e, request.message)
            if debug_info:
                debug_info.error_info = str(e)
                debug_info.generation_time = time.time() - start_time
            yield error_result
        finally:
            if debug_info:
                self.debug_logger.log_debug_info(debug_info)
            self.reset()

    async def _handle_error(self, error: Exception, message: str):
        """Handle thought process API errors"""
        print(f"Thought response error: {error}")
        error_msg = {"role": "assistant", "content": f"思考过程生成失败: {error!s}"}
        return [{"role": "user", "content": message}, error_msg], gr.update(value="")


class MultimodalThoughtResponseGenerator(BaseResponseGenerator):
    """Generator for multimodal models with thought process"""

    async def generate_response(
        self, request: ChatRequest, chatbot_instance=None
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """Generate multimodal response with thought process"""
        if not request.message:
            yield [], gr.update(value="")
            return

        self.reset()
        start_time = time.time()
        debug_info = None

        try:
            client = self.client_factory(request.port)
            messages = await MessageProcessor.build_message_history(
                request.message,
                request.history,
                request.role_setting,
                request.system_prompt,
                is_multimodal=True,
                file_input=request.file_input,
                chatbot_instance=chatbot_instance,
            )

            # Create debug info
            debug_info = self._create_debug_info(
                request, messages, ModelType.MULTIMODAL_THOUGHT
            )

            response = await self._create_chat_completion(
                client, messages, request, enable_thinking=True
            )
            new_history, assistant_response = self._create_history_with_response(
                request
            )

            current_thought = ""
            current_response = ""
            token_count = 0

            for chunk in response:
                if self.stop_generation:
                    break

                if chunk.choices[0].delta:
                    # Extract thought and response content
                    thought_part = (
                        getattr(chunk.choices[0].delta, "reasoning_content", "") or ""
                    )
                    answer_part = getattr(chunk.choices[0].delta, "content", "") or ""

                    current_thought += thought_part
                    current_response += answer_part
                    token_count += 1

                    # Format multimodal response with thought process
                    formatted_response = ResponseFormatter.format_thought_response(
                        current_thought, current_response
                    )

                    assistant_response["content"] = formatted_response
                    yield new_history, gr.update(value="")
                    await asyncio.sleep(0.01)

            # Update debug info
            debug_info.thought_content = current_thought
            debug_info.response_content = current_response
            debug_info.generation_time = time.time() - start_time
            debug_info.token_count = token_count

            yield new_history, gr.update(value="")

        except Exception as e:
            error_result = await self._handle_error(e, request.message)
            if debug_info:
                debug_info.error_info = str(e)
                debug_info.generation_time = time.time() - start_time
            yield error_result
        finally:
            if debug_info:
                self.debug_logger.log_debug_info(debug_info)
            self.reset()

    async def _handle_error(self, error: Exception, message: str):
        """Handle multimodal thought API errors"""
        print(f"Multimodal thought response error: {error}")
        error_msg = {
            "role": "assistant",
            "content": f"多模态思考过程生成失败: {error!s}",
        }
        return [{"role": "user", "content": message}, error_msg], gr.update(value="")


class ChatBotGenerator:
    """
    Main ChatBot class supporting multiple model types with debug capabilities:
    - Text models
    - Multimodal models
    - Thought models
    - Multimodal thought models
    """

    def __init__(
        self, debug_enabled: bool = False, debug_session_id: Optional[str] = None
    ):
        self.default_ip = "0.0.0.0"
        self.debug_logger = DebugLogger()

        if debug_enabled:
            self.enable_debug(debug_session_id)

        self.generators = self._initialize_generators()

    def _initialize_generators(self) -> Dict[str, BaseResponseGenerator]:
        """Initialize response generators for different model types"""
        client_factory = self._create_openai_client

        return {
            ModelType.TEXT: TextResponseGenerator(client_factory, self.debug_logger),
            ModelType.MULTIMODAL: MultimodalResponseGenerator(
                client_factory, self.debug_logger
            ),
            ModelType.THOUGHT: ThoughtResponseGenerator(
                client_factory, self.debug_logger
            ),
            ModelType.MULTIMODAL_THOUGHT: MultimodalThoughtResponseGenerator(
                client_factory, self.debug_logger
            ),
        }

    def _create_openai_client(self, port: int) -> openai.Client:
        """Create OpenAI client connection"""
        base_url = f"http://{self.default_ip}:{port}/v1"
        return openai.Client(base_url=base_url, api_key="EMPTY_API_KEY")

    def _classifie_file_by_ext(self, file_input: List) -> Dict:
        classified = {"image_url": [], "video_url": [], "non_type": []}

        for file_path in file_input:
            # 获取文件后缀（转为小写以便不区分大小写）
            ext = ""
            if "." in file_path:
                ext = file_path.split(".")[-1].lower()
                ext = f".{ext}"  # 确保后缀以点开头

            # 根据后缀分类
            if config._is_image_file(ext):
                classified["image_url"].append(file_path)
            elif config._is_video_file(ext):
                classified["video_url"].append(file_path)
            else:
                classified["non_type"].append(file_path)

        return classified

    # Debug methods
    def enable_debug(self, session_id: Optional[str] = None):
        """Enable debug mode"""
        self.debug_logger.enable_debug(session_id)

    def disable_debug(self):
        """Disable debug mode"""
        self.debug_logger.disable_debug()

    def get_debug_logs(self, session_id: Optional[str] = None) -> List[DebugInfo]:
        """Get debug logs for a session"""
        return self.debug_logger.get_session_logs(session_id)

    def export_debug_logs(
        self, session_id: Optional[str] = None, format: str = "json"
    ) -> str:
        """Export debug logs"""
        return self.debug_logger.export_session_logs(session_id, format)

    def clear_debug_logs(self, session_id: Optional[str] = None):
        """Clear debug logs"""
        self.debug_logger.clear_session_logs(session_id)

    def stop(self):
        """Stop all generators"""
        for generator in self.generators.values():
            generator.stop()

    def reset(self):
        """Reset all generators"""
        for generator in self.generators.values():
            generator.reset()

    def _determine_model_type(
        self, model_name: str, enable_thought: bool = False
    ) -> str:
        """Determine model type based on model name and configuration"""
        is_multimodal = config.is_vl_models(model_name)
        is_thought_capable = (
            True if enable_thought else config.is_thought_model(model_name)
        )

        if is_multimodal and is_thought_capable:
            return ModelType.MULTIMODAL_THOUGHT
        elif is_multimodal:
            return ModelType.MULTIMODAL
        elif is_thought_capable:
            return ModelType.THOUGHT
        else:
            return ModelType.TEXT

    # Legacy compatibility methods
    async def text_response(
        self,
        message: str,
        history: List[Dict[str, str]],
        role_setting: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_length: int = 1000,
        top_p: float = 0.8,
        temperature: float = 0.7,
        port: int = 8188,
        file_input: Optional[List[str]] = None,  # Added file_input parameter
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """Generate text-only response (legacy compatibility)"""
        request = ChatRequest(
            message=message,
            history=history,
            role_setting=role_setting,
            system_prompt=system_prompt,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            port=port,
            file_input=file_input,
        )

        generator = self.generators[ModelType.TEXT]
        async for result in generator.generate_response(request, self):
            yield result

    async def multimodal_response(
        self,
        message: str,
        history: List[Dict[str, str]],
        role_setting: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_length: int = 1000,
        top_p: float = 0.8,
        temperature: float = 0.7,
        port: int = 8188,
        file_input: Optional[List[str]] = None,  # Added file_input parameter
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """Generate multimodal response (legacy compatibility)"""
        request = ChatRequest(
            message=message,
            history=history,
            role_setting=role_setting,
            system_prompt=system_prompt,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            port=port,
            file_input=file_input,
        )

        generator = self.generators[ModelType.MULTIMODAL]
        async for result in generator.generate_response(request, self):
            yield result

    async def thought_response(
        self,
        message: str,
        history: List[Dict[str, str]],
        role_setting: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_length: int = 1000,
        top_p: float = 0.8,
        temperature: float = 0.7,
        port: int = 8188,
        file_input: Optional[List[str]] = None,  # Added file_input parameter
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """Generate response with thought process (legacy compatibility)"""
        request = ChatRequest(
            message=message,
            history=history,
            role_setting=role_setting,
            system_prompt=system_prompt,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            port=port,
            enable_thinking=True,
            file_input=file_input,
        )

        generator = self.generators[ModelType.THOUGHT]
        async for result in generator.generate_response(request, self):
            yield result

    async def generate_response(
        self,
        message: str,
        history: List[Dict[str, str]],
        model_name: str,
        enable_thought: bool = False,
        role_setting: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_length: int = 1000,
        top_p: float = 0.8,
        temperature: float = 0.7,
        port: int = 8188,
        file_input: Optional[List[str]] = None,  # Added file_input parameter
    ) -> AsyncGenerator[Tuple[List[Dict], gr.update], None]:
        """
        Unified interface to generate response based on model capabilities

        Args:
            message: User message (can contain image/video URLs for multimodal models)
            history: Conversation history
            model_name: Name of the model to use
            enable_thought: Whether to enable thought process
            role_setting: Role configuration
            system_prompt: System prompt
            max_length: Maximum response length
            top_p: Nucleus sampling probability
            temperature: Sampling temperature
            port: Service port
            file_input: List of file paths for multimodal content
        """
        request = ChatRequest(
            message=message,
            model_name=model_name,
            history=history,
            role_setting=role_setting,
            system_prompt=system_prompt,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            port=port,
            enable_thinking=enable_thought,
            file_input=file_input,
        )

        print("chat_request:", request, "\n")

        model_type = self._determine_model_type(model_name, enable_thought)
        generator = self.generators[model_type]

        print("chat_generator:", generator, "\n")
        print("model_type:", model_type, "\n")

        async for result in generator.generate_response(request, self):
            yield result


chatbot = ChatBotGenerator(debug_enabled=True, debug_session_id="default_session")


# Example usage functions for debug mode
def example_debug_usage():
    """Example of how to use debug mode"""

    # Enable debug mode
    chatbot.enable_debug("my_session_001")

    print("Debug mode enabled. Now all chat interactions will be logged.")

    # After some chat interactions, you can:

    # 1. Get debug logs
    logs = chatbot.get_debug_logs()
    print(f"Found {len(logs)} debug entries")

    # 2. Export logs as JSON
    json_logs = chatbot.export_debug_logs(format="json")
    with open("debug_logs.json", "w", encoding="utf-8") as f:
        f.write(json_logs)

    # 3. Export logs as text
    txt_logs = chatbot.export_debug_logs(format="txt")
    with open("debug_logs.txt", "w", encoding="utf-8") as f:
        f.write(txt_logs)

    # 4. Clear logs
    # chatbot.clear_debug_logs()

    # 5. Disable debug mode
    # chatbot.disable_debug()


def create_debug_chatbot(session_id: Optional[str] = None) -> ChatBotGenerator:
    """
    Create a new chatbot instance with debug mode enabled

    Args:
        session_id: Optional session identifier for organizing logs

    Returns:
        ChatBotGenerator instance with debug enabled
    """
    return ChatBotGenerator(debug_enabled=True, debug_session_id=session_id)


# Debug utility functions
def analyze_debug_logs(
    chatbot_instance: ChatBotGenerator, session_id: Optional[str] = None
):
    """
    Analyze debug logs and provide insights

    Args:
        chatbot_instance: The chatbot instance to analyze
        session_id: Session to analyze (None for current session)
    """
    logs = chatbot_instance.get_debug_logs(session_id)

    if not logs:
        print("No debug logs found.")
        return

    print("\n📊 DEBUG LOG ANALYSIS")
    print("=" * 50)
    print(f"Total conversations: {len(logs)}")

    # Model type statistics
    model_types = {}
    total_time = 0
    total_tokens = 0

    for log in logs:
        model_type = log.model_type
        model_types[model_type] = model_types.get(model_type, 0) + 1
        total_time += log.generation_time
        total_tokens += log.token_count

    print("\nModel type usage:")
    for model_type, count in model_types.items():
        print(f"  - {model_type}: {count} times")

    print("\nPerformance metrics:")
    print(f"  - Total generation time: {total_time:.2f}s")
    print(f"  - Average time per request: {total_time / len(logs):.2f}s")
    print(f"  - Total tokens generated: {total_tokens}")
    print(f"  - Average tokens per request: {total_tokens / len(logs):.0f}")

    # Error analysis
    errors = [log for log in logs if log.error_info]
    if errors:
        print(f"\nErrors found: {len(errors)}")
        for i, error_log in enumerate(errors, 1):
            print(f"  {i}. {error_log.timestamp}: {error_log.error_info}")
    else:
        print(f"\nNo errors found in {len(logs)} requests.")


def export_debug_summary(
    chatbot_instance: ChatBotGenerator, session_id: Optional[str] = None
) -> str:
    """
    Export a summary of debug information

    Args:
        chatbot_instance: The chatbot instance
        session_id: Session to export (None for current session)

    Returns:
        Summary string
    """
    logs = chatbot_instance.get_debug_logs(session_id)

    if not logs:
        return "No debug logs found."

    summary = []
    summary.append("# Debug Session Summary")
    summary.append(f"Session ID: {logs[0].session_id}")
    summary.append(f"Total Requests: {len(logs)}")
    summary.append(f"Time Range: {logs[0].timestamp} to {logs[-1].timestamp}")

    # Performance summary
    total_time = sum(log.generation_time for log in logs)
    total_tokens = sum(log.token_count for log in logs)

    summary.append("\n## Performance")
    summary.append(f"- Total Generation Time: {total_time:.2f}s")
    summary.append(f"- Average Response Time: {total_time / len(logs):.2f}s")
    summary.append(f"- Total Tokens: {total_tokens}")
    summary.append(f"- Average Tokens per Request: {total_tokens / len(logs):.0f}")

    # Model usage
    model_usage = {}
    for log in logs:
        model_usage[log.model_type] = model_usage.get(log.model_type, 0) + 1

    summary.append("\n## Model Usage")
    for model_type, count in model_usage.items():
        summary.append(f"- {model_type}: {count} requests")

    # Recent requests
    summary.append("\n## Recent Requests (Last 5)")
    for log in logs[-5:]:
        req_msg = log.request_data.get("message", "")[:50]
        if len(log.request_data.get("message", "")) > 50:
            req_msg += "..."
        summary.append(f"- {log.timestamp}: {req_msg}")

    return "\n".join(summary)
