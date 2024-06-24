# Copyright 2024 AllenAI. All rights reserved.
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

from openai import OpenAI
from .dummy_chatbot import MockOpenAIStream

class ModelClientHandler:
    def __init__(self, api_key, port, debug=False):
        self.model_url = f"http://localhost:{port}/v1"
        if debug:
            # Use a mock client when debugging
            self.model_client = MockOpenAIStream()
        else:
            # Use a real OpenAI client otherwise
            self.model_client = OpenAI(api_key=api_key, base_url=self.model_url)

    def predict(self, message, history, temperature, safety_filter_checkbox, completion_mode, model):
        if completion_mode:
            # Streamed completions for interactive mode
            response = self.model_client.chat.completions.create(
                model=model, messages=message, temperature=temperature, stream=True
            )

            partial_message = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    partial_message += chunk.choices[0].delta.content
                    yield partial_message
        else:
            # History format for chat-like interaction
            history_openai_format = []
            for human, assistant in history:
                history_openai_format.append({"role": "user", "content": human})
                history_openai_format.append({"role": "assistant", "content": assistant})

            history_openai_format.append({"role": "user", "content": message})
            response = self.model_client.chat.completions.create(
                model=model,
                messages=history_openai_format,
                temperature=temperature,
                stream=True,
            )

            partial_message = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    partial_message += chunk.choices[0].delta.content
                    yield partial_message
