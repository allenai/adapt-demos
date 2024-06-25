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

from .dummy_chatbot import MockOpenAI, MockOpenAIStream
from .prompts import WILDGUARD_INPUT_FORMAT


def run_dummy_safety_filter(message, history, temperature, safety_filter_checkbox):
    return "Safety filter not enabled"


class ModelClientHandler:
    def __init__(self, model, api_key, port, debug=False, stream=True):
        self.model_url = f"http://localhost:{port}/v1"
        self.model = model
        self.debug = debug
        if debug:
            if stream:
                # Use a mock client when debugging
                self.model_client = MockOpenAIStream()
            else:
                self.model_client = MockOpenAI()
        else:
            # Use a real OpenAI client otherwise
            self.model_client = OpenAI(api_key=api_key, base_url=self.model_url)

    def predict(self, message, history, temperature, safety_filter_checkbox, completion_mode):
        if completion_mode:
            # Streamed completions for interactive mode
            response = self.model_client.chat.completions.create(
                model=self.model, messages=message, temperature=temperature, stream=True
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
                model=self.model,
                messages=history_openai_format,
                temperature=temperature,
                stream=True,
            )

            partial_message = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    partial_message += chunk.choices[0].delta.content
                    yield partial_message

    def predict_safety(self, message, history, temperature, safety_filter_checkbox):
        if not safety_filter_checkbox:
            return "Safety filter not enabled"
        history_openai_format = []
        for human, assistant in history[:-1]:
            history_openai_format.append({"role": "user", "content": human})
            history_openai_format.append({"role": "assistant", "content": assistant})

        last_query, last_response = history[-1]
        safety_formatted_input = WILDGUARD_INPUT_FORMAT.format(prompt=last_query, response=last_response)
        safety_history_openai_format = history_openai_format + [{"role": "user", "content": safety_formatted_input}]
        safety_response = self.model_client.chat.completions.create(
            model=self.model,
            messages=safety_history_openai_format,
            temperature=temperature,
            stream=False,
        )
        return """### Safety info: \n""" + safety_response.choices[0].message.content.replace("yes", "yes\n").replace(
            "no", "no\n"
        )
