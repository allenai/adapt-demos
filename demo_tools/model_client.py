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

import logging
from collections import OrderedDict

from gradio.components import HTML
from openai import OpenAI

from .dummy_chatbot import MockOpenAI, MockOpenAIStream
from .prompts import MAKE_SAFE_PROMPT, WILDGUARD_INPUT_FORMAT

logger = logging.getLogger(__name__)


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

    def predict(self, message, history, temperature, safety_filter_checkbox, reprompt_text, completion_mode):
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


class SafetyClientHandler(ModelClientHandler):
    def __init__(self, model, api_key, port, response_client: ModelClientHandler, debug=False, stream=True):
        super().__init__(model, api_key, port, debug, stream)
        self.response_client = response_client

    def predict_safety(self, message, history, temperature, safety_filter_checkbox, reprompt_text):
        if not safety_filter_checkbox:
            return "Safety filter not enabled", ""

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

        safety_labels = [
            [s.strip() for s in label.split(":")]
            for label in safety_response.choices[0].message.content.split("\n")
            if label.strip() and len(label.split(":")) > 1
        ]

        safety_unwanted_labels = ["yes"] * len(safety_labels)
        if any(k for k, v in safety_labels if k.lower().startswith("response refusal")):
            for i, (k, v) in enumerate(safety_labels):
                if k.lower().startswith("response refusal"):
                    safety_unwanted_labels[i] = "no"

        safety_labels_html = "\n<br/>\n".join(
            [
                f"{key} <span class='badge text-bg-{'warning' if label.lower() == safety_unwanted_labels[i] else 'success'}'>"  # noqa
                f"{label.capitalize()}"
                f"</span>"
                for i, (key, label) in enumerate(safety_labels)
            ]
        )
        safety_labels_html = f"<div class='classifier-text'>{safety_labels_html}</div>"

        safety_labels = OrderedDict(safety_labels)
        if not safety_labels or "Response refusal" not in safety_labels:
            logger.error(f"Safety class response cannot be parsed: " f"[{safety_response.choices[0].message.content}]")
            safety_labels_html = "<p class='text-danger'>Safety response cannot be parsed, please try again</p>"
            safe_response = ""
        elif (
            safety_labels[next(iter(safety_labels))].lower() == "yes"
            and safety_labels["Response refusal"].lower() == "no"
        ):
            reprompt_text = reprompt_text or MAKE_SAFE_PROMPT

            reprompt_kwargs = {}
            if "{prompt}" in reprompt_text:
                reprompt_kwargs["prompt"] = last_query
            if "{response}" in reprompt_text:
                reprompt_kwargs["response"] = last_response

            if not reprompt_kwargs:
                logger.warning(
                    "Make safe prompt template does not include user input ({prompt}) or assistant response ({response})"  # noqa
                )
            make_response_safe_input = reprompt_text.format(**reprompt_kwargs)
            logger.debug(" --- MAKE SAFE PROMPT ---")
            logger.debug(make_response_safe_input)
            logger.debug(" ---")
            make_response_safe_openai_format = history_openai_format + [
                {"role": "user", "content": make_response_safe_input}
            ]

            response = self.response_client.model_client.chat.completions.create(
                model=self.response_client.model,
                messages=make_response_safe_openai_format,
                temperature=temperature,
            )

            safe_response = HTML(
                f"""<div class="card text-bg-success">
                        <h4 class="card-title safe-title">Safe Response</h4>
                        <div class="card-body safe-text">{response.choices[0].message.content}
                        </div>
                </div>"""
            )
        else:
            safe_response = "Assistant's response is safe"

        return HTML(safety_labels_html), safe_response
