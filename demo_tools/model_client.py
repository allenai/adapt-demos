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
    def __init__(self, model, api_key, port, model_name=None, debug=False, stream=True):
        self.model_url = f"http://localhost:{port}/v1"
        self.model = model
        self.model_name = model_name or model
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

    def predict(
        self,
        message,
        history,
        system_prompt,
        temperature,
        safety_filter_checkbox,
        refusal_rewrite_text,
        completion_mode,
    ):
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
            if len(system_prompt) > 0:
                history_openai_format.append({"role": "system", "content": system_prompt})
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

    def predict_safety(
        self, message, history, system_prompt, temperature, safety_filter_checkbox, refusal_rewrite_text
    ):
        if not safety_filter_checkbox:
            return "Safety filter not enabled", ""

        history_openai_format = []
        if len(system_prompt) > 0:
            history_openai_format.append({"role": "system", "content": system_prompt})
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

        safety_label_styles = {"default": {"yes": "warning", "no": "success"}}
        if any(k for k, v in safety_labels if k.lower().startswith("response refusal")):
            refusal_index = None
            for i, (k, v) in enumerate(safety_labels):
                if k.lower().startswith("response refusal"):
                    safety_label_styles[k] = {"yes": "info", "no": "secondary"}
                    refusal_index = i
            if refusal_index is not None:
                safety_labels.append(safety_labels.pop(refusal_index))

        safety_labels_html = "\n<br/>\n".join(
            [
                f"<span style='color: black'>{key}</span>&nbsp;"
                f"<span class='badge text-bg-{safety_label_styles.get(key, safety_label_styles['default'])[label.lower()]}'>"  # noqa
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
            refusal_rewrite_text = refusal_rewrite_text or MAKE_SAFE_PROMPT

            refusal_rewrite_kwargs = {}
            if "{prompt}" in refusal_rewrite_text:
                refusal_rewrite_kwargs["prompt"] = last_query
            if "{response}" in refusal_rewrite_text:
                refusal_rewrite_kwargs["response"] = last_response

            if not refusal_rewrite_kwargs:
                logger.warning(
                    "Make safe prompt template does not include user input ({prompt}) or assistant response ({response})"  # noqa
                )
            make_response_safe_input = refusal_rewrite_text.format(**refusal_rewrite_kwargs)
            if self.debug:
                logger.info(" --- MAKE SAFE PROMPT ---")
                logger.info(make_response_safe_input)
                logger.info(" ---")
            make_response_safe_openai_format = history_openai_format + [
                {"role": "user", "content": make_response_safe_input}
            ]

            response = self.response_client.model_client.chat.completions.create(
                model=self.response_client.model,
                messages=make_response_safe_openai_format,
                temperature=temperature,
            )

            if not response.choices[0].message.content.strip():
                logger.warning("Refusal rewrite response is empty")

            safe_response = f"""<div class="card white-background" style='background-color: white; padding: 10px;>
                        <h4 class="card-title safe-title">Safe Response</h4>
                        <div class="card-body safe-text">{response.choices[0].message.content}
                        </div>
                </div>"""
        else:
            safe_response = "<p style='color: black'>Assistant's response is safe</p>"
            if self.debug:
                safe_response = "NOTE: FILTER OFF IN DEBUG MODE.\n"

        # modify the responses with html for white background
        safety_labels_html = f"<div style='background-color: white; color: black; padding: 10px;'>{safety_labels_html}</div>"
        safe_response = f"<div style='background-color: white; padding: 10px;>{safe_response}</div>"

        return HTML(safety_labels_html), HTML(safe_response)
