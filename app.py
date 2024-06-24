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

import argparse

import gradio as gr
from openai import OpenAI

from src.dummy_chatbot import MockOpenAI, MockOpenAIStream
from src.interface import SafetyChatInterface
from src.prompts import WILDGUARD_INPUT_FORMAT

# Define an argument parser
parser = argparse.ArgumentParser(description="Gradio App with Custom OpenAI API Port")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode (does not ping models)")
parser.add_argument("--port", type=int, default=8000, help="Port to connect to OpenAI API server")
parser.add_argument(
    "--safety_filter_port", type=int, required=False, default=None, help="Port to connect to safety filter server"
)
parser.add_argument("--model", type=str, required=True, help="Model to connect to")
parser.add_argument("--safety_model", type=str, required=False, help="Safety model to connect to")
parser.add_argument("--completion_mode", action="store_true", default=False, help="Use completion mode for OpenAI API")
args = parser.parse_args()

# OpenAI configuration
api_key = "EMPTY"  # OpenAI API key (empty for custom server)
model_url = f"http://localhost:{args.port}/v1"  # Construct base URL with provided port

if args.debug:
    # Use mock client for debugging
    model_client = MockOpenAIStream()
else:
    model_client = OpenAI(api_key=api_key, base_url=model_url)

if args.safety_filter_port or args.safety_model:
    # if one of them, both need to be set
    if not args.safety_filter_port or not args.safety_model:
        raise ValueError("Both safety filter port and safety model need to be set")
    safety_url = f"http://localhost:{args.safety_filter_port}/v1"  # Construct base URL with provided port
    if args.debug:
        # Use mock client for debugging
        safety_client = MockOpenAI()
    else:
        safety_client = OpenAI(api_key=api_key, base_url=safety_url)
    SAFETY_FILTER_ON = True
else:
    SAFETY_FILTER_ON = False


# Prediction function for Gradio
def predict(message, history, temperature, safety_filter_checkbox):
    # Create completion with OpenAI client
    if args.completion_mode:
        response = model_client.chat.completions.create(
            model=args.model, messages=message, temperature=temperature, stream=True
        )

        # Generate partial message based on streamed response
        partial_message = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                partial_message = partial_message + chunk.choices[0].delta.content
                yield partial_message

    # Use chat API
    else:
        history_openai_format = []
        for human, assistant in history:
            history_openai_format.append({"role": "user", "content": human})
            history_openai_format.append({"role": "assistant", "content": assistant})

        # Add the latest message to the history
        history_openai_format.append({"role": "user", "content": message})
        response = model_client.chat.completions.create(
            model=args.model,
            messages=history_openai_format,
            temperature=temperature,
            stream=True,
        )
        # Generate partial message based on streamed response
        partial_message = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                partial_message = partial_message + chunk.choices[0].delta.content
                yield partial_message


if SAFETY_FILTER_ON:

    def run_safety_filter(message, history, temperature, safety_filter_checkbox):
        if not safety_filter_checkbox:
            return "Safety filter not enabled"
        history_openai_format = []
        for human, assistant in history[:-1]:
            history_openai_format.append({"role": "user", "content": human})
            history_openai_format.append({"role": "assistant", "content": assistant})

        last_query, last_response = history[-1]
        safety_formatted_input = WILDGUARD_INPUT_FORMAT.format(prompt=last_query, response=last_response)
        safety_history_openai_format = history_openai_format + [{"role": "user", "content": safety_formatted_input}]
        safety_response = safety_client.chat.completions.create(
            model=args.safety_model,
            messages=safety_history_openai_format,
            temperature=temperature,
            stream=False,
        )
        return """### Safety info: \n""" + safety_response.choices[0].message.content.replace("yes", "yes\n").replace(
            "no", "no\n"
        )

else:
    # placeholder for when off
    def run_safety_filter(message, history, temperature, safety_filter_checkbox):
        return "Safety filter not enabled"


# Launch Gradio app
temperature_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label="Temperature")
safety_filter_checkbox = gr.Checkbox(label="Run Safety Filter", value=SAFETY_FILTER_ON)

demo = SafetyChatInterface(
    predict,
    run_safety_filter,
    additional_inputs=[temperature_slider, safety_filter_checkbox],
    title="AI2 Internal Demo Model",
    description=f"""Model: {args.model}

                            Safety Model: {args.safety_model}""",
)

demo.queue().launch(share=True)
