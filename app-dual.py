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
import logging

import gradio as gr

from demo_tools import (
    ModelClientHandler,
    SafetyClientHandler,
    EnhancedChatInterface,
    run_dummy_safety_filter,
)
from demo_tools.prompts import MAKE_SAFE_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define an argument parser
parser = argparse.ArgumentParser(description="Gradio App with Custom OpenAI API Port")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode (does not ping models)")
parser.add_argument("--port_one", type=int, default=8000, help="Port to connect to OpenAI API server")
parser.add_argument("--port_two", type=int, required=True, default=8001, help="Port to connect to second inference server")
parser.add_argument("--model_one", type=str, required=True, help="Model to connect to")
parser.add_argument("--model_two", type=str, required=False, help="Second model")
parser.add_argument("--completion_mode", action="store_true", default=False, help="Use completion mode for OpenAI API")
parser.add_argument(
    "--safety_filter_port", type=int, required=False, default=None, help="Port to connect to safety filter server"
)
parser.add_argument("--safety_model", type=str, required=False, help="Safety model to connect to")
args = parser.parse_args()

# OpenAI configuration
api_key = "EMPTY"  # OpenAI API key (empty for custom server)

model_client = ModelClientHandler(args.model_one, api_key, args.port_one, debug=args.debug, stream=True)
model_client_2 = ModelClientHandler(args.model_two, api_key, args.port_two, debug=args.debug, stream=True)

# Launch Gradio app
temperature_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label="Temperature")
additional_inputs = [temperature_slider]

if args.safety_filter_port or args.safety_model:
    # if one of them, both need to be set
    if not args.safety_filter_port or not args.safety_model:
        raise ValueError("Both safety filter port and safety model need to be set")

    safety_client = SafetyClientHandler(
        args.safety_model, api_key, args.safety_filter_port, model_client, debug=args.debug, stream=False
    )
    safety_client_2 = SafetyClientHandler(
        args.safety_model, api_key, args.safety_filter_port, model_client_2, debug=args.debug, stream=False
    )
    SAFETY_FILTER_ON = True

    safety_filter_checkbox = gr.Checkbox(label="Run Safety Filter", value=SAFETY_FILTER_ON)
    reprompt_textarea = gr.TextArea(
        label="Prompt to make assistant safe if detected unsafe. Use placeholder {prompt} for user input and {response} for assistant response.",  # noqa
        value=MAKE_SAFE_PROMPT,
        lines=12,
    )
    additional_inputs += [safety_filter_checkbox, reprompt_textarea]
    logger.info(f"Safety filter: ON, connecting to {safety_client.model_url}")
else:
    SAFETY_FILTER_ON = False
    logger.info("Safety filter: OFF")

# Custom style code
js_url = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
css_url = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
header = f"""
<link rel="stylesheet" href="{css_url}">
<script src="{js_url}" crossorigin="anonymous"></script>
"""

css = """
.classifier-text {
    font-size: 20px !important;
}
.safe-text {
    font-size: 16px !important;
    color: white;
}
.safe-title {
    color: white;
}
"""

demo = EnhancedChatInterface(
    fn=model_client.predict,
    safety_fn=safety_client.predict_safety if SAFETY_FILTER_ON else run_dummy_safety_filter,
    model_client=model_client,
    fn_2=model_client_2.predict,
    safety_fn_2=safety_client_2.predict_safety if SAFETY_FILTER_ON else run_dummy_safety_filter,
    model_client_2=model_client_2,
    additional_inputs=additional_inputs,
    head=header,
    css=css,
    title="AI2 Internal Demo Model",
    description=f"""Model 1 (left): {args.model_one}

                            Model 2 (right): {args.model_two}""",
)

demo.queue().launch(share=True)
