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

from demo_tools import (
    css_style,
    theme,
    ModelClientHandler,
    EnhancedChatInterface,
    run_dummy_safety_filter,
)

# Define an argument parser
parser = argparse.ArgumentParser(description="Gradio App with Custom OpenAI API Port")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode (does not ping models)")
parser.add_argument("--port_one", type=int, default=8000, help="Port to connect to OpenAI API server")
parser.add_argument("--port_two", type=int, required=True, default=8001, help="Port to connect to second inference server")
parser.add_argument("--model_one", type=str, required=True, help="Model to connect to")
parser.add_argument("--model_two", type=str, required=False, help="Second model")
parser.add_argument("--completion_mode", action="store_true", default=False, help="Use completion mode for OpenAI API")
args = parser.parse_args()

# OpenAI configuration
api_key = "EMPTY"  # OpenAI API key (empty for custom server)

model_client = ModelClientHandler(args.model_one, api_key, args.port_one, debug=args.debug, stream=True)
model_client_2 = ModelClientHandler(args.model_two, api_key, args.port_two, debug=args.debug, stream=True)

# Launch Gradio app
system_prompt_1 = gr.Textbox(label="Left System Prompt", placeholder="You are a helpful assistant, please respond to the user prompt.")
system_prompt_2 = gr.Textbox(label="Right System Prompt", placeholder="You are a helpful assistant, please respond to the user prompt.")
temperature_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label="Temperature")
safety_filter_checkbox = gr.Checkbox(label="Run Safety Filter", value=False)

demo = EnhancedChatInterface(
    fn=model_client.predict,
    safety_fn=run_dummy_safety_filter, # no safety filter on side-by-side demo
    model_client=model_client,
    fn_2=model_client_2.predict,
    model_client_2=model_client_2,
    additional_inputs=[system_prompt_1, system_prompt_2, temperature_slider, safety_filter_checkbox],
    title="AI2 Internal Model Demo",
    css=css_style,
    theme=theme,
    concurrency_limit=4,
)

demo.queue().launch(allowed_paths=['assets/'], share=True)
