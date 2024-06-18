import argparse
import logging
import gradio as gr
from collections import OrderedDict
from gradio.components import HTML
from openai import OpenAI
from src.prompts import WILDGUARD_INPUT_FORMAT, MAKE_SAFE_PROMPT
from src.interface import SafetyChatInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define an argument parser
parser = argparse.ArgumentParser(description="Gradio App with Custom OpenAI API Port")
parser.add_argument("--port", type=int, default=8000, help="Port to connect to OpenAI API server")
parser.add_argument(
    "--safety_filter_port",
    type=int,
    required=False,
    default=8001,
    help="Port to connect to safety filter server",
)
parser.add_argument("--model", type=str, required=True, help="Model to connect to")
parser.add_argument("--safety_model", type=str, required=False, help="Safety model to connect to")
parser.add_argument("--completion_mode", action="store_true", default=False, help="Use completion mode for OpenAI API")
args = parser.parse_args()

# OpenAI configuration
api_key = "EMPTY"  # OpenAI API key (empty for custom server)
model_url = f"http://localhost:{args.port}/v1"  # Construct base URL with provided port
model_client = OpenAI(api_key=api_key, base_url=model_url)
logger.info(f"Connecting to {model_url}")

if args.safety_model:
    safety_url = f"http://localhost:{args.safety_filter_port}/v1"  # Construct base URL with provided port
    safety_client = OpenAI(api_key=api_key, base_url=safety_url)
    SAFETY_FILTER_ON = True
    logger.info(f"Safety filter: ON, connecting to {safety_url}")
else:
    SAFETY_FILTER_ON = False
    logger.info(f"Safety filter: OFF")


# Prediction function for Gradio
def predict(message, history, temperature, safety_filter_checkbox, reprompt_text):
    # Create completion with OpenAI client
    if args.completion_mode:
        logger.debug(" --- PROMPT FOR COMPLETION ---")
        logger.debug(message)
        logger.debug(" ---")
        response = model_client.chat.completions.create(
            model=args.model,
            messages=message,
            temperature=temperature,
            stream=True
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
        logger.debug(" --- CHAT HISTORY ---")
        logger.debug(history_openai_format)
        logger.debug(" ---")
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


temperature_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label="Temperature")
additional_inputs = [temperature_slider]

if SAFETY_FILTER_ON:
    safety_filter_checkbox = gr.Checkbox(label="Run Safety Filter", value=SAFETY_FILTER_ON)
    reprompt_textarea = gr.TextArea(
        label="Prompt to make assistant safe if detected unsafe. Use placeholder {prompt} for user input and {response} for assistant response.",
        value=MAKE_SAFE_PROMPT,
        lines=12,
    )
    additional_inputs += [safety_filter_checkbox, reprompt_textarea]

    def run_safety_filter(message, history, temperature, safety_filter_checkbox, reprompt_text):
        if not safety_filter_checkbox:
            return "Safety filter not enabled", ""

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
                f"{key} <span class='badge text-bg-{'warning' if label.lower() == safety_unwanted_labels[i] else 'success'}'>"
                f"{label.capitalize()}"
                f"</span>"
                for i, (key, label) in enumerate(safety_labels)
            ]
        )
        safety_labels_html = f"<div class='classifier-text'>{safety_labels_html}</div>"

        safety_labels = OrderedDict(safety_labels)
        if not safety_labels or "Response refusal" not in safety_labels:
            logger.error(
                f"Safety class response cannot be parsed: "
                f"[{safety_response.choices[0].message.content}]"
            )
            safety_labels_html = "<p class='text-danger'>Safety response cannot be parsed, please try again</p>"
            safe_response = ""
        elif safety_labels[next(iter(safety_labels))].lower() == "yes" and safety_labels["Response refusal"].lower() == "no":
            reprompt_text = reprompt_text or MAKE_SAFE_PROMPT

            reprompt_kwargs = {}
            if "{prompt}" in reprompt_text:
                reprompt_kwargs["prompt"] = last_query
            if "{response}" in reprompt_text:
                reprompt_kwargs["response"] = last_response

            if not reprompt_kwargs:
                logger.warning(
                    "Make safe prompt template does not include user input ({prompt}) or assistant response ({response})"
                )
            make_response_safe_input = reprompt_text.format(**reprompt_kwargs)
            logger.debug(" --- MAKE SAFE PROMPT ---")
            logger.debug(make_response_safe_input)
            logger.debug(" ---")
            make_response_safe_openai_format = history_openai_format + [{"role": "user", "content": make_response_safe_input}]

            response = model_client.chat.completions.create(
                model=args.model,
                messages=make_response_safe_openai_format,
                temperature=temperature,
            )

            safe_response = HTML(f"""<div class="card text-bg-success">
                <h4 class="card-title safe-title">Safe Response</h4>
                <div class="card-body safe-text">{response.choices[0].message.content}
                </div>
            </div>""")
        else:
            safe_response = "Assistant's response is safe"

        return HTML(safety_labels_html), safe_response
else:
    def run_safety_filter(message, history, temperature):
        return "Safety filter not enabled"

# Launch Gradio app


header = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js" crossorigin="anonymous"></script>
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

demo = SafetyChatInterface(
    predict,
    run_safety_filter,
    additional_inputs=additional_inputs,
    title="AI2 Internal Demo Model",
    description=f"Model: {args.model}\n\nSafety Model: {args.safety_model}",
    head=header,
    css=css,
)

demo.queue().launch(share=True)
