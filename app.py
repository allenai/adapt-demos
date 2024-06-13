import argparse
import gradio as gr
from openai import OpenAI
from src.prompts import WILDGUARD_INPUT_FORMAT

# Define an argument parser
parser = argparse.ArgumentParser(description="Gradio App with Custom OpenAI API Port")
parser.add_argument("--port", type=int, default=8000, help="Port to connect to OpenAI API server")
parser.add_argument("--safety_filter_port", type=int, required=False, default=8001, help="Port to connect to safety filter server")
parser.add_argument("--model", type=str, required=True, help="Model to connect to")
parser.add_argument("--safety_model", type=str, required=False, help="Safety model to connect to")
parser.add_argument("--completion_mode", action="store_true", default=False, help="Use completion mode for OpenAI API")
args = parser.parse_args()

# OpenAI configuration
api_key = "EMPTY"  # OpenAI API key (empty for custom server)
model_url = f"http://localhost:{args.port}/v1"  # Construct base URL with provided port
model_client = OpenAI(api_key=api_key, base_url=model_url)

if args.safety_filter_port or args.safety_model:
    # if one of them, both need to be set
    if not args.safety_filter_port or not args.safety_model:
        raise ValueError("Both safety filter port and safety model need to be set") 
    safety_url = f"http://localhost:{args.safety_filter_port}/v1"  # Construct base URL with provided port
    safety_client = OpenAI(api_key=api_key, base_url=safety_url)
    SAFETY_FILTER_ON = True
else:
    SAFETY_FILTER_ON = False

# Prediction function for Gradio
def predict(message, history, temperature, use_safety_filter):
    # Create completion with OpenAI client
    if args.completion_mode:
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
        
        # Check if safety filter is enabled
        if use_safety_filter:
            safety_formatted_input = WILDGUARD_INPUT_FORMAT.format(prompt=message, response=partial_message)
            safety_history_openai_format = history_openai_format[:-1] + [{"role": "user", "content": safety_formatted_input}]
            safety_response = safety_client.chat.completions.create(
                model=args.safety_model,
                messages=safety_history_openai_format,
                temperature=temperature,
                stream=True,
            )
            safety_message = partial_message + "\n\nSafety Filter Feedback:\n\n"
            for chunk in safety_response:
                if chunk.choices[0].delta.content is not None:
                    safety_message = safety_message + chunk.choices[0].delta.content
                    yield safety_message

        #     # Update the filter_feedback textbox
        #     filter_feedback.update(value=safety_response.choices[0].message.content)
        # else:
        #     # If safety filter is not used, clear the feedback box
        #     filter_feedback.update(value="Safety Filter Not Enabled")

        # Note, when you use streaming you'll want to use different logic.
        # See the completion mode example above which uses streaming.

        # return main_response


# Launch Gradio app
temperature_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label="Temperature")

safety_filter_checkbox = gr.Checkbox(label="Run Safety Filter", value=SAFETY_FILTER_ON)

with gr.Blocks(fill_height=True) as app:
    demo = gr.ChatInterface(predict, 
                                additional_inputs=[temperature_slider, safety_filter_checkbox],
                                title="AI2 Internal Demo Model"
                                )
    with gr.Row():
        # add a textbox for safety feedback
        filter_text = gr.Markdown(f"""# Details
                                    Model: {args.model}

                                    Safety Model: {args.safety_model}

                                    Note: Safety only is applied on the most recent prompt.
                                    """)
        filter_feedback = gr.Textbox(label="Feedback", placeholder="(safety filter text here once run)", type="text")

app.queue().launch(share=True)
