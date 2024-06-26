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

import gradio as gr

css_style = """
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
footer {visibility: hidden}
"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.pink,
    secondary_hue=gr.themes.colors.emerald,
    font=[gr.themes.GoogleFont("Source Sans Pro")],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono")],
).set(
    loader_color="#0FCB8C",
    slider_color="#0FCB8C",
    block_border_color="#e1d9d1",  # darkened faf2e9
    block_border_width="2.5px",
    # block_shadow="*shadow_drop_lg",
    # input
    border_color_primary="#F0529C",
    # textbox_border_width="2px",
    # textbox_shadow="none",
    # buttons
    button_shadow="none",
    button_border_width="0px",
    button_primary_background_fill_hover="#B11BE8",
    button_primary_background_fill="#F0529C",
    button_primary_border_color="#F0529C",
    button_primary_border_color_dark="#B11BE8",
    # button_secondary_background_fill_hover="#0FCB8C", # "#e1d9d1",
    # button_secondary_background_fill="#0CA270", # "#FAF2E9",
    # button_secondary_border_color="#0FCB8C", #"#FAF2E9",
    # button_secondary_border_color_dark="#0FCB8C", #"#FAF2E9",
)