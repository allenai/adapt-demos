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

from typing import Iterable, Union

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

css_style = """
.classifier-text {
    font-size: 20px !important;
}
.safe-text {
    font-size: 16px !important;
    color: green;
}
.safe-title {
    color: green;
}
.white-background {
    background-color: white;
}
footer {visibility: hidden}
hr { margin: 0.2em auto; }
"""

js_url = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
css_url = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
header = f"""
<link rel="stylesheet" href="{css_url}">
<script src="{js_url}" crossorigin="anonymous"></script>
"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.pink,
    secondary_hue=gr.themes.colors.emerald,
    font=[gr.themes.GoogleFont("Source Sans Pro")],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono")],
).set(
    # core
    body_background_fill="#0A3235",
    # misc
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

ai2_off_white = colors.Color(
    name="ai2-off-white",
    c50="#fffcfa",
    c100="#fdf9f4",
    c200="#fcf6f0",
    c300="#fbf3eb",
    c400="#faf2e9",  # default
    c500="#e1dad2",
    c600="#c8c2ba",
    c700="#afa9a3",
    c800="#96918c",
    c900="#7d7975",
    c950="#64615d",
)

ai2_pink = colors.Color(
    name="ai2-pink",
    c50="#feeef5",
    c100="#fcdceb",
    c200="#f9bad7",
    c300="#f697c4",
    c400="#f375b0",
    c500="#f0529c",  # default
    c600="#d84a8c",
    c700="#a8396d",
    c800="#78294e",
    c900="#48192f",
    c950="#180810",
)


ai2_dark_green = colors.Color(
    name="ai2-dark-green",
    c50="#e7ebeb",
    c100="#ced6d7",
    c200="#9dadae",
    c300="#6c8486",
    c400="#3b5b5d",
    c500="#0a3235",  # default
    c600="#092d30",
    c700="#072325",
    c800="#05191b",
    c900="#030f10",
    c950="#010505",
)

ai2_light_green = colors.Color(
    name="ai2-light-green",
    c50="#e7faf4",
    c100="#cff5e8",
    c200="#9fead1",
    c300="#6fe0ba",
    c400="#3fd5a3",
    c500="#0fcb8c",  # default
    c600="#0eb77e",
    c700="#0b8e62",
    c800="#086646",
    c900="#043d2a",
    c950="#01140e",
)


ai2_purple = colors.Color(
    name="ai2-purple",
    c50="#f7e8fd",
    c100="#efd1fa",
    c200="#e0a4f6",
    c300="#d076f1",
    c400="#c149ed",
    c500="#b11be8",  # default
    c600="#9f18d1",
    c700="#7c13a2",
    c800="#590e74",
    c900="#350846",
    c950="#120317",
)


class Ai2Theme(Base):
    def __init__(
        self,
        *,
        primary_hue: Union[colors.Color, str] = ai2_pink,
        secondary_hue: Union[colors.Color, str] = ai2_dark_green,
        neutral_hue: Union[colors.Color, str] = ai2_off_white,
        spacing_size: Union[sizes.Size, str] = sizes.spacing_md,
        radius_size: Union[sizes.Size, str] = sizes.radius_md,
        text_size: Union[sizes.Size, str] = sizes.text_lg,
        font: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("Manrope"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            # Body Attributes: These set set the values for the entire body of the app.
            body_background_fill="*neutral_400",
            body_background_fill_dark="*secondary_800",
            body_text_color="*secondary_500",
            body_text_color_dark="*neutral_400",
            body_text_size=None,
            body_text_color_subdued="*secondary_300",
            body_text_color_subdued_dark="*neutral_600",
            body_text_weight="600",
            embed_radius="*radius_lg",
            # Element Colors: These set the colors for common elements.
            background_fill_primary="*neutral_50",
            background_fill_primary_dark="*secondary_700",
            background_fill_secondary="*neutral_300",
            background_fill_secondary_dark="*secondary_800",
            border_color_accent="*primary_500",
            border_color_accent_dark="*primary_600",
            border_color_accent_subdued="*primary_500",
            border_color_accent_subdued_dark="*primary_600",
            border_color_primary="*secondary_200",
            border_color_primary_dark="*secondary_600",
            color_accent="*primary_500",
            color_accent_soft="*primary_200",
            color_accent_soft_dark="*primary_500",
            # Text: This sets the text styling for text elements.
            link_text_color="*primary_500",
            link_text_color_dark="*primary_500",
            link_text_color_active="*primary_300",
            link_text_color_active_dark="*primary_300",
            link_text_color_hover="*primary_500",
            link_text_color_hover_dark="*primary_400",
            link_text_color_visited="*primary_700",
            link_text_color_visited_dark="*primary_700",
            prose_text_size=None,
            prose_text_weight=None,
            prose_header_text_weight=None,
            code_background_fill="*neutral_400",
            code_background_fill_dark="*secondary_900",
            # Shadows: These set the high-level shadow rendering styles.
            # These variables are often referenced by other component-specific shadow variables.
            shadow_drop=None,
            shadow_drop_lg=None,
            shadow_inset=None,
            shadow_spread=None,
            shadow_spread_dark=None,
            # Layout Atoms: These set the style for common layout elements, such as
            # the blocks that wrap components.
            block_background_fill="*neutral_50",
            block_background_fill_dark="*secondary_500",
            block_border_color="*secondary_200",
            block_border_color_dark=None,
            block_border_width="*spacing_xs",
            block_border_width_dark=None,
            block_info_text_color="*secondary_500",
            block_info_text_color_dark=None,
            block_info_text_size=None,
            block_info_text_weight=None,
            block_label_background_fill="*secondary_50",
            block_label_background_fill_dark=None,
            block_label_border_color=None,
            block_label_border_color_dark=None,
            block_label_border_width=None,
            block_label_border_width_dark=None,
            block_label_shadow=None,
            block_label_text_color="*secondary_500",
            block_label_text_color_dark=None,
            block_label_margin=None,
            block_label_padding=None,
            block_label_radius=None,
            block_label_right_radius=None,
            block_label_text_size=None,
            block_label_text_weight=None,
            block_padding=None,
            block_radius="*radius_lg",
            block_shadow=None,
            block_shadow_dark=None,
            block_title_background_fill=None,
            block_title_background_fill_dark=None,
            block_title_border_color=None,
            block_title_border_color_dark=None,
            block_title_border_width=None,
            block_title_border_width_dark=None,
            block_title_text_color="*secondary_500",
            block_title_text_color_dark="*primary_500",
            block_title_padding=None,
            block_title_radius=None,
            block_title_text_size=None,
            block_title_text_weight="800",
            container_radius="*radius_lg",
            form_gap_width=None,
            layout_gap=None,
            panel_background_fill=None,
            panel_background_fill_dark=None,
            panel_border_color=None,
            panel_border_color_dark=None,
            panel_border_width=None,
            panel_border_width_dark=None,
            section_header_text_size=None,
            section_header_text_weight=None,
            # Component Atoms: These set the style for elements within components.
            accordion_text_color="*secondary_300",
            accordion_text_color_dark=None,
            table_text_color=None,
            table_text_color_dark=None,
            checkbox_background_color=None,
            checkbox_background_color_dark="*primary_100",
            checkbox_background_color_focus=None,
            checkbox_background_color_focus_dark="*primary_100",
            checkbox_background_color_hover=None,
            checkbox_background_color_hover_dark="*primary_100",
            checkbox_background_color_selected=None,
            checkbox_background_color_selected_dark="*primary_500",
            checkbox_border_color="*secondary_400",
            checkbox_border_color_dark="*primary_600",
            checkbox_border_color_focus=None,
            checkbox_border_color_focus_dark="*primary_700",
            checkbox_border_color_hover="*secondary_400",
            checkbox_border_color_hover_dark="*primary_500",
            checkbox_border_color_selected="*secondary_400",
            checkbox_border_color_selected_dark="*primary_500",
            checkbox_border_radius="*spacing_sm",
            checkbox_border_width="*spacing_xs",
            checkbox_border_width_dark=None,
            checkbox_check=None,
            radio_circle=None,
            checkbox_shadow=None,
            checkbox_label_background_fill=None,
            checkbox_label_background_fill_dark="*secondary_700",
            checkbox_label_background_fill_hover=None,
            checkbox_label_background_fill_hover_dark="*secondary_700",
            checkbox_label_background_fill_selected=None,
            checkbox_label_background_fill_selected_dark="*secondary_700",
            checkbox_label_border_color="*secondary_400",
            checkbox_label_border_color_dark=None,
            checkbox_label_border_color_hover=None,
            checkbox_label_border_color_hover_dark=None,
            checkbox_label_border_width="*spacing_xs",
            checkbox_label_border_width_dark=None,
            checkbox_label_gap=None,
            checkbox_label_padding=None,
            checkbox_label_shadow=None,
            checkbox_label_text_size=None,
            checkbox_label_text_weight=None,
            checkbox_label_text_color=None,
            checkbox_label_text_color_dark=None,
            checkbox_label_text_color_selected=None,
            checkbox_label_text_color_selected_dark=None,
            error_background_fill=None,
            error_background_fill_dark=None,
            error_border_color=None,
            error_border_color_dark=None,
            error_border_width=None,
            error_border_width_dark=None,
            error_text_color=None,
            error_text_color_dark=None,
            error_icon_color=None,
            error_icon_color_dark=None,
            input_background_fill="*neutral_400",
            input_background_fill_dark="*secondary_700",
            input_background_fill_focus=None,
            input_background_fill_focus_dark=None,
            input_background_fill_hover=None,
            input_background_fill_hover_dark=None,
            input_border_color="*secondary_200",
            input_border_color_dark=None,
            input_border_color_focus=None,
            input_border_color_focus_dark=None,
            input_border_color_hover=None,
            input_border_color_hover_dark=None,
            input_border_width="*spacing_xs",
            input_border_width_dark=None,
            input_padding=None,
            input_placeholder_color=None,
            input_placeholder_color_dark=None,
            input_radius="*radius_md",
            input_shadow=None,
            input_shadow_dark=None,
            input_shadow_focus=None,
            input_shadow_focus_dark=None,
            input_text_size=None,
            input_text_weight="300",
            loader_color="*primary_500",
            loader_color_dark=None,
            slider_color="*primary_500",
            slider_color_dark=None,
            stat_background_fill="*neutral_50",
            stat_background_fill_dark="*secondary_600",
            table_border_color="*secondary_200",
            table_border_color_dark=None,
            table_even_background_fill="white",
            table_even_background_fill_dark="*secondary_400",
            table_odd_background_fill="*neutral_100",
            table_odd_background_fill_dark="*secondary_500",
            table_radius="*radius_md",
            table_row_focus="*primary_100",
            table_row_focus_dark=None,
            # Buttons: These set the style for buttons.
            button_border_width="*spacing_xs",
            button_border_width_dark=None,
            button_shadow=None,
            button_shadow_active=None,
            button_shadow_hover=None,
            button_transition=None,
            button_large_padding="*spacing_md",
            button_large_radius="*spacing_lg",
            button_large_text_size="*text_lg",
            button_large_text_weight=None,
            button_small_padding="*spacing_sm",
            button_small_radius="*spacing_md",
            button_small_text_size="*text_sm",
            button_small_text_weight=None,
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_dark="*primary_500",
            button_primary_background_fill_hover="*primary_400",
            button_primary_background_fill_hover_dark="*primary_400",
            button_primary_border_color="*primary_700",
            button_primary_border_color_dark="*primary_700",
            button_primary_border_color_hover=None,
            button_primary_border_color_hover_dark=None,
            button_primary_text_color="*secondary_500",
            button_primary_text_color_dark="*secondary_50",
            button_primary_text_color_hover=None,
            button_primary_text_color_hover_dark=None,
            button_secondary_background_fill="*secondary_50",
            button_cancel_background_fill_dark="*neutral_950",
            button_secondary_background_fill_hover="*secondary_100",
            button_cancel_background_fill_hover_dark="*neutral_900",
            button_secondary_border_color="*neutral_800",
            button_cancel_border_color_dark="*neutral_950",
            button_secondary_border_color_hover=None,
            button_cancel_border_color_hover_dark=None,
            button_secondary_text_color="*secondary_500",
            button_cancel_text_color_dark="*secondary_50",
            button_secondary_text_color_hover=None,
            button_cancel_text_color_hover_dark=None,
            button_cancel_background_fill="*secondary_500",
            button_secondary_background_fill_dark="*secondary_500",
            button_cancel_background_fill_hover="*secondary_400",
            button_secondary_background_fill_hover_dark="*secondary_400",
            button_cancel_border_color="*secondary_500",
            button_secondary_border_color_dark="*secondary_500",
            button_cancel_border_color_hover=None,
            button_secondary_border_color_hover_dark=None,
            button_cancel_text_color="*neutral_400",
            button_secondary_text_color_dark="*neutral_400",
            button_cancel_text_color_hover=None,
            button_secondary_text_color_hover_dark=None,
        )
