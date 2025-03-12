import gradio as gr
import globals
from pathlib import Path
import os
import re

g = globals.get_globals()
from .ui_helpers import create_textbox, create_checkbox, create_slider, create_dropdown, create_button, create_gallery

# Initialize wildcard cache in settings_data
if 'wildcard_cache' not in g.settings_data:
    g.settings_data['wildcard_cache'] = {}
    WILDCARD_DIR = Path("wildcards")
    for file_path in WILDCARD_DIR.glob("**/*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Use only the filename, no folder or extension
            filename = file_path.stem  # 'negative_hand-neg' from 'embeddings/negative_hand-neg.txt'
            wildcard_syntax = f"__{filename}__"
            g.settings_data['wildcard_cache'][wildcard_syntax] = content.lower()
        except Exception:
            pass

import re

def get_wildcard_suggestions(text, enable_suggestions, phrase_cap):
    if not enable_suggestions or not text:
        return gr.update(choices=[], value=None, visible=False), "No suggestions available"

    # Split into phrases by commas
    phrases = [phrase.strip() for phrase in text.split(',') if phrase.strip()]
    current_phrase = phrases[-1] if phrases else ""

    # Split current phrase into words (spaces only)
    split_words = [word for word in re.split(r'\s+', current_phrase) if word]
    last_word = split_words[-1].lower() if split_words else ""
    last_few_words = " ".join(split_words[-phrase_cap:]).lower() if len(split_words) >= phrase_cap else " ".join(split_words).lower()
    full_text = text.lower()

    print(f"Raw input: '{text}'")
    print(f"Phrases: {phrases}")
    print(f"Current phrase: '{current_phrase}'")
    print(f"Split words: {split_words}")
    print(f"Last word: '{last_word}'")
    print(f"Last few words (cap {phrase_cap}): '{last_few_words}'")

    # Get suggestions for each
    last_word_suggestions = [w for w, c in g.settings_data['wildcard_cache'].items() if last_word in c]
    last_few_suggestions = [w for w, c in g.settings_data['wildcard_cache'].items() if last_few_words in c]
    # full_text_suggestions = [w for w, c in g.settings_data['wildcard_cache'].items() if full_text in c]  # Optional, dropping for now

    # Clean and limit to 5 each, exclude overlaps from last_few in last_word
    last_few_cleaned = [f"__{Path(sug.strip('_')).name}__" for sug in list(dict.fromkeys(last_few_suggestions))][:5]
    last_word_cleaned = [f"__{Path(sug.strip('_')).name}__" for sug in list(dict.fromkeys(last_word_suggestions)) if sug not in last_few_suggestions][:5]

    # Combine with separator
    separator = "----------"
    cleaned_suggestions = last_few_cleaned + ([separator] if last_few_cleaned and last_word_cleaned else []) + last_word_cleaned

    status = f"Phrase (cap {phrase_cap}): {len(last_few_suggestions)}, Word: {len(last_word_suggestions)}" if cleaned_suggestions else "No matching wildcards found"
    print(f"Suggestions: {cleaned_suggestions}")
    return gr.update(choices=cleaned_suggestions, value=None, visible=True), status

def insert_wildcard(text, suggestion):
    if suggestion == "----------":
        return text
    return text + (suggestion if suggestion else "")

def sailing_main_view():
    components = {}
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                components['sail_result_images'] = create_gallery("output images")
            with gr.Row():
                with gr.Column(scale=3):
                    components['sail_text'] = create_textbox(
                        "Start your journey with this search. This will be used all along. Change it during sailing to change course.",
                        g.settings_data['sailing']['sail_text'], "Where do we set our sails", elem_id='sail-input-text'
                    )
                with gr.Column(scale=1):
                    components['sail_keep_text'] = create_checkbox(
                        "Use the input during the whole trip",
                        g.settings_data['sailing']['sail_keep_text'],
                        "if set to true there is no dynamic prompting, only the context is changed by the sailing."
                    )
            with gr.Row():
                components['sail_suggestions_checkbox'] = create_checkbox(
                    "Enable Wildcard Suggestions", False, "Show wildcard suggestions for text input"
                )
                components['sail_phrase_cap'] = create_slider(
                    "Phrase Word Limit", value=3, min_val=1, max_val=5, step=1,
                    info="Max number of words to consider in the current phrase"
                )
                components['sail_text_suggestions'] = gr.Dropdown(
                    label="Suggested Wildcards", choices=[], interactive=True, visible=False
                )

            with gr.Row():
                components['sail_result'] = create_textbox("Your journey journal", "", "Your journey logs", autoscroll=True)
        with gr.Column(scale=1):
            components['sail_status'] = create_textbox("Status", "", "status")
            components['sail_suggestion_status'] = create_textbox("Suggestion Status", "No suggestions available", interactive=False)  # New field
            with gr.Row():
                components['sail_submit_button'] = create_button("Start your journey")
                components['sail_stop_button'] = create_button("Interrupt your journey")
                components['sail_count_button'] = create_button("Count possible results")
                components['sail_check_connect_button'] = create_button("Check API Available")
            components['sail_max_gallery_size'] = create_slider(
                "Max Gallery size", g.settings_data['sailing']['sail_max_gallery_size'], max_val=500,
                info="Limit the number of images kept in the gallery choose between 1 and 500"
            )


    components['sail_text_suggestions'].change(
        fn=insert_wildcard,
        inputs=[components['sail_text'], components['sail_text_suggestions']],
        outputs=components['sail_text']
    )

    components['sail_text'].change(
        fn=get_wildcard_suggestions,
        inputs=[components['sail_text'], components['sail_suggestions_checkbox'], components['sail_phrase_cap']],
        outputs=[components['sail_text_suggestions'], components['sail_suggestion_status']]
    )


    return components


def sailing_setup_view():
    components = {}
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                components['sail_width'] = create_slider("Sail steps", g.settings_data['sailing']['sail_width'])
                components['sail_depth'] = create_slider("Sail distance", g.settings_data['sailing']['sail_depth'])
                components['sail_depth_preset'] = create_slider(
                    "Sail distance preset", g.settings_data['sailing']['sail_depth_preset'], max_val=1000000
                )

            with gr.Row():
                components['sail_sinus'] = create_checkbox(
                    "Add a sinus to the distance",
                    g.settings_data['sailing']['sail_sinus'],
                    "This will create a sinus wave based movement along the distance."
                )
                components['sail_sinus_freq'] = create_slider(
                    "Sinus Frequency", g.settings_data['sailing']['sail_sinus_freq'], min_val=0.1, max_val=10, step=0.1
                )
                components['sail_sinus_range'] = create_slider(
                    "Sinus Multiplier", g.settings_data['sailing']['sail_sinus_range'], max_val=500
                )
            with gr.Row():
                components['sail_generate'] = create_checkbox(
                    "Generate with A1111 / SD Forge",
                    g.settings_data['sailing']['sail_generate'],
                    "Do you want to directly generate the images?"
                )
                components['sail_unload_llm'] = create_checkbox(
                    "Unload LLM while generate?",
                    g.settings_data['sailing']['sail_unload_llm']
                )
                components['automa_alt_vae'] = create_dropdown(
                    "Alternate VAE for fixing black images",
                    g.settings_data['automa']['automa_vaes'],
                    g.settings_data['automa']['automa_alt_vae']
                )

    return components



def sailing_filters():
    components = {}
    with gr.Row():
        with gr.Column(scale=1):
            components['sail_filter_count_button'] = create_button("Count possible results")
        with gr.Column(scale=3):
            components['sail_filter_status'] = create_textbox("Count", "", "0")
    with gr.Row():
        with gr.Column(scale=1):
            components['sail_filter_prompt'] = create_checkbox(
                "Filter on prompt Level?",
                g.settings_data['sailing']['sail_filter_prompt'],
                "With this you filter entries from the prompt generation. It may lead to long wait time until a prompt will match."
            )
            components['sail_filter_context'] = create_checkbox(
                "Filter on context Level?",
                g.settings_data['sailing']['sail_filter_context'],
                "With this you filter entries from the context prior to prompt generation. It may lead to empty context."
            )
        with gr.Column(scale=3):
            components['sail_filter_text'] = create_textbox(
                "List of negative words, words that are not allowed to be in context.",
                g.settings_data['sailing']['sail_filter_text'],
                "Comma separated list of words you dont want in your prompt"
            )
            components['sail_filter_not_text'] = create_textbox(
                "List of positive words, words that must be in context.",
                g.settings_data['sailing']['sail_filter_not_text'],
                "Comma separated list of words that must be part of the prompt"
            )
    with gr.Row():
        with gr.Column(scale=1):
            components['sail_neg_filter_context'] = create_checkbox(
                "Filter on negative prompt context Level?",
                g.settings_data['sailing']['sail_filter_context'],
                "With this you filter entries from the context prior to prompt generation. It may lead to empty context."
            )
        with gr.Column(scale=3):
            components['sail_neg_filter_text'] = create_textbox(
                "List of negative words, words that are not allowed to be in negative prompt context.",
                g.settings_data['sailing']['sail_filter_text'],
                "Comma separated list of words you dont want in your negative prompt"
            )
            components['sail_neg_filter_not_text'] = create_textbox(
                "List of positive words, words that must be in negative prompt context.",
                g.settings_data['sailing']['sail_filter_not_text'],
                "Comma separated list of words that must be part of the negative prompt"
            )
    with gr.Row():
        with gr.Column(scale=1):
            components['sail_add_search'] = create_checkbox(
                "Add search specification",
                g.settings_data['sailing']['sail_add_search'],
                "Add a text to each vector search."
            )
        with gr.Column(scale=3):
            components['sail_search'] = create_textbox(
                "Search Spec", g.settings_data['sailing']['sail_search'], "Enter your additional search"
            )
    return components

def sailing_prompt_manipulation():
    components = {}
    with gr.Row():
        components['sail_dyn_neg'] = create_checkbox(
            "Use dynamic Negative Prompt",
            g.settings_data['sailing']['sail_dyn_neg'],
            "Uses the negative if we find one, or the default. Be warned this can cause black images or other troubles."
        )
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            components['sail_add_neg'] = create_checkbox(
                "Add to negative prompt",
                g.settings_data['sailing']['sail_add_neg'],
                "Add a text to each negative prompt"
            )
            components['prompt_suggestions_checkbox'] = create_checkbox(
                "Enable Wildcard Suggestions", False, "Show wildcard suggestions for text inputs"
            )
            components['phrase_cap'] = create_slider(
                "Phrase Word Limit", value=3, min_val=1, max_val=5, step=1,
                info="Max number of words to consider in the current phrase"
            )
            components['prompt_suggestion_status'] = create_textbox("Suggestion Status", "No suggestions available", interactive=False)  # New field
        with gr.Column(scale=3):
            components['sail_neg_prompt'] = create_textbox(
                "Negative Prompt addon", g.settings_data['sailing']['sail_neg_prompt'],
                "Enter your negative prompt addon"
            )
            components['sail_neg_prompt_suggestions'] = gr.Dropdown(
                label="Suggested Wildcards", choices=[], interactive=True, visible=False
            )
        with gr.Column(scale=3):
            components['sail_neg_embed'] = create_textbox(
                "Negative embeddings", g.settings_data['sailing']['sail_neg_embed'],
                "Enter your negative prompt embeddings"
            )
            components['sail_neg_embed_suggestions'] = gr.Dropdown(
                label="Suggested Wildcards", choices=[], interactive=True, visible=False
            )
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            components['sail_add_style'] = create_checkbox(
                "Add style specification",
                g.settings_data['sailing']['sail_add_style'],
                "Add a text to each prompt"
            )

        with gr.Column(scale=3):
            components['sail_style'] = create_textbox(
                "Style Spec", g.settings_data['sailing']['sail_style'], "Enter your hardcoded style"
            )
            components['sail_style_suggestions'] = gr.Dropdown(
                label="Suggested Wildcards", choices=[], interactive=True, visible=False
            )
        with gr.Column(scale=3):
            components['sail_pos_embed'] = create_textbox(
                "Positive embeddings", g.settings_data['sailing']['sail_pos_embed'],
                "Enter your positive embeddings"
            )
            components['sail_pos_embed_suggestions'] = gr.Dropdown(
                label="Suggested Wildcards", choices=[], interactive=True, visible=False
            )

    with gr.Row():
        components['sail_summary'] = create_checkbox(
            "Do summary of LLM prompt",
            g.settings_data['sailing']['sail_summary'],
            "The prompt will get reduced to a summary"
        )
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            components['sail_rephrase'] = create_checkbox(
                "Rephrase LLM prompt",
                g.settings_data['sailing']['sail_rephrase'],
                "The prompt gets rephrased based on the rephrase prompt"
            )
            components['sail_gen_rephrase'] = create_checkbox(
                "Generate the input Prompt too",
                g.settings_data['sailing']['sail_gen_rephrase'],
                "To see the effect of the rephrasing you can check here to get both prompts generated"
            )
        with gr.Column(scale=3):
            components['sail_rephrase_prompt'] = create_textbox(
                "Rephrase Prompt", g.settings_data['sailing']['sail_rephrase_prompt'],
                "Enter your rephrase prompt", lines=4
            )

    for textbox, suggestions in [
        ('sail_neg_prompt', 'sail_neg_prompt_suggestions'),
        ('sail_neg_embed', 'sail_neg_embed_suggestions'),
        ('sail_style', 'sail_style_suggestions'),
        ('sail_pos_embed', 'sail_pos_embed_suggestions')
    ]:
        components[textbox].change(
            fn=get_wildcard_suggestions,
            inputs=[components[textbox], components['prompt_suggestions_checkbox'], components['phrase_cap']],
            outputs=[components[suggestions], components['prompt_suggestion_status']]
        )
        components[suggestions].change(
            fn=insert_wildcard,
            inputs=[components[textbox], components[suggestions]],
            outputs=components[textbox]
        )

    return components

def sailing_generation_sailing(ui_code):
    components = {}
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column():
                    components['sail_gen_enabled'] = create_checkbox(
                        "Enable generation parameters",
                        g.settings_data['sailing']['sail_gen_enabled'],
                        "Enable dynamic generation parameters?"
                    )
                    components['sail_override_settings_restore'] = create_checkbox(
                        "Restore overriden Settings after each image",
                        g.settings_data['sailing']['sail_override_settings_restore'],
                        "If set to true the Checkpoint and VAE will be set to the settings in SD Forge/Auto1111..."
                    )
                    components['sail_store_folders'] = create_checkbox(
                        "store the images in folders per model?",
                        g.settings_data['sailing']['sail_store_folders'],
                        "Should the images be stored in different Folders per model?"
                    )
                with gr.Column():
                    components['sail_gen_any_combination'] = create_checkbox(
                        "Generate any combination selected for any single prompt?",
                        g.settings_data['sailing']['sail_gen_any_combination'],
                        "Should any prompt be generated for any possible combination?"
                    )
                    components['sail_gen_type'] = gr.Radio(
                        ['Random', 'Linear'], label="Select type of change, Random or linear after n steps",
                        value=g.settings_data['sailing']['sail_gen_type'], interactive=True
                    )
                    components['sail_gen_steps'] = create_slider(
                        "Steps", g.settings_data['sailing']['sail_gen_steps'], max_val=100
                    )
            with gr.Row():
                components['sail_dimensions'] = ui_code.prompt_iterator.setting_dropdown(
                    label='Dimensions', choices=g.settings_data['model_test']['model_test_dimensions_list'],
                    initial_value=g.settings_data['sailing']['sail_dimensions']
                )
            with gr.Row():
                components['sail_sampler'] = ui_code.prompt_iterator.setting_dropdown(
                    choices=g.settings_data['automa']['automa_samplers'],
                    initial_value=g.settings_data['sailing']['sail_sampler'], label='Sampler'
                )
                components['sail_scheduler'] = ui_code.prompt_iterator.setting_dropdown(
                    choices=g.settings_data['automa']['automa_schedulers'],
                    initial_value=g.settings_data['sailing']['sail_scheduler'], label='Scheduler'
                )
                components['sail_checkpoint'] = ui_code.prompt_iterator.setting_dropdown(
                    choices=g.settings_data['automa']['automa_checkpoints'],
                    initial_value=g.settings_data['sailing']['sail_checkpoint'], label='Checkpoint'
                )
                components['sail_vae'] = ui_code.prompt_iterator.setting_dropdown(
                    choices=g.settings_data['automa']['automa_vaes'],
                    initial_value=g.settings_data['sailing']['sail_vae'], label='VAE'
                )
        with gr.Column(scale=1):
            components['sail_gen_refresh_button'] = create_button("Refresh")
    return components

def sailing_show():
    components = {}
    with gr.Row():
        components['sail_show_submit_button'] = create_button("Start your journey")
    with gr.Row():
        components['sail_show_image'] = gr.Image(height=800, width=1300)
    with gr.Row():
        components['sail_show_result'] = create_textbox("Your journey journal", "", "Your journey logs", lines=4)
    return components

def setup_sailing_tab(sailor, ui_code):
    with gr.Tab("Sailing") as sailing:
        with gr.Tab("Main view"):
            main_components = sailing_main_view()
        with gr.Tab("Setup"):
            setup_components = sailing_setup_view()
        with gr.Tab("Filters"):
            filter_components = sailing_filters()
        with gr.Tab("Prompt manipulation"):
            prompt_components = sailing_prompt_manipulation()
        with gr.Tab("Generation Sailing") as gen_sail:
            gen_components = sailing_generation_sailing(ui_code)
    with gr.Tab("Show"):
        show_components = sailing_show()

    all_components = {**main_components,**setup_components, **filter_components, **prompt_components, **gen_components, **show_components}

    inputs = [
        all_components[key] for key in [
            'sail_text', 'sail_keep_text', 'sail_width', 'sail_depth', 'sail_generate', 'sail_summary',
            'sail_rephrase', 'sail_rephrase_prompt', 'sail_gen_rephrase', 'sail_sinus', 'sail_sinus_freq',
            'sail_sinus_range', 'sail_add_style', 'sail_style', 'sail_add_search', 'sail_search',
            'sail_max_gallery_size', 'sail_dyn_neg', 'sail_add_neg', 'sail_neg_prompt', 'sail_filter_text',
            'sail_filter_not_text', 'sail_filter_context', 'sail_filter_prompt', 'sail_neg_filter_text',
            'sail_neg_filter_not_text', 'sail_neg_filter_context', 'automa_alt_vae', 'sail_checkpoint',
            'sail_sampler', 'sail_vae', 'sail_dimensions', 'sail_gen_type', 'sail_gen_any_combination',
            'sail_gen_steps', 'sail_gen_enabled', 'sail_override_settings_restore', 'sail_store_folders',
            'sail_depth_preset', 'sail_scheduler', 'sail_unload_llm', 'sail_neg_embed', 'sail_pos_embed'
        ]
    ]
    gr.on(triggers=[comp.change for comp in inputs], fn=ui_code.set_sailing_settings, inputs=inputs, outputs=None)

    outputs = [
        all_components[key] for key in [
            'sail_text', 'sail_width', 'sail_depth', 'sail_generate', 'sail_summary', 'sail_rephrase',
            'sail_rephrase_prompt', 'sail_gen_rephrase', 'sail_sinus', 'sail_sinus_freq', 'sail_sinus_range',
            'sail_add_style', 'sail_style', 'sail_add_search', 'sail_search', 'sail_max_gallery_size',
            'sail_filter_text', 'sail_filter_not_text', 'sail_filter_context', 'sail_filter_prompt',
            'sail_neg_filter_text', 'sail_neg_filter_not_text', 'sail_neg_filter_context', 'automa_alt_vae',
            'sail_checkpoint', 'sail_sampler', 'sail_vae', 'sail_dimensions', 'sail_gen_type', 'sail_gen_steps',
            'sail_gen_enabled', 'sail_override_settings_restore', 'sail_store_folders', 'sail_depth_preset',
            'sail_scheduler', 'sail_neg_embed', 'sail_pos_embed'
        ]
    ]
    gr.on(triggers=[sailor.select], fn=ui_code.get_sailing_settings, inputs=None, outputs=outputs)

    gr.on(triggers=[gen_sail.select], fn=ui_code.automa_sail_refresh, inputs=None,
          outputs=[all_components['sail_sampler'], all_components['sail_checkpoint'], all_components['sail_vae'], all_components['sail_scheduler']])

    start_sail = all_components['sail_submit_button'].click(
        fn=ui_code.run_t2t_sail,
        inputs=[],
        outputs=[all_components['sail_result'], all_components['sail_result_images'], all_components['sail_status']]
    )
    all_components['sail_stop_button'].click(
        fn=ui_code.stop_job,
        inputs=None,
        outputs=None,  # No change needed here since cancels handles it
        cancels=[start_sail]
    )
    all_components['sail_check_connect_button'].click(fn=ui_code.check_api_avail, inputs=None, outputs=all_components['sail_status'])
    all_components['sail_count_button'].click(fn=ui_code.count_context, inputs=None, outputs=all_components['sail_status'])
    all_components['sail_filter_count_button'].click(fn=ui_code.count_context, inputs=None, outputs=all_components['sail_filter_status'])
    all_components['sail_gen_refresh_button'].click(
        fn=ui_code.automa_sail_refresh, inputs=None,
        outputs=[all_components['sail_sampler'], all_components['sail_checkpoint'], all_components['sail_vae'], all_components['sail_scheduler']]
    )

    start_sail_show = all_components['sail_show_submit_button'].click(
        fn=ui_code.run_t2t_show_sail, inputs=None, outputs=[all_components['sail_show_result'], all_components['sail_show_image']]
    )