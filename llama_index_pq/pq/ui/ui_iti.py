# ui/iti_ui.py
import gradio as gr
import globals
from settings.io import settings_io


g = globals.get_globals()


def update_setting(key, value):
    # Key = "iti.input_folder" → ["iti", "input_folder"]
    keys = key.split(".")
    if keys[0] != "iti":  # Safety—expect "iti" prefix
        raise ValueError(f"Key must start with 'iti': {key}")
    current = g.settings_data["iti"]  # Start at "iti" level
    sub_keys = keys[1:]  # Skip "iti"—["input_folder"]
    for k in sub_keys[:-1]:  # Empty or nested (e.g., "output_dirs.good" → ["output_dirs"])
        current = current[k]
    current[sub_keys[-1]] = value  # "input_folder"
    settings_io().write_settings(g.settings_data)



def work(iti_manager):
    components = {}
    with gr.Row():
        with gr.Column(scale=3):
            components['input_folder'] = gr.Textbox(
                label="Input Folder",
                value=g.settings_data["iti"]["input_folder"],
                placeholder="Paste folder path (e.g., C:/Images)"
            )
            components['last6_running'] = gr.Gallery(label="Last 6 Running", columns=6, height="auto")
            components['last6_good'] = gr.Gallery(label="Last 6 Good (>=threshold)", columns=6, height="auto")
            components['log_output'] = gr.Textbox(label="Processing Log", lines=10, interactive=False)
        with gr.Column(scale=1):
            components['status'] = gr.Textbox(label="Status", value="Processed 0 images", interactive=False)
            components['run_button'] = gr.Button("Run Batch")
            components['clear_button'] = gr.Button("Clear Log")
            components['stop_button'] = gr.Button("Stop Run")

    components['run_button'].click(
        fn=iti_manager.run_batch,
        inputs=[components['input_folder']],
        outputs=[components['log_output'], components['last6_running'], components['last6_good'], components['status']]
    )
    components['clear_button'].click(
        fn=lambda: ("", [], [], "Processed 0 images"),
        outputs=[components['log_output'], components['last6_running'], components['last6_good'], components['status']]
    )
    components['stop_button'].click(
        fn=iti_manager.stop_job,
        outputs=None
    )
    components['input_folder'].change(
        fn=update_setting, inputs=[gr.State("iti.input_folder"), components['input_folder']]
    )
    return components

def model_research(iti_manager):
    components = {}
    with gr.Tab("Runner"):

        with gr.Row():
            with gr.Column(scale=3):

                components['last6_running'] = gr.Gallery(label="Last 6 Running", columns=6, height="auto")
                components['last6_good'] = gr.Gallery(label="Last 6 Good (>=threshold)", columns=6, height="auto")
                components['log_output'] = gr.Textbox(label="Processing Log", lines=10, interactive=False)
            with gr.Column(scale=1):
                components['status'] = gr.Textbox(label="Status", value="Processed 0 Models", interactive=False)
                components['run_button'] = gr.Button("Run Research")
                components['topics_button'] = gr.Button("get Topics")
                components['stop_research'] = gr.Button("Stop Run")
            # On_change handler to update prompt array
        # On_change handler

    with gr.Tab("Research Prompts"):
        components['prompt_1'] = gr.Textbox(
            label="Prompt 1",
            value=g.settings_data["iti"]["research"]["prompts"][0][0],
            lines=5,
            interactive=True
        )
        components['prompt_1_topic'] = gr.Textbox(
            label="Topic 1",
            value=g.settings_data["iti"]["research"]["prompts"][0][1],
            interactive=True
        )
        components['prompt_2'] = gr.Textbox(
            label="Prompt 2",
            value=g.settings_data["iti"]["research"]["prompts"][1][0],
            lines=5,
            interactive=True
        )
        components['prompt_2_topic'] = gr.Textbox(
            label="Topic 2",
            value=g.settings_data["iti"]["research"]["prompts"][1][1],
            interactive=True
        )
        components['prompt_3'] = gr.Textbox(
            label="Prompt 3",
            value=g.settings_data["iti"]["research"]["prompts"][2][0],
            lines=5,
            interactive=True
        )
        components['prompt_3_topic'] = gr.Textbox(
            label="Topic 3",
            value=g.settings_data["iti"]["research"]["prompts"][2][1],
            interactive=True
        )

    with gr.Tab("Cleanup Results"):
        with gr.Row():
            with gr.Column(scale=3):
                # Drag-and-drop delete zone
                components['delete_drop'] = gr.File(
                    label="Drag Files Here to Delete from DB and Filesystem",
                    file_count="multiple",
                    file_types=["image"]
                )
                components['file_list'] = gr.Markdown(label="Current Files in DB")  # Display DB files
            with gr.Column(scale=1):
                with gr.Row():
                    components['delete_status'] = gr.Textbox(label="Deletion Status", interactive=False)

    with gr.Tab("Down score"):
        with gr.Row():
            with gr.Column(scale=3):
                # Drag-and-drop delete zone
                components['down_score_drop'] = gr.File(
                    label="Drag Files Here to Down score false positives",
                    file_count="multiple",
                    file_types=["image"]
                )
                components['file_list'] = gr.Markdown(label="Current Files in DB")  # Display DB files
            with gr.Column(scale=1):
                with gr.Row():
                    components['down_score_status'] = gr.Textbox(label="Down score Status", interactive=False)

    components['delete_drop'].upload(fn=iti_manager.delete_files,
                                     inputs=components['delete_drop'],
                                     outputs=[components['delete_status'],components['delete_drop']])

    components['down_score_drop'].upload(fn=iti_manager.down_score_files,
                                         inputs=components['down_score_drop'],
                                         outputs=[components['down_score_status'], components['down_score_drop']])


    def update_prompt_array(p1, t1, p2, t2, p3, t3):
        prompts = [[p1.strip(), t1.strip()], [p2.strip(), t2.strip()], [p3.strip(), t3.strip()]]
        g.settings_data["iti"]["research"]["prompts"] = prompts
        settings_io().write_settings(g.settings_data)
        return prompts

    # Attach on_change only to Textbox components
    for comp in components.values():
        if isinstance(comp, gr.Textbox):  # Only Textbox gets the event
            comp.change(
                fn=update_prompt_array,
                inputs=[
                    components['prompt_1'], components['prompt_1_topic'],
                    components['prompt_2'], components['prompt_2_topic'],
                    components['prompt_3'], components['prompt_3_topic']
                ],
                outputs=None
            )
    components['run_button'].click(
        fn=iti_manager.learn_model_performance,
        inputs=None,
        outputs=[components['log_output'], components['last6_running'], components['last6_good'], components['status']]
    )
    components['topics_button'].click(
        fn=iti_manager.get_research_topics,
        inputs=None,
        outputs=[components['log_output'], components['last6_running'], components['last6_good'], components['status']]
    )
    components['stop_research'].click(
        fn=iti_manager.stop_job,
        outputs=None
    )
    return components

def prompting():
    components = {}
    components['pos_style'] = gr.Textbox(
        label="Positive Style addon",
        value=g.settings_data["iti"]["prompt"]["pos_style"],
        interactive=True
    )
    components['pos_lora'] = gr.Textbox(
        label="positive LORA add on",
        value=g.settings_data["iti"]["prompt"]["pos_lora"],
        interactive=True
    )
    components['automa_neg'] = gr.Textbox(
        label="Automa Negative Prompt",
        value=g.settings_data["iti"]["automa"]["negative_prompt"],
        interactive=True
    )
    components['neg_style'] = gr.Textbox(
        label="Negative Style addon",
        value=g.settings_data["iti"]["prompt"]["neg_style"],
        interactive=True
    )
    components['neg_lora'] = gr.Textbox(
        label="Negative LORA add on",
        value=g.settings_data["iti"]["prompt"]["neg_lora"],
        interactive=True
    )

    components['automa_neg'].change(fn=update_setting, inputs=[gr.State("iti.automa.negative_prompt"), components['automa_neg']])
    components['pos_style'].change(fn=update_setting, inputs=[gr.State("iti.prompt.pos_style"), components['pos_style']])
    components['pos_lora'].change(fn=update_setting, inputs=[gr.State("iti.prompt.pos_lora"), components['pos_lora']])
    components['neg_style'].change(fn=update_setting, inputs=[gr.State("iti.prompt.neg_style"), components['neg_style']])
    components['neg_lora'].change(fn=update_setting, inputs=[gr.State("iti.prompt.neg_lora"), components['neg_lora']])

    return components


def settings_folders():
    components = {}
    components['good_folder'] = gr.Textbox(
        label="Good Folder", value=g.settings_data["iti"]["output_dirs"]["good"]
    )
    components['drafts_folder'] = gr.Textbox(
        label="Drafts Folder", value=g.settings_data["iti"]["output_dirs"]["drafts"]
    )
    components['log_file'] = gr.Textbox(
        label="Log File (optional)", value=g.settings_data["iti"]["output"]["log_file"]
    )

    components['good_folder'].change(fn=update_setting, inputs=[gr.State("iti.output_dirs.good"), components['good_folder']])
    components['drafts_folder'].change(fn=update_setting, inputs=[gr.State("iti.output_dirs.drafts"), components['drafts_folder']])
    components['log_file'].change(fn=update_setting, inputs=[gr.State("iti.output.log_file"), components['log_file']])

    return components


def settings_generating(iti_manager,generator_manager, ui_code):
    components = {}
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                components['iti_sail_sampler'] = ui_code.prompt_iterator.setting_dropdown(
                    choices=g.settings_data['automa']['automa_samplers'],
                    initial_value=g.settings_data['iti']['sailing']['sail_sampler'], label='Sampler'
                )
                components['iti_sail_scheduler'] = ui_code.prompt_iterator.setting_dropdown(
                    choices=g.settings_data['automa']['automa_schedulers'],
                    initial_value=g.settings_data['iti']['sailing']['sail_scheduler'], label='Scheduler'
                )
            with gr.Row():
                components['iti_sail_checkpoint'] = ui_code.prompt_iterator.setting_dropdown(
                    choices=g.settings_data['automa']['automa_checkpoints'],
                    initial_value=g.settings_data['iti']['sailing']['sail_checkpoint'], label='Checkpoint'
                )
                components['iti_sail_vae'] = ui_code.prompt_iterator.setting_dropdown(
                    choices=g.settings_data['automa']['automa_vaes'],
                    initial_value=g.settings_data['iti']['sailing']['sail_vae'], label='VAE'
                )
            with gr.Row():
                components['iti_pos_sail_lora'] = ui_code.prompt_iterator.setting_dropdown(
                    choices=g.settings_data['automa']['automa_loras'],
                    initial_value=g.settings_data['iti']['sailing']['pos_sail_lora'], label='posLoras'
                )
                components['iti_neg_sail_lora'] = ui_code.prompt_iterator.setting_dropdown(
                    choices=g.settings_data['automa']['automa_loras'],
                    initial_value=g.settings_data['iti']['sailing']['neg_sail_lora'], label='negLoras'
                )
            with gr.Row():
                components['iti_pos_sail_embedding'] = ui_code.prompt_iterator.setting_dropdown(
                    choices=g.settings_data['automa']['automa_embeddings'],
                    initial_value=g.settings_data['iti']['sailing']['pos_sail_embedding'], label='posEmbeddings'
                )
                components['iti_neg_sail_embedding'] = ui_code.prompt_iterator.setting_dropdown(
                    choices=g.settings_data['automa']['automa_embeddings'],
                    initial_value=g.settings_data['iti']['sailing']['neg_sail_embedding'], label='negEmbeddings'
                )
        with gr.Column(scale=1):
            components['iti_refresh_button'] = gr.Button("Refresh")

    components['iti_refresh_button'].click(
        fn=generator_manager.iti_automa_refresh,
        inputs=None,
        outputs=[components['iti_sail_sampler'], components['iti_sail_checkpoint'],
                 components['iti_sail_vae'], components['iti_sail_scheduler'],
                 components['iti_pos_sail_lora'], components['iti_neg_sail_lora'],
                 components['iti_pos_sail_embedding'], components['iti_neg_sail_embedding'],
                 ]
    )
    return components




def settings_sailing():
    components = {}

    components['sail_enhance_prompt'] = gr.Checkbox(
        label="Enhance the Prompt during scoring",
        value=g.settings_data["iti"]["sailing"]["sail_enhance_prompt"]
    )


    components['enable_sailing_step'] = gr.Checkbox(
        label="Enable Sailing Step per image",
        value=g.settings_data["iti"]['sailing'].get("enable_sailing_step", False),
        interactive=True
    )
    components['enable_sailing'] = gr.Checkbox(
        label="Enable Sailing for initial prompt",
        value=g.settings_data["iti"]['sailing'].get("enable_sailing", False),
        interactive=True
    )
    components['sail_steps'] = gr.Number(
        label="Sailing Steps",
        minimum=1, step=1,
        value=g.settings_data["iti"]['sailing'].get("sail_steps", 10),
        interactive=True
    )
    components['sail_enhance_prompt'].change(
        fn=update_setting,
        inputs=[gr.State("iti.sailing.sail_enhance_prompt"), components['sail_enhance_prompt']]
    )
    components['enable_sailing_step'].change(
        fn=update_setting,
        inputs=[gr.State("iti.sailing.enable_sailing_step"), components['enable_sailing_step']]
    )
    components['enable_sailing'].change(
        fn=update_setting,
        inputs=[gr.State("iti.sailing.enable_sailing"), components['enable_sailing']]
    )
    components['sail_steps'].change(
        fn=update_setting,
        inputs=[gr.State("iti.sailing.sail_steps"), components['sail_steps']]
    )

    return components



def settings_scoring_weights(iti_manager):
    components = {'comp': {}, 'single': {}}


    with gr.Blocks(title="Scoring Weights") as scoring_ui:


        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tab('comparison Scores'):
                    # Comparison Mode Section
                    with gr.Group():
                        gr.Markdown("### Comparison Mode Weights (Sum to 1.0)")
                        for scorer in ["clip", "ssim", "color", "phash", "aesthetic", "detail", "gen_clip", "blur", "artifacts"]:
                            with gr.Row():
                                enabled = gr.Checkbox(
                                    label=f"Enable {scorer.capitalize()}",
                                    value=g.settings_data["iti"]["scorers"][scorer]["enabled"]
                                )
                                weight = gr.Slider(
                                    label=f"{scorer.capitalize()} Weight",
                                    minimum=0, maximum=1, step=0.05,
                                    value=g.settings_data["iti"]["scorers"][scorer]["weight"]
                                )
                                components['comp'][f'{scorer}_enabled'] = enabled
                                components['comp'][f'{scorer}_weight'] = weight
                with gr.Tab('single Scores'):
                    # Single Mode Section
                    with gr.Group():
                        gr.Markdown("### Single Mode Weights (Sum to 1.0)")
                        for scorer in ["aesthetic", "detail", "gen_clip", "blur", "artifacts"]:
                            with gr.Row():
                                enabled = gr.Checkbox(
                                    label=f"Enable {scorer.capitalize()}",
                                    value=g.settings_data["iti"]["scorers_single"][scorer]["enabled"]
                                )
                                weight = gr.Slider(
                                    label=f"{scorer.capitalize()} Weight",
                                    minimum=0, maximum=1, step=0.05,
                                    value=g.settings_data["iti"]["scorers_single"][scorer]["weight"]
                                )
                                components['single'][f'{scorer}_enabled'] = enabled
                                components['single'][f'{scorer}_weight'] = weight
                with gr.Tab('Settings'):
                    with gr.Row():
                        components['enhance_prompt'] = gr.Checkbox(
                            label="Enhance the Prompt during scoring",
                            value=g.settings_data["iti"]["refinement"]["enhance_prompt"]
                        )
                        components['cycle_enhance_prompt'] = gr.Checkbox(
                            label="Enhance the Prompt during model cycles",
                            value=g.settings_data["iti"]["refinement"]["cycle_enhance_prompt"]
                        )

                        components['enable_model_cycle'] = gr.Checkbox(label="Enable Model Cycles", value=g.settings_data["iti"]["refinement"]["enable_model_cycle"])
                        components['combo_size'] = gr.Number(
                            label="Size of Combo", minimum=1, step=1,
                            value=g.settings_data["iti"]["refinement"]["combo_size"],
                            interactive=True
                        )
                    with gr.Row():
                        components['improvement_threshold'] = gr.Number(
                            label="Improvement Threshold", minimum=1, step=1,
                            value=g.settings_data["iti"]["refinement"]["improvement_threshold"]
                        )
                        components['max_iter'] = gr.Number(
                            label="Max Iterations", minimum=1, step=1,
                            value=g.settings_data["iti"]["refinement"]["max_iterations"]
                        )
                        components['target_score'] = gr.Number(
                            label="Target Score", minimum=0, maximum=100, step=1,
                            value=g.settings_data["iti"]["refinement"]["target_score"]
                        )
            with gr.Column(scale=1):
                # Update Button and Status
                status = gr.Textbox(label="Status", interactive=False)
                update_btn = gr.Button("Update Weights")



        components['combo_size'].change(
            fn=update_setting,
            inputs=[gr.State("iti.refinement.combo_size"), components['combo_size']]
        )
        components['enhance_prompt'].change(
            fn=update_setting,
            inputs=[gr.State("iti.refinement.enhance_prompt"), components['enhance_prompt']]
        )
        components['cycle_enhance_prompt'].change(
            fn=update_setting,
            inputs=[gr.State("iti.refinement.cycle_enhance_prompt"), components['cycle_enhance_prompt']]
        )
        components['enable_model_cycle'].change(
            fn=update_setting,
            inputs=[gr.State("iti.refinement.enable_model_cycle"), components['enable_model_cycle']]
        )
        components['improvement_threshold'].change(
            fn=update_setting,
            inputs=[gr.State("iti.refinement.improvement_threshold"), components['improvement_threshold']]
        )
        components['max_iter'].change(
            fn=update_setting,
            inputs=[gr.State("iti.refinement.max_iterations"), components['max_iter']]
        )
        components['target_score'].change(
            fn=update_setting,
            inputs=[gr.State("iti.refinement.target_score"), components['target_score']]
        )
        comp_inputs = [components['comp'][f'{k}_{v}'] for k in ["clip", "ssim", "color", "phash", "aesthetic", "gen_clip", "detail", "blur", "artifacts"] for v in ["enabled", "weight"]]
        single_inputs = [components['single'][f'{k}_{v}'] for k in ["aesthetic", "detail", "gen_clip", "blur", "artifacts"] for v in ["enabled", "weight"]]
        # Outputs: All weight sliders + status
        comp_outputs = [components['comp'][f'{k}_weight'] for k in ["clip", "ssim", "color", "phash", "aesthetic", "detail", "gen_clip", "blur", "artifacts"]]
        single_outputs = [components['single'][f'{k}_weight'] for k in ["aesthetic", "detail", "gen_clip", "blur", "artifacts"]]

        # gr.on(
        #     triggers=[comp.change for comp in comp_inputs],
        #     fn=iti_manager.update_weights,
        #     inputs=comp_inputs + single_inputs, # Only pass the relevant inputs!
        #     outputs=comp_outputs + single_outputs + [status]
        # )
        # gr.on(
        #     triggers=[comp.change for comp in single_inputs],
        #     fn=iti_manager.update_weights,
        #     inputs=comp_inputs + single_inputs, # Only pass the relevant inputs!
        #     outputs=comp_outputs + single_outputs + [status]
        # )


        update_btn.click(
            fn=iti_manager.update_weights,
            inputs=comp_inputs + single_inputs,
            outputs=comp_outputs + single_outputs + [status]
        )

        return components


def settings_molmo():
    components = {}
    components['molmo_prompt'] = gr.Textbox(
        label="Custom Prompt",
        value=g.settings_data["iti"]["molmo"]["custom_prompt"],
        lines=10,
        placeholder="Enter your custom prompt template here..."
    )
    components['molmo_style'] = gr.Dropdown(
        label="Molmo Prompt Style", choices=["concise", "vivid", "custom"],
        value=g.settings_data["iti"]["molmo"]["prompt_style"]
    )
    components['molmo_tokens'] = gr.Slider(
        label="Molmo Max Tokens", minimum=50, maximum=400, step=10,
        value=g.settings_data["iti"]["molmo"]["max_tokens"]
    )
    components['refine_prompt'] = gr.Textbox(
        label="Refine Prompt",
        value=g.settings_data["iti"]["molmo"]["refine_prompt"],
        lines=10,
        placeholder="Enter your refine prompt template here..."
    )
    components['refine_max_tokens'] = gr.Slider(
        label="Molmo Max Tokens", minimum=50, maximum=400, step=10,
        value=g.settings_data["iti"]["molmo"]["refine_max_tokens"]
    )

    components['molmo_style'].change(fn=update_setting, inputs=[gr.State("iti.molmo.prompt_style"), components['molmo_style']])
    components['molmo_tokens'].change(fn=update_setting, inputs=[gr.State("iti.molmo.max_tokens"), components['molmo_tokens']])
    components['refine_max_tokens'].change(fn=update_setting, inputs=[gr.State("iti.molmo.refine_max_tokens"), components['refine_max_tokens']])
    components['molmo_prompt'].change(fn=update_setting, inputs=[gr.State("iti.molmo.custom_prompt"), components['molmo_prompt']])
    components['refine_prompt'].change(fn=update_setting, inputs=[gr.State("iti.molmo.refine_prompt"), components['refine_prompt']])

    return components


def settings_misc():
    components = {}



    components['automa_count'] = gr.Number(
        label="Image Count", minimum=1, step=1,
        value=g.settings_data["iti"]["automa"]["image_count"]
    )
    components['feedback_style'] = gr.Dropdown(
        label="Feedback Style", choices=["simple", "detailed"],
        value=g.settings_data["iti"]["refinement"]["feedback_style"]
    )
    components['revert_drop'] = gr.Checkbox(
        label="Revert on Drop",
        value=g.settings_data["iti"]["refinement"]["revert_on_drop"]
    )
    components['early_stop'] = gr.Number(
        label="Early Stop Gain", minimum=1, step=1,
        value=g.settings_data["iti"]["refinement"]["early_stop_gain"]
    )

    components['automa_count'].change(fn=update_setting, inputs=[gr.State("iti.automa.image_count"), components['automa_count']])
    components['feedback_style'].change(fn=update_setting, inputs=[gr.State("iti.refinement.feedback_style"), components['feedback_style']])
    components['revert_drop'].change(fn=update_setting, inputs=[gr.State("iti.refinement.revert_on_drop"), components['revert_drop']])
    components['early_stop'].change(fn=update_setting, inputs=[gr.State("iti.refinement.early_stop_gain"), components['early_stop']])


    return components


def setup_iti_tab(tab, iti_manager, generator_manager, ui_code):
    with gr.Tab("Work") as work_tab:
        work_components = work(iti_manager)
    with gr.Tab("Research") as research_tab:
        research_components = model_research(iti_manager)
    with gr.Tab("Prompting") as prompting_tab:
        prompting_components = prompting()

    with gr.Tab("Settings") as settings_tab:
        with gr.Tab("Folders") as folder_tab:
            folders_components = settings_folders()

        with gr.Tab("Generating") as gen_tab:
            generating_components = settings_generating(iti_manager, generator_manager, ui_code)

        with gr.Tab("Sailing") as sailing_tab:
            sailing_components = settings_sailing()

        with gr.Tab("Scoring") as scoring_tab:
            scoring_components = settings_scoring_weights(iti_manager)

        with gr.Tab("Molmo") as molmo_tab:
            molmo_components = settings_molmo()

        with gr.Tab("Misc") as molmo_tab:
            misc_components = settings_misc()


    # Collect all components into a single dictionary for easier access
    all_components = {
        **work_components,
        **prompting_components,
        **folders_components,
        **generating_components,
        **sailing_components,
        **scoring_components,
        **molmo_components,
        **misc_components
    }

    # Define the list of inputs that should trigger an update (similar to your example)
    input_keys = [
        # Work tab
        'input_folder',

        # Settings > Folders tab
        'good_folder', 'drafts_folder', 'log_file',
        'improvement_threshold',

        'max_iter', 'target_score',

        # Settings > Molmo tab
        'molmo_prompt', 'molmo_tokens', 'refine_prompt',
        'refine_max_tokens',

        # Miscellaneous settings
        'automa_count', 'feedback_style', 'revert_drop', 'early_stop'
    ]

    # Create the inputs list from all_components
    inputs = [all_components[key] for key in input_keys if key in all_components]

    # Attach the event handler to all these inputs
    gr.on(
        triggers=[comp.change for comp in inputs],
        fn=iti_manager.set_iti_settings,  # Assuming this is the function to call
        inputs=inputs,
        outputs=None
    )
    # Dropdown inputs
    dropdown_keys = [
        'iti_sail_sampler', 'iti_sail_scheduler', 'iti_sail_checkpoint', 'iti_sail_vae',
        'iti_pos_sail_lora', 'iti_neg_sail_lora', 'iti_pos_sail_embedding', 'iti_neg_sail_embedding',
        'pos_style', 'pos_lora', 'automa_neg', 'neg_style', 'neg_lora', 'molmo_style',
        'enable_sailing_step', 'enable_sailing', 'sail_steps'
    ]
    dropdown_inputs = [all_components[key] for key in dropdown_keys if key in all_components]
    gr.on(
        triggers=[comp.change for comp in dropdown_inputs],
        fn=iti_manager.set_gen_iti_settings,
        inputs=dropdown_inputs,  # Only pass the relevant inputs!
        outputs=None
    )


    return {
        "input_folder": work_components["input_folder"],
        "log_output": work_components["log_output"],
        "last6_running": work_components["last6_running"],
        "last6_good": work_components["last6_good"],
        "status": work_components["status"]
    }