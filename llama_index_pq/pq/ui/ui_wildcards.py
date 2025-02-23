import gradio as gr
import os
from pathlib import Path
import globals

g = globals.get_globals()
WILDCARD_DIR = Path("wildcards")

# Ensure the wildcard directory exists
if not WILDCARD_DIR.exists():
    WILDCARD_DIR.mkdir(parents=True)

# Initialize wildcard cache in settings_data
if 'wildcard_cache' not in g.settings_data:
    g.settings_data['wildcard_cache'] = {}
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

# --- Shared Functions ---
def get_subfolders():
    """Return a list of subfolders in WILDCARD_DIR."""
    subfolders = [""]
    for subdir in WILDCARD_DIR.glob("**/"):
        if subdir.is_dir() and subdir != WILDCARD_DIR:
            subfolders.append(str(subdir.relative_to(WILDCARD_DIR)))
    return subfolders

# --- Search Tab Functions ---
def search_wildcard_files(search_text):
    if not search_text or not WILDCARD_DIR.exists():
        return [], "No search text provided or wildcard directory not found.", ""

    results = []
    for file_path in WILDCARD_DIR.glob("**/*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if search_text.lower() in content.lower():
                    relative_path = file_path.relative_to(WILDCARD_DIR)
                    wildcard_syntax = f"__{str(relative_path).replace(os.sep, '/').replace('.txt', '')}__"
                    results.append(f"{wildcard_syntax} ({file_path})")
        except Exception as e:
            results.append(f"Error reading {file_path}: {str(e)}")

    if not results:
        return [], "No files found in wildcards/ or its subfolders.", ""
    return results, f"Found {len(results)} files across subfolders. Select to view or edit.", ""

def get_file_content(selected_file):
    if not selected_file:
        return "No file selected."

    try:
        file_path = selected_file.split("(", 1)[1].rstrip(")")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def save_edited_file(selected_file, new_content):
    if not selected_file:
        return "Please select a file to edit."

    try:
        file_path = selected_file.split("(", 1)[1].rstrip(")")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        wildcard_syntax = selected_file.split(" (")[0]
        g.settings_data['wildcard_cache'][wildcard_syntax] = new_content.lower()
        return f"Changes saved to '{file_path}'"
    except Exception as e:
        return f"Error saving changes: {str(e)}"

def update_search_results(search_text):
    choices, status, _ = search_wildcard_files(search_text)
    return (
        gr.update(choices=choices, value=None),
        status,
        "No file selected."
    )

# --- Create Tab Functions ---
def save_wildcard_file(filename, content, subfolder, overwrite):
    """Save a new wildcard file, hinting at refresh if a new subfolder is created."""
    if not filename:
        return "Please provide a filename."

    filename = filename.strip()
    if not filename.endswith(".txt"):
        filename += ".txt"

    subfolder = subfolder.strip().strip("/\\") if subfolder else ""
    target_dir = WILDCARD_DIR if not subfolder else WILDCARD_DIR / subfolder
    target_path = target_dir / filename

    is_new_folder = not target_dir.exists()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return f"Error creating subfolder '{target_dir}': {str(e)}"

    if target_path.exists() and not overwrite:
        return f"File '{target_path}' already exists. Check 'Overwrite' to replace it."

    try:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        relative_path = target_path.relative_to(WILDCARD_DIR)
        wildcard_syntax = f"__{str(relative_path).replace(os.sep, '/').replace('.txt', '')}__"
        g.settings_data['wildcard_cache'][wildcard_syntax] = content.lower()
        if is_new_folder and subfolder:
            return f"New subfolder '{subfolder}' created! Click 'Refresh Subfolders' to update. File saved: {target_path}"
        return f"File saved successfully: {target_path}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

def update_subfolder_text(selected_subfolder):
    """Update the subfolder textbox when a dropdown option is selected."""
    return selected_subfolder if selected_subfolder else ""

def refresh_subfolders():
    """Refresh the subfolder dropdown."""
    return gr.update(choices=get_subfolders())

def create_wildcard_ui():
    # Search Tab (with Editing)
    with gr.Tab("Search"):
        gr.Markdown("### Search & Edit Wildcard Files")
        gr.Markdown("Search for files by content (press Enter or click Search), then view or edit the selected file.")
        with gr.Row():
            with gr.Column(scale=3):
                search_input = gr.Textbox(label="Search Text", placeholder="Type text and press Enter to search...")
                search_results = gr.Dropdown(label="Matching Files", choices=[], interactive=True)
                file_content = gr.Textbox(label="File Content (Editable)", lines=10, placeholder="Select a file to view or edit its content...")
            with gr.Column(scale=1):
                status_output = gr.Textbox(label="Status", placeholder="Results will appear here...", interactive=False)
                search_button = gr.Button("Search")
                save_edit_button = gr.Button("Save Changes")

        search_input.submit(fn=update_search_results, inputs=search_input, outputs=[search_results, status_output, file_content])
        search_button.click(fn=update_search_results, inputs=search_input, outputs=[search_results, status_output, file_content])
        search_results.change(fn=get_file_content, inputs=search_results, outputs=file_content)
        save_edit_button.click(fn=save_edited_file, inputs=[search_results, file_content], outputs=status_output)

    # Create Tab
    with gr.Tab("Create"):
        gr.Markdown("### Create New Wildcard File")
        gr.Markdown("Enter a filename and content. Choose or type a subfolder (new or existing).")
        with gr.Row():
            with gr.Column(scale=3):
                filename_input = gr.Textbox(label="Filename", placeholder="e.g., my_wildcard.txt")
                subfolder_dropdown = gr.Dropdown(label="Existing Subfolders", choices=get_subfolders(), value="", interactive=True)
                subfolder_input = gr.Textbox(label="Subfolder Path", placeholder="e.g., folder/subfolder (type new or edit)", value="")
                content_input = gr.Textbox(label="Content", lines=10, placeholder="Enter the wildcard content here...")
            with gr.Column(scale=1):
                create_status = gr.Textbox(label="Status", placeholder="Status will appear here...", interactive=False)
                overwrite_checkbox = gr.Checkbox(label="Overwrite if exists", value=False)
                save_button = gr.Button("Save")
                refresh_button = gr.Button("Refresh Subfolders")

        subfolder_dropdown.change(fn=update_subfolder_text, inputs=subfolder_dropdown, outputs=subfolder_input)
        save_button.click(fn=save_wildcard_file, inputs=[filename_input, content_input, subfolder_input, overwrite_checkbox], outputs=create_status)
        refresh_button.click(fn=refresh_subfolders, inputs=None, outputs=subfolder_dropdown)

def setup_wildcards_tab():
    create_wildcard_ui()

if __name__ == "__main__":
    with gr.Blocks(title="Wildcard Maintenance") as demo:
        with gr.Tab("Wildcards"):
            create_wildcard_ui()
    demo.launch()