## MacOS Installation (using Docker and Llama-Index):

The following instructions are for installing Prompt Quill using Docker, Llama-Index and QDrant on an Apple Silicon machine.

### First Steps
1. Download the `lli_3.2M_v1.1` vector data from CivitAI here, and extract the snapshot file (doesn't matter where) https://civitai.com/models/330412?modelVersionId=407093
2. Using terminal, `cd` to the path you want your Prompt Quill folder to be in.
3. Run `git clone https://github.com/osi1880vr/prompt_quill` to download the latest Prompt Quill files. This will create a folder called `prompt_quill` wherever you currently are in the terminal shell.

### Docker/Llama-Index/qdrant Setup
1. If you don't already have it, install Docker Desktop: https://docs.docker.com/desktop/install/mac-install/
    - Once installed, close and reopen Terminal to ensure it recognizes Docker being added to your machine.
2. From the `prompt_quill` folder you cloned Prompt Quill into, `cd docker/qdrant` to get to the folder containing the `docker_compose.yaml` file.
3. `docker compose up` to create the new docker container for qdrant.
4. Once the docker container is installed (you should see it running in Docker Desktop's dashboard), open a browser tab to http://localhost:6333/dashboard
5. Click `Upload Snapshot`, name the new Collection `prompts_large_meta`, then select the .snapshot file you downloaded and unzipped from CivitAI during the First Steps. Wait for the import to complete before continuing.
    - Once complete, you can free up space by deleting the .snapshot file and the .zip you downloaded initially, as it's been imported into qdrant and is no longer required for Prompt Quill.

### Prompt Quill Installation
1. In a new terminal window, `cd` to your `prompt_quill` folder.
2. `python3 -m venv env` - This creates a new virtual environment called 'env'.
3. `source ./venv/bin/activate` - This activates the virtual environment.
4. `cd llama_index_pq`
5. `pip install -r ./requirements_apple_silicon.txt` to install the necessary dependencies for Prompt Quill.
6. Start Prompt Quill with `python3 pq/prompt_quill_ui_qdrant.py`, it should open into a browser window automatically. You can also open it yourself at http://localhost:1337 if necessary.
7. To shutdown, simply close the terminal window that has the Prompt Quill logs active, and stop the Docker container for qdrant.

## Starting Prompt Quill

Once you have Prompt Quill installed, to do a clean startup in the future follow these steps:

1. Start the qdrant Docker container in Docker Desktop.
2. Open Terminal and `cd` to your `prompt_quill` folder.
3. `source ./venv/bin/activate` to enable the virtual environment.
4. `python3 pq/prompt_quill_ui_qdrant.py` to start Prompt Quill.
5. Open the UI at http://localhost:1337 if it doesn't auto-launch.