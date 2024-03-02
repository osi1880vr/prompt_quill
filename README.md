# Welcome to Prompt Quill

Here you find the sources to run your own instance of Prompt Quill.

A Gradio UI that will help you to create more sophisticated text to image prompts.

If you are a power prompt engineer you might not need this. It still could help to inspire you.

It is manly useful for users that are poor prompt engineers like myself.

It helps create beautiful prompts based on simple input like "sail to italy".

Here you find just the code to run the UI or to insert data into your instance of the vector store.


# Features

* Create prompts from your input based on similar existing prompts from the vector store.
* Select the Model to be used and set some of the parameters that get exposed by the different frameworks.
* Edit the magic Prompt that's send to make the LLM a prompt engineering helper.
* You find 3 different versions that try to offer the same service, I have implemented it in llama-index, haystack and llmware. They offer different connections to vector stores. As vector Stores it supports Milvus and Qdrant.


# No Feature

There is nothing preventing you from prompt injection or change the way the LLM will behave. 
This is not focus of this project as that is a whole different story to solve. 

This will become a topic once the project will find a hosting provider.


# The best feature :)

In the Character tab you will find "the magic prompt" that's the history that gets set when you enter 
your prompt to generate a new one.

Here you get full control about to make it as close as possible to your personal prompting style.

Right now there is just a few examples in there to make it work, but if you change the Query and Answer to your type of prompts it will tend to generate such prompts in your prefered style.
You might need to play a little with that, but during development I learned that with this you get a powerfull tool to make it generate prompts in your style as close as possible.


# Roadmap

* Add more models to the model lists.
* Add negative prompting.
* Add model / LORA advisor.
* Add history to the conversation if not already included by the framework.
* Add longterm history by storing conversations to disk.
* Add more settings to finetune the vector search.
* Add REST API
* Build Comfyui Plugin
* Find someone do a plugin for Auto1111 :D


# Looking for a hosting solution

If you like the idea, and you are able to sponsor to longterm host this including the data feel free to contact me.


# More prompts :)

If you like and can provide large numbers of prompts please get in contact.
I like to compile a growing vector DB which I then like to share at some place where I can upload those files.

# Contact

You can find me on discord: https://discord.gg/gMDTAwfQAP

# Install

to run this thing you need to decide which brand you like most, 
than you got to setup a vector store and start playing with it.

If you did download a snapshot from civitai you have to run the Qdrant vectror store
A docker compose file you will find in the docker folder

just cd to the qdrant folder and run:

docker compose up

this will start the qdrant server.
Once it is up and running you should find it a http://localhost:6333 
Under collections there is a little blue arrow right and there you can upload the snapshot
Once that is done you are ready to go to get the prompt quill running

get into the folder of your brand you like to run
llama_index, llmware or haystack

run pip install -r .\requirements.txt


any other than llmware you have to do this following steps:
and finaly if you like to run on GPU you have to setup llama-cpp and torch to run on GPU for your environment
to do this if on windows please check the file in llama-cpp_windows it will tell the further steps

on any other platform please find how you do it, I do not have any other platform, if you find out please let me know and I add it here

if you only run on CPU the last steps is

run pip install -r .\requirements_cpu.txt


longterm I will create a one click installer but thats not today ;)








