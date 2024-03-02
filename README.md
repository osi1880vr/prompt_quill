<img src="images/pq_large.png" alt="Logo"  width="100px" height="105px" />

# Welcome to Prompt Quill

Here you find the sources to run your own instance of Prompt Quill.

A Gradio UI that will help you to create more sophisticated text to image prompts.

If you are a power prompt engineer you might not need this. It still could help to inspire you.

It is mainly useful for users that are poor prompt engineers like myself.

It helps to create beautiful prompts based on simple input like "sail to italy".

Here you find just the code to run the UI or to insert data into your instance of the vector store.


# Features

* Create prompts from your input based on similar existing prompts from the vector store.
* Select the Model to be used and set some of the parameters that get exposed by the different frameworks.
* Edit the magic Prompt that's send to make the LLM a prompt engineering helper.
* You find 3 different versions that try to offer the same service, I have implemented it in llama-index, haystack and llmware. They offer different connections to vector stores. As vector Stores it supports Milvus and Qdrant.


# No Feature

There is nothing preventing you from prompt injection or changing the way the LLM will behave. 
This is not the focus of this project as that is a whole different story to solve. 

This will become a topic once the project will find a hosting provider.


# The best feature :)

In the Character tab you will find "the magic prompt" that's the history that gets set when you enter 
your prompt to generate a new one.

Here you will get full control and you can make it as close as possible to your personal prompting style.

Right now there is just a few examples in there to make it work, but if you change the Query and Answer to your type of prompts it will tend to generate such prompts in your prefered style.
You might need to play a little with that, but during development I learned that with this you get a powerful tool to make it generate prompts in your style as close as possible.



# simple prompt vs Prompt Quill prompt

This is the prompt _rocket man in space_

<img src="images/prompt_sample/rocket_man.webp" alt="Logo"  width="500px" height="286px" />

Here we see the image the prompt from Prompt Quill created out of _rocket man in space_.

<img src="images/prompt_sample/rocket_man_pq.webp" alt="Logo"  width="500px" height="286px" />

The full prompt from Prompt Quill was:

_High-quality digital art, ultra-detailed, professional, clear, high contrast, high saturation, vivid deep blacks, crystal clear, ((rocket man in space)), wearing a full helmet and leather jacket, leather gloves, standing in front of an advanced high-tech space rocket, surrounded by the vastness of outer space, with intense, vibrant colors, colorful, dark, modern art style, the rocket illuminated by the cosmic light, the rocketman standing solo against the cosmic backdrop, bokeh effect creating a blurry background, photography-style composition, on eye level, masterpiece._


# robust prompts even with no negative prompt
Here you see images that are a hand made prompt and a prompt made by Propmt Quill

the left is the Prompt Quill prompt and the right is the hand made prompt, they use the same settings and seed

also they use a very detailed handcrafted negative prompt, the right image is the more nicer one

<img src="images/neg_sample/left_1.webp" alt="Logo"  width="329px" height="586px" /><img src="images/neg_sample/right_1.webp" alt="Logo"  width="329px" height="586px" />

NBow we see same prompts same seed but with no negative prompt and we can see how the right one drifts away from the first one whiole the left also drifts but not as hard

<img src="images/neg_sample/left_2.webp" alt="Logo"  width="329px" height="586px" /><img src="images/neg_sample/right_2.webp" alt="Logo"  width="329px" height="586px" />


# The data needed for all the fun

To get data into your prompt quill there are two ways: the hard and the easy one ;)

The hard way is to go and get a large number of prompts and put them into a vector store. The scripts to do so are included.

For the easy way, just download more than 1.5 million prompts ready to go here: https://civitai.com/models/330412


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

If you did download a snapshot from civitai you have to run the Qdrant vectror store.
A docker compose file you will find in the docker folder

just cd to the qdrant folder and run:

docker compose up

this will start the qdrant server.
Once it is up and running you should find it a http://localhost:6333 .
Under collections there is a little blue arrow right and there you can upload the snapshot.
Once that is done you are ready to go to get the prompt quill running.

get into the folder of any of the following brands you like to run on:
llama_index, llmware or haystack

run pip install -r .\requirements.txt.


Unless you use llmware you have to do this following steps to run llama-cpp:

and finally if you like to run on GPU you have to setup llama-cpp and torch to run on GPU for your environment
to do this if on windows please check the file in llama-cpp_windows it will tell the further steps

On any other platform please find how you do it, I do not have any other platform, if you find out please let me know and I add it here.

If you only run on CPU the last steps is

run pip install -r .\requirements_cpu.txt


longterm I will create a one click installer but thats not today ;)








