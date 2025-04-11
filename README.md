

# Welcome to Prompt Quill <img src="images/pq_large.png" alt="Logo"  width="100px" height="105px" />

![Static Badge](https://img.shields.io/badge/python-3.9_%7C_3.10%7C_3.11-blue?color=blue)
[![fazier](https://fazier.com/api/v1/public/badges/embed_image.svg?launch_id=229&badge_type=daily)](https://fazier.com/launches/prompt-quill)

I will stop sending updates. 
It feels like a dead end since there is no input coming only requests, that's the opposite of why I do open source.
The latest version is now moving into a fully automated production suite, and so I think since I don't get back any value here I keep pushing it on my private repo.
My view might change as soon people start contributing code instead of complaints. 
I'm tired of fixing non issues on local installs that are not caused by PQ itself but by the local systems.


Prompt Quill was created to help users make better prompts for creating AI images.

Even if you are an expert, it could still be used to inspire other prompts.

It is useful for poor prompt engineers like me who struggle with coming up with all the detailed instructions that are needed to create beautiful images using models like Stable Diffusion or other image generators.

The **world's first RAG driven prompt engineer** helper at this large scale. Use it with more than 3.2 million prompts in the vector store. This number will keep growing as I plan to release ever-growing vector stores when they are available.

The Gradio UI will also help you to create more sophisticated text to image prompts.

Here you find just the code to run the UI or to insert data into your instance of the vector store.

There is no hardcoded need to use Docker at all, I understand this scares a lot of people. Docker is only here as it is a convenient way to run the vector stores, but it is easy to run them without docker too. Prompt Quill itself does not need Docker to run.

With the latest faeture added it also adds a way for automated model testing if you are into training models.

# Features

* One Click installer available for all version (windows)
* Create prompts from your input based on similar existing prompts from the vector store.
* Select the LLM to be used and set some of the parameters that get exposed by the different frameworks. There is now a large list of LLMs available.
* Edit the Magic Prompt that's sent to make the LLM a prompt engineering helper.
* I have implemented Prompt Quill using llmware, llama-index. They offer different connections to vector stores. (We currently support Milvus and Qdrant.)
* Translate, you can enable translate and then enter your prompt in your native language, it will then get translated using Google Translate to english. The translated prompt is shown in the output so you can see what it was tranbslated to.
* Deep Dive, this will allow you to see the context for your input prompt. when you click a line it will fetch the context as if that would be your input, also it sets the clicked one as the prompt for prompt generation.
* Deep dive also allows for direct search, you can type into the search field and this will trigger lookups for your entered text.
* batch mode, this will create a prompt for each result from the context as a input prompt. this will help to give you a more broad idea of what your input could get to. Be a little patient as it will take more time to generate.
* Image generation via the API of auto1111 or forge
* Sail the data Ocean, this allows you to sail along the vector distance through the ocean of prompts. It allows you to take an automated journey and see the seven seas of data pushing out nice prompts you may not have the idea to even think about. It also allows you to direct create images using the a1111/Forge API. Ahoy sailor, fair winds and following seas.
* Model tester, this feature will allow you to have your newly trained model tested on some real world prompts. During training you tend to check on the data you trained and only spend little time to see the normal usage of the model. It can help you to see the tipping point when your model starts drifting away for normal usage. So it will help you to save time and maybe money if you are paying for GPU resources.
* Moondream interrogation. get a description of any image and also ask questions about the image. A really magic thing, it also can give advice hoe to improve the image.

# The best feature :)

In the Character tab you will find "the magic prompt", which is the history that gets set when you enter your prompt to generate a new one.

Here you will get full control, and you can make it as close as possible to your personal prompting style.

Right now there are just a few examples, but if you change the Query and Answer to your type of prompts, it will generate prompts corresponding to your preferred style.
You might need to experiment with this but during development, I learned that with this you can get a powerful tool to make it generate prompts as close as possible to your style.

There is now 2 different templates that will show how much difference you can get by changing the character. template_a tends to create story telling prompts, while template_b will create lists of concepts as most of SD prompters do.

# Simple prompt vs Prompt Quill prompt

This is the prompt _rocket man in space_

<img src="images/prompt_sample/rocket_man.webp" alt="spaceman"  width="500px" height="286px" />

Here we can see the image the prompt from Prompt Quill created out of _rocket man in space_.

<img src="images/prompt_sample/rocket_man_pq.webp" alt="spaceman"  width="500px" height="286px" />

The full prompt from Prompt Quill was:

_High-quality digital art, ultra-detailed, professional, clear, high contrast, high saturation, vivid deep blacks, crystal clear, ((rocket man in space)), wearing a full helmet and leather jacket, leather gloves, standing in front of an advanced high-tech space rocket, surrounded by the vastness of outer space, with intense, vibrant colors, colorful, dark, modern art style, the rocket illuminated by the cosmic light, the rocketman standing solo against the cosmic backdrop, bokeh effect creating a blurry background, photography-style composition, on eye level, masterpiece._

Looking at this prompt, keep in mind that you get full control about the prompt style by editing the history in the character tab.

You might wonder where the rocket is from the prompt. It is clearly not in the image and maybe it is hiding behind the guy in front.
The outcome depends on the model you use and on how many samples you take using different seeds. The example you see here is the one I liked the most from the few samples I took.
Also, the image dimensions used will have a large impact on what gets shown. A wide image will look totally different than a high image. (But that's AI image generation theories which are not part of this here.) Here, it's all just about making nicer prompts. It's up to you to create the right image setup for your prompts.
Here are a few more samples using a different model.

<img src="images/space_man/1.webp" alt="spaceman"  width="250" height="176" /><img src="images/space_man/2.webp" alt="spaceman"  width="250" height="176" /><img src="images/space_man/3.webp" alt="spaceman"  width="250" height="176" />


# one click install demo video (click the image ;)

[![Prompt Quill on youtube](images/video/video1.png)](https://youtu.be/FjvV4-5MU9k "Prompt Quill installed in one click")



# Robust prompts even with no negative prompt
Here you see images comparing a regular, hand made prompt versus a prompt made by Prompt Quill.

The left is the Prompt Quill prompt and the right is the hand made prompt, they use the same settings and seed.

Also they use a very detailed handcrafted negative prompt, the right image is the nicer one.

<img src="images/neg_sample/left_1.webp" alt="Logo"  width="329px" height="586px" /><img src="images/neg_sample/right_1.webp" alt="Logo"  width="329px" height="586px" />

Now we see same prompts, same seed but with no negative prompt, and we can see how much the right one drifts away from the first one while the left drifts not as hard.

<img src="images/neg_sample/left_2.webp" alt="Logo"  width="329px" height="586px" /><img src="images/neg_sample/right_2.webp" alt="Logo"  width="329px" height="586px" />


# the sail to italy example

_"sail to italy"_ would give you a prompt like this:

_Stunning aerial photograph of a sailboat gracefully navigating through the crystal-clear waters of the Mediterranean Sea, en route to Italy. The sun casts a warm golden glow on the boat, while the distant villages and medieval towns along the coastline appear as intricate, detailed miniatures in the background. The image captures the beauty of the Italian coastline, with its picturesque beaches, islands, and castles, all bathed in the warm, golden light of the setting sun. The sailboat's sails billow with the wind, creating a sense of movement and adventure in this breathtaking scene. The photograph is taken in 8K resolution, offering an incredibly detailed and immersive view of the scene, with the stars twinkling in the night sky above, creating an Aurora-style atmosphere that adds to the magic of the moment._

Here are a few samples from 8 different free online image generators:
We see the prompts are not just working for stable diffusion models.


<img src="images/sail_to_italy/1.jpg" alt="sailing"  width="400" height="400" /><img src="images/sail_to_italy/2.webp" alt="sailing"  width="400" height="400" />
<img src="images/sail_to_italy/3.jpg" alt="sailing"  width="266" height="266" /><img src="images/sail_to_italy/4.jpg" alt="sailing"  width="266" height="266" /><img src="images/sail_to_italy/5.png" alt="sailing"  width="266" height="266" />
<img src="images/sail_to_italy/6.jpg" alt="sailing"  width="266" height="266" /><img src="images/sail_to_italy/7.jpg" alt="sailing"  width="266" height="266" /><img src="images/sail_to_italy/8.jpg" alt="sailing"  width="266" height="266" />

# negative prompts and model examples

Based on the information I retrieved for all those prompts data I also know the negative prompt that was used for a prompt and also I know what model was used to generate the image I took the prompt from, So why not give that information to you too.
I managed to get this working in llama-index and haystack, the data is available on civitai.

<img src="images/neg_prompts/neg_prompt_sneak_peek.png" alt="negative prompts"  width="1348" height="627" />

Here you see an example "a nice cat" with no negative prompt on the left and on the right same prompt and seed but with negative prompt

<img src="images/neg_prompts/no_neg.png" alt="sailing"  width="266" height="266" /><img src="images/neg_prompts/with_neg.png" alt="sailing"  width="266" height="266" />

# Image generation

I added the free online image generations from civitai and hordeai  to allow you to get a first idea of where the prompt might go once you run it inside your image generation tool.
Its easy to use but please be aware that your local results might be much nicer than what you get from that online things. In both cases the number of models you can use is quite limited.
But still it gives you a first idea of how it might go.

<img src="images/image_gen/dogNcat.png" alt="sailing"  width="1369" height="754" />



# The data needed for all the fun

To get data into your prompt quill there are two ways: the hard and the easy one ;)

The hard way is to go and get a large number of prompts and put them into a vector store. The scripts to do so are included.

For the easy way, you just run the one click install he will do all you need to run your prompts
The data can be found here also: https://civitai.com/models/330412


# Roadmap

* Add more models to the model lists. (Done)
* Add negative prompting. (Done)
* Add model / LORA advisor. (Done)
* Add history to the conversation if not already included by the framework.
* Add longterm history by storing conversations to disk.
* Add more settings to finetune the vector search. (Done)
* Add REST API (Done)
* Build Comfyui Plugin (Done)
* Find someone do a plugin for Auto1111 :D


# Looking for a hosting solution

If you like this project, and you are willing and able to sponsor this project as a longterm host (this including the data), feel free to contact me.


# More prompts :)

If you like and can provide large numbers of prompts please get in contact with me.
I am compiling a growing vector DB which I can then share at some place where I can upload those files.


# Install

If you are on windows, you can use the one_click_install.bat.
Since this is downloading very large files it can happen that downloads fail.
You can use prepare_download_cache.bat to download the files and store them in a cache folder.
The one_click_install.bat will check if the files are there and does not download them.



# No Feature

There is nothing preventing you from prompt injection or changing the way the LLM will behave.
This is not the focus of this project as that is a whole different story to solve.

This will become a topic once there is a hosting provider for the project.


# Contact

Please find me and join us on discord: https://discord.gg/gMDTAwfQAP


[![Star History Chart](https://api.star-history.com/svg?repos=osi1880vr/prompt_quill&type=Date)](https://star-history.com/#osi1880vr/prompt_quill&Date)




