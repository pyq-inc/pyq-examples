# pyq-examples
Examples of projects that can be deployed using pyq. This includes a simple Hello World to help you get started, as well as examples of HuggingFace and Scikit-Learn models, accepting files and/or JSON as input. You can, of course, use other model frameworks based on your needs.

All of these models can be deployed using pyq and accessed via our API.  Our model zoo https://pyqai.com also has a variety of open source models already deployed. 

Examples include:
1. Bloom from HuggingFace, an LLM which does text generation. It accepts JSON text input. Source: https://huggingface.co/bigscience/bloom-1b7

2. Whisper by OpenAI. It is an audio transcription and translation model with outstanding results. Accepts an m4a recording of one or more people speaking and returns a timestamped transcript of the file. Source: https://github.com/openai/whisper  This model can also be called 
directly from our model zoo - check it out here : https://dev.pyqai.com/zoo/models/11

3. HelloWorld by us :). It is an blank template that you can use to call any model you find interesting. 

4. Bart by Facebook.  This model classifies text.  It takes in some input as well as a list of categories and returns how well the text matches each category.  Source: https://huggingface.co/facebook/bart-large-mnli. This model can also be called directly from our model zoo - check it out here : https://dev.pyqai.com/zoo/models/5

5. Image captioner from HuggingFace.  This model takes in a jpeg or png, and will output a brief description of what is in the image.  https://huggingface.co/Vasanth/image_captioner_vit_gpt2. This model can also be called directly from our model zoo - check it out here : https://dev.pyqai.com/zoo/models/2 


