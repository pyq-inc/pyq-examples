# pyq-examples
Examples of projects that can be deployed using pyq. This includes a simple Hello World to help you get started, as well as examples of HuggingFace and Scikit-Learn models, accepting files and/or JSON as input. You can, of course, use other model frameworks based on your needs.

All of these models can be deployed using pyq and accessed via our API.

Examples include:
1. Google ViT from HuggingFace, which is an image classifier. The example that accepts files allows the user to send an image to the model directly, while the JSON version accepts a URL to a publicly available image. Source: https://huggingface.co/google/vit-base-patch16-224
2. Bloom from HuggingFace, an LLM which does text generation. It accepts JSON text input and returns text input. This is a big model. Source: https://huggingface.co/bigscience/bloom-1b7
3. A handwriting recognition model built using scikit-learn. It can recognize handwritten numbers and accepts PNG files. Source: https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
4. A regression model built using scikit-learn. It can predict Boston house prices and accepts a list of integers as input via a JSON object. Source: https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset
5. Whisper by OpenAI. It is an audio transcription and translation model with outstanding results. Accepts an mp4 recording of one or more people speaking and returns a timestamped transcript of the file. Source: https://github.com/openai/whisper
