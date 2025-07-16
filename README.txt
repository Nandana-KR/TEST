#  Vision-Language Integration App

This project is a Python-based application that combines **Computer Vision** and **Natural Language Generation** using a Large Language Model (LLM). It accepts an image and a text prompt as input, detects objects in the image, and generates a coherent textual response that relates to both the visual content and the user prompt.

---
 Features

-  **Object Detection**: Detects multiple objects in an image with YOLOv5.
-  **Text Generation**: Uses Falcon LLM (`tiiuae/falcon-rw-1b`) to generate meaningful responses.
-  **Multimodal Integration**: Connects computer vision and language models into one seamless pipeline.
-  **User-Friendly Interface**: CLI-based inputs with clear prompts.
-  **Robust Error Handling**: Gracefully manages invalid files or exceptions.

---

##  Sample Usage

```bash
$ python app.py

Image path: dog.jpg
Text prompt: What can you tell about this scene?

Detected objects:
- dog: 0.94
- ball: 0.81

=== Model Response ===
This image likely depicts a playful dog with a ball, possibly in an outdoor setting. It seems to be an active and joyful scene.
