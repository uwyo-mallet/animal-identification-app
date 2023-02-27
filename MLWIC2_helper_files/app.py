# app.py
# Chet Russell
# Last edited: Feb 27

import gradio as gr
import tempfile
import shutil
import os

def classify(images):
    # Create temporary folder and put temporary images in it.
    with tempfile.TemporaryDirectory() as tmpdirname:
        for image in images:
            shutil.move(image.name, tmpdirname)
    # Classify images. Probably not the best way to do it.
    os.system('')

title = "Animal Classification"
preview = gr.Interface(fn=classify, 
             title=title,
             inputs=gr.inputs.File(file_count='multiple'), 
             outputs="image")

if __name__ == "__main__":
    preview.launch()