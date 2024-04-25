<h1 align="center">Mallet Lab Animal Identification</h1>

## Info

This application was written by the [Mallet Lab](https://www.mallet.ai/) to process game camera images. The code provided includes an updated [MLWIC2](https://github.com/mikeyEcology/MLWIC2) model in python 3 and a user interface implemented through [Gradio](https://github.com/gradio-app/gradio). This app specifically is made to process images from Reconyx and Browning trail cameras but can be used to process images from any camera.

## Usage

The MLWIC2_Python3 folder contains the updated MLWIC2 python 3 model. Running app.py will host a session of Gradio through a web address, where you will be able to upload your images in a zip folder format then process them. After processing, you will have an option to download the processed images, or view them in the viewer built into Gradio.

Note: this application is meant to be hosted on a machine for people to utilize through their web browser, rather than being hosted on a personal machine. It can be hosted on a personal machine just fine, as long as you connect to the address Gradio outputs in the terminal.

## Model Specifications

As mentioned above, the model that is being utilized is the [MLWIC2](https://github.com/mikeyEcology/MLWIC2) model, updated in python 3. Tensorflow version 1.14 is being utilized, so if you would like to switch out this model to another, you would need to modify the app.py file. Line 117 contains the dictionary values for the MLWIC2 model, and there are a couple calls to tensorflow above line 117. These will have to be modified to accomodate for a different model.
