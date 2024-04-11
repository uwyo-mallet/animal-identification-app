# app.py
# Chet Russell
# Last edited: April 11, 2024

import gradio as gr
import os
import csv
import shutil
import run
import tensorflow as tf
import utils
from argparse import Namespace
import extract
import pandas as pd
import tempfile
from collections import defaultdict
import math
import zipfile

parser = run.gen_argparser()

# Deletes all files in a directory
def clean(directory):
    shutil.rmtree(directory)
    if not os.path.isdir(directory):
        os.makedirs(directory)

# Found on stackoverflow
def fix_zip_file(zipFileContainer):
    # Read the contents of the file
    content = zipFileContainer.read()
    pos = content.rfind(b'\x50\x4b\x05\x06')  # reverse find: this string of bytes is the end of the zip's central directory.
    if pos>0:  # Double check we're not at the beginning of the file so we don't blank out the file
        zipFileContainer.seek(pos+20)  # Seek to +20 in the file; see secion V.I in 'ZIP format' link above.
        zipFileContainer.truncate()  # Delete everything that comes after our current position in the file (where we `seek` to above).
        zipFileContainer.write(b'\x00\x00') # Zip file comment length: 0 byte length; tell zip applications to stop reading.
        zipFileContainer.seek(0)  # Go back to the beginning of the file so the contents can be read again.

    return zipFileContainer  # return the file handle even if we didn't make any changes (`pos` was zero)

# Unzipping function, eventually calls the classify
def unzip(f, progress=gr.Progress()):
    # Progress bar call
    progress(0, desc="Unzipping")

    # Declaring where folders are located
    zipfolder = "./zipfolder/"
    original = "./original_images/"

    # Removing previous images and files from folders
    clean(zipfolder)
    clean(original)

    # Grabbing name of file from tempfilewrapper
    name = f.name
    print(name)

    # Copying file to the zipfolder to unzip
    shutil.copy(name, zipfolder)

    # Opening zipfile and fixing potential errors in zipfile
    zip = open(zipfolder + name.split('/')[-1], 'r+b')  # 'r+b' where 'r+' is read+write and 'b' is binary
    zip = fix_zip_file(zip)
    print(zipfile.is_zipfile(f.name))

    # Unzipping
    with zipfile.ZipFile(zip, 'r') as zip_ref:
        zip_ref.extractall(original)

    # Grabbing directory names
    dirnames = []
    for (original, dir_names, file_names) in os.walk(original):
       dirnames.extend(dir_names)
    print(dirnames)

    # Grabbing file names
    res = []
    for (original, dir_names, file_names) in os.walk(original):
        res.extend(file_names)
    print(res)

    # Classify call here
    return classify(res, original + "/", progress)

# Main function to classify images
def classify(images, unzipped, progress):

    
    total = 0 # needed for progress bar
    original = unzipped # where the unzipped images are
    resized = "./resized_images/" # where the resized images will be placed
    temperature = "./temp_folder/" # where the temperature values extracted by tesseract will be put

    # Clean images folder
    clean(resized)
    clean(temperature)

    # Counting amount of steps for progress bar
    for image in images:
        total += 1
        print(image)

    ims = []
    metadata = defaultdict(list) # creating dict for image metadata

    with open("images.txt", "w") as f:
        # Loop for progress bar
        for index, image in enumerate(images, start=1):
            progress((index) * (.5/total), desc="Image Preprocessing") # actual incrementation of progress bar
            final_image = image # declaration of separate image variable
            f.write(final_image)
            f.write("\n")
            ims.append(final_image)
            print(final_image)

            extract.crop(original + final_image, final_image, resized) # cropping image to fit with parameters of model
            extract.im_meta_data(original + final_image, final_image, temperature) # gathering temperature metadata of image
            # Extracting the rest of the metadata of image
            extract.meta_dict(
                original + final_image, final_image, original, temperature, metadata
            )

    print(ims)

    # Tensorflow part

    tf.reset_default_graph() # readying tensorflow model
    args = parser.parse_args(["inference"]) # setting model to inference mode
    dict_args = vars(args) # gathering model arguments

    # Setting the model arguments
    dict_args["path_prefix"] = resized
    dict_args["log_dir"] = "./species_model"
    dict_args["snapshot_prefix"] = "./species_model"
    dict_args["depth"] = 18
    dict_args["val_info"] = "./images.txt"
    batch_size = math.ceil(len(ims)/100)
    dict_args["batch_size"] = batch_size

    # Total amount of steps for the model to process
    total_steps = len(ims)/batch_size

    # Tensorflow declarations
    namespace_args = Namespace(**dict_args)
    (
        namespace_args.num_val_samples,
        namespace_args.num_val_batches,
    ) = utils.count_input_records(args.val_info, args.batch_size)
    namespace_args.inference_only = True

    # Logging the runtime information if requested
    if args.log_debug_info:
        namespace_args.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        namespace_args.run_metadata = tf.RunMetadata()
    else:
        namespace_args.run_options = None
        namespace_args.run_metadata = None

    sess = tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=namespace_args.log_device_placement,
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
        )
    )

    run.do_evaluate(sess, namespace_args, progress, total_steps)
    # Done with model

    # Starting post-processing

    imagedata = {}

    # Dictionary associating the names of animals with the integer value of the model
    # NOTE: this will have to change depending on the model that is being used.
    MLWIC2_names = {
        0: "Moose",
        1: "Cow",
        2: "Quail",
        3: "Coyote",
        4: "Elk",
        5: "American_marten",
        6: "American_crow",
        7: "Armadillo",
        8: "Wild_turkey",
        9: "Opossum",
        10: "Horse",
        11: "Human",
        12: "Sylvilagus_family",
        13: "Bobcat",
        14: "Striped_skunk",
        15: "Dog",
        16: "Cricetidae_muridae_families",
        17: "Mule_deer",
        18: "White-tailed-deer",
        19: "Raccoon",
        20: "Mountain_lion",
        21: "California_ground_squirrel",
        22: "Wild_pig",
        23: "Grey_fox",
        24: "Black_bear",
        25: "Vehicle",
        26: "Wolf",
        27: "Empty",
        28: "Other_mustelids",
        29: "Gray_jay",
        30: "Donkey",
        31: "Black-tailed_jackrabbit",
        32: "Snowshoe_hare",
        33: "Marmota_genus",
        34: "Porcupine",
        35: "Grey_squirrel",
        36: "Red_squirrel",
        37: "Red_fox",
        38: "Accipitridae_family",
        39: "Anatidae_family",
        40: "Bighorn_sheep",
        41: "Black-billed_magpie",
        42: "Black-tailed_prairie_dog",
        43: "Canada_lynx",
        44: "Clarks_nutcracker",
        45: "Common_raven",
        46: "Domestic_sheep",
        47: "Golden-mantled_ground_squirrel",
        48: "Grizzly_bear",
        49: "Grouse",
        50: "Gunnisons_prairie_dog",
        51: "Pacific_fisher",
        52: "Passeriformes",
        53: "Prairie_chicken",
        54: "Pronghorn",
        55: "River_otter",
        56: "Sellers_jay",
        57: "Swift_fox",
        58: "Wolverine",
    }

    # Fill dictionary with top classification result
    with open("predictions.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            imagedata[row[1].strip()[17:]] = (
                MLWIC2_names[int(row[2].strip())],
                row[7].strip(),
            )

    print(metadata)
    print(imagedata)

    # Appending model confidence to animal inference
    for name in metadata["Name"]:
        animal, conf = imagedata[name]
        metadata["Animal"].append(animal)
        metadata["Confidence"].append(conf)

    print(metadata)

    # Length checking the dictionary
    print(len(metadata["Name"]),len(metadata["Date"]),len(metadata["Time"]),len(metadata["Temp(C)"]),len(metadata["Temp(F)"]),len(metadata["Animal"]),len(metadata["Confidence"]))

    # Creating a pandas dataframe with the dictionary
    df = pd.DataFrame(metadata)
    # Casting dictionary to a csv, for downloading
    df.to_csv("results.csv", index=False)

    # Return tuples for gradio
    final_tuples = []

    # Creating an output for the gradio interface
    for key in imagedata:
        print(key)
        final_tuples.append(
            [
                original + key.replace("'", ""),
                key + " | " + imagedata[key][0],
            ]
        )

    # Returning each output necessary for gradio
    return {
        out0: gr.update(value=final_tuples, visible=True),
        out1: gr.update(value=df, visible=True),
        out2: "results.csv",
        b1: gr.update(visible=False),
        }

def visibleButton():
    return gr.update(visible=True)

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as display:
    gr.Markdown(
        """
    <h1 align="center"> Animal Classification </h1>
    Input your images below to see the output.
    """
    )
    with gr.Accordion("JPG Images") as a1:
        inp = gr.File(file_count="single", file_types=[".zip"])

    b1 = gr.Button("Classify", visible=False, variant="primary")

    out0 = gr.Gallery(visible=False)

    with gr.Accordion("CSV File", open=True, visible=True) as a2:
        out2 = gr.File()

    out1 = gr.Dataframe(max_rows=3, overflow_row_behaviour="paginate", visible=False)

    # with gr.Accordion("CSV Download"):
    #    out2 = gr.File()
    #    #b2 = gr.Button("Download CSV", elem_id="results")

    inp.upload(visibleButton, outputs=b1)

    b1.click(unzip, inputs=inp, outputs=[out0, out1, out2, b1])

if __name__ == "__main__":
    display.queue().launch(share=True)