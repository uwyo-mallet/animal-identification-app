# app.py
# Chet Russell
# Last edited: November 9, 2023

import gradio as gr
import os
import csv
import glob
import shutil
import run
import tensorflow as tf
import utils
from argparse import Namespace
import extract
import pandas as pd
import tempfile
from collections import defaultdict
import copy

parser = run.gen_argparser()


def clean(directory):
    old_images = glob.glob(directory + "*")
    for f in old_images:
        os.remove(f)


def download(url):
    # r = requests.get(url, allow_redirects=True)
    # open("results.csv", "wb").write(r.content)
    print(url)


# Main function to classify images
def classify(images, progress=gr.Progress()):

    total = 0

    # images.remove(images[0])
    for image in images:
        total += 1
        print(image.name)

    original = "./original_images/"
    resized = "./resized_images/"
    temperature = "./temp_folder/"

    # Clean images folder
    clean(original)
    clean(resized)
    clean(temperature)

    ims = []
    metadata = defaultdict(list)

    # with open("images.txt", "w") as f:
    with open("images.txt", "w") as f:
        #for image in progress.tqdm(images, desc="Image Preprocessing"):
        for index, image in enumerate(images, start=1):
            progress((index) * (.5/total), desc="Image Preprocessing")
            f.write(image.name.strip("/tmp/")[:-53] + ".jpg")
            f.write("\n")
            shutil.move(image.name, original)
            old_file = os.path.join(original, image.name.strip("/tmp/"))
            new_file = os.path.join(original, image.name.strip("/tmp/")[:-53])
            os.rename(old_file, new_file + ".jpg")
            final_image = new_file.strip(original) + ".jpg"
            ims.append(final_image)
            print(final_image)
            # f.write(new_file.strip(original) + ".jpg\n")
            # extract.crop(new_file, resized)
            extract.crop(new_file + ".jpg", final_image, resized)
            extract.im_meta_data(new_file + ".jpg", final_image, temperature)
            extract.meta_dict(
                new_file + ".jpg", final_image, original, temperature, metadata
            )

    print(ims)


    # Running into error where there is a key difference in the dictionaries? Not entirely sure what was happening.
    # Something with the model is causing this error, for the life of me I cannot figure it out. This fix is a stopgap.
    # This loop essentially restarts the model if it errors, and because it errors very rarely, it theoretically wont
    # be an issue.

    # Maximum number of restarts
    #max_restarts = 5
    #while max_restarts != 0:

    # for img in progress.tqdm(ims, desc="Classifying Images"):
    tf.reset_default_graph()
    args = parser.parse_args(["inference"])
    dict_args = vars(args)
    # Create temporary folder and put resized image in it.
    # with tempfile.TemporaryDirectory() as tmpdirname:

    #    for img in ims:
    #        shutil.copyfile(resized + img, tmpdirname + '/' + img)

    # with open("images.txt", "w") as f:
    #    f.write(img)

    # Starting the model

    dict_args["path_prefix"] = resized
    dict_args["log_dir"] = "./species_model"
    dict_args["snapshot_prefix"] = "./species_model"
    dict_args["depth"] = 18
    dict_args["val_info"] = "./images.txt"
    dict_args["batch_size"] = 1

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

    sess = tf.compat.v1.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=namespace_args.log_device_placement,
        )
    )

    run.do_evaluate(sess, namespace_args, progress, total)
    # Done with model

    # Starting post-processing

    imagedata = {}

    names = {
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
                names[int(row[2].strip())],
                row[7].strip(),
            )

    # Throw predictions in results csv file

    # This ensures that the new dictionary is a complete copy of the old one.
    # This makes it certain that if it needs to restart, the copy will be deleted, rather than the original.
    #test_metadata = copy.deepcopy(metadata)

    print(metadata)
    print(imagedata)

    for name in metadata["Name"]:
        animal, conf = imagedata[name]
        metadata["Animal"].append(animal)
        metadata["Confidence"].append(conf)

    #try:
    #    for name in test_metadata["Name"]:
    #        animal, conf = imagedata[name]
    #        test_metadata["Animal"].append(animal)
    #        test_metadata["Confidence"].append(conf)
    #    # If it gets to this point, it did not error, so break the restart loop
    #    break
    #except:
    #    # Key error with the dictionary return
    #    print("A Key Error exception occured, restarting")
    #    # Delete the copy of the dictionary and start again
    #    del test_metadata
    #    # decrement the max_restarts
    #    max_restarts -= 1

    print(metadata)

    print(len(metadata["Name"]),len(metadata["Date"]),len(metadata["Time"]),len(metadata["Temp(C)"]),len(metadata["Temp(F)"]),len(metadata["Animal"]),len(metadata["Confidence"]))

    df = pd.DataFrame(metadata)
    df.to_csv("results.csv", index=False)

    # return tuples for gradio
    final_tuples = []

    for key in imagedata:
        print(key)
        final_tuples.append(
            [
                "./original_images/" + key.replace("'", ""),
                key + " | " + imagedata[key][0],
            ]
        )

    return final_tuples, df, "results.csv"


with gr.Blocks() as display:
    gr.Markdown(
        """
    <h1 align="center"> Animal Classification </h1>
    Input your images below to see the output.
    """
    )
    with gr.Accordion("JPG Images"):
        inp = gr.File(file_count="multiple", file_types=["image", ".jpg", ".jpeg"])
    
    b1 = gr.Button("Classify")

    with gr.Accordion("Image Preview", open=False):
        out0 = gr.Gallery()
    with gr.Accordion("CSV Preview", open=False):
        out1 = gr.Dataframe(max_rows=3, overflow_row_behaviour="paginate")

    with gr.Accordion("CSV File", open=True):
        out2 = gr.File(open=False)
                
    # with gr.Accordion("CSV Download"):
    #    out2 = gr.File()
    #    #b2 = gr.Button("Download CSV", elem_id="results")

    

    b1.click(classify, inputs=inp, outputs=[out0, out1, out2])
    # b2.click(download, inputs=out2)

if __name__ == "__main__":
    display.queue().launch(share=True)
