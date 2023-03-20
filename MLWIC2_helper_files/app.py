# app.py
# Chet Russell
# Last edited: Mar 20, 2023

import gradio as gr
import os
import csv
import glob
import tempfile
import shutil
import run
import tensorflow as tf
import utils
from argparse import Namespace

parser = run.gen_argparser()


def classify(images):
    args = parser.parse_args(["inference"])
    dict_args = vars(args)
    # Create temporary folder and put images in it.
    # with tempfile.TemporaryDirectory() as tmpdirname:
    dirname = "./images/"

    # Clean images folder
    old_images = glob.glob(dirname + "*")
    for f in old_images:
        os.remove(f)

    image_list = []

    with open("images.txt", "w") as f:
        for image in images:
            shutil.move(image.name, dirname)
            f.write(image.name.strip("/tmp/") + "\n")
            image_list.append(dirname + image.name.strip("/tmp/"))

    dict_args["path_prefix"] = dirname
    dict_args["log_dir"] = "./species_model"
    dict_args["snapshot_prefix"] = "./species_model"
    dict_args["depth"] = 18
    dict_args["val_info"] = "./images.txt"

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
        )
    )

    run.do_evaluate(sess, namespace_args)

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

    # Fill dictionary with top 3 classification results
    with open("predictions.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            # print([names[int(row[2].strip())], row[7].strip()])
            imagedata[row[1][2:]] = [
                [names[int(row[2].strip())], row[7].strip()],
                [names[int(row[3].strip())], row[8].strip()],
                [names[int(row[4].strip())], row[9].strip()],
            ]

    print(imagedata)

    # TODO: return a tuple for each image description

    return image_list


title = "Animal Classification"
preview = gr.Interface(
    fn=classify,
    title=title,
    inputs=gr.inputs.File(file_count="multiple"),
    outputs="gallery",
    allow_flagging="never",
)

if __name__ == "__main__":
    preview.launch()
