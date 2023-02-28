# app.py
# Chet Russell
# Last edited: Feb 27

import gradio as gr
import tempfile
import shutil
import run
import tensorflow as tf
import utils
from argparse import Namespace

parser =  run.gen_argparser()

def classify(images):
    args = parser.parse_args(["inference"])
    dict_args = vars(args)
    # Create temporary folder and put temporary images in it.
    with tempfile.TemporaryDirectory() as tmpdirname:
        for image in images:
            shutil.move(image.name, tmpdirname)

        dict_args['path_prefix'] = tmpdirname
        dict_args['log_dir'] = '/home/chet/Documents/MLWIC_Python3/Animal_identification_app/MLWIC2_helper_files/species_model'
        dict_args['snapshot_prefix'] = '/home/chet/Documents/MLWIC_Python3/Animal_identification_app/MLWIC2_helper_files/species_model'
        dict_args['depth'] = 18
        dict_args['val_info'] = 'images.txt'

        namespace_args = Namespace(**dict_args)
        namespace_args.num_val_samples, namespace_args.num_val_batches = utils.count_input_records(args.val_info, args.batch_size)
        namespace_args.inference_only = True

        # Logging the runtime information if requested
        if args.log_debug_info:
          namespace_args.run_options = tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE)
          namespace_args.run_metadata = tf.RunMetadata()
        else:
          namespace_args.run_options = None
          namespace_args.run_metadata = None


        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement= True, 
            log_device_placement= namespace_args.log_device_placement))

        run.do_evaluate(sess, namespace_args)
    
    
title = "Animal Classification"
preview = gr.Interface(
    fn=classify,
    title=title,
    inputs=gr.inputs.File(file_count="multiple"),
    outputs="image",
)

if __name__ == "__main__":
    preview.launch()
