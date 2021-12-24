# Command line interface

## Installation

To use DeepBump from the terminal, you must have Python 
installed as well as the following dependencies :

    pip install onnxruntime
    pip install Pillow

By installing `onnxruntime`, be aware 
of [Microsoft conditions](https://github.com/microsoft/onnxruntime/blob/master/docs/Privacy.md). This add-on use available APIs to disable telemetry.

Then either [download DeepBump as a ZIP](https://github.com/HugoTini/DeepBump/releases) and extract it somewhere, or clone it with :

    git clone https://github.com/HugoTini/DeepBump.git

## Usage

Go to the DeepBump folder, then use with :

    python deepbump_cli.py -i <path_input_image> -n <path_generated_normal> -o <overlap>

where `<overlap>` can be `small`, `medium` or `large`.