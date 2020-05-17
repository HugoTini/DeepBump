![DeepBump](banner.jpg)

# DeepBump

DeepBump is a tool to generate normal maps from single 
pictures. See this [blog post](https://hugotini.github.io/deepbump) 
for an introduction. It can be used either as a Blender add-on or 
as a command-line program.

# Blender add-on

## Installation

To install DeepBump as a Blender add-on, some python 
dependencies must be installed first. Go to Blender's 
bundled Python :

    cd <blender-path>/<blender-version>/python/bin/
    
and install the following dependencies :
    
    ./python3.7m -m pip install --upgrade pip
    ./python3.7m -m pip install onnxruntime

(On Windows you might need to replace `./python3.7m` with `python.exe` in the
commands above)
    
By installing `onnxruntime`, be aware of 
[Microsoft conditions](https://github.com/microsoft/onnxruntime/blob/master/docs/Privacy.md).
    
Once dependencies are met,
[download DeepBump as a ZIP](https://github.com/HugoTini/DeepBump/archive/master.zip). 
Then in Blender go to _Edit -> Preferences -> Add-ons -> Install_ 
and select the downloaded ZIP file. Then enable the add-on.

## Usage

In the Shader Editor, select an _Image Texture Node_ and 
click _Generate Normal Map_ in the right panel under 
the _DeepBump_ tab (as illustrated on the
[blog post](https://hugotini.github.io/deepbump) 
first video).

# Command-line

## Installation

To use DeepBump from the terminal, you must have Python 
installed as well as the following dependencies :

    pip install --upgrade pip
    pip install onnxruntime
    pip install Pillow

By installing `onnxruntime`, be aware 
of [Microsoft conditions](https://github.com/microsoft/onnxruntime/blob/master/docs/Privacy.md).

Then either  download DeepBump 
[as a ZIP](https://github.com/HugoTini/DeepBump/archive/master.zip) and extract it somewhere, 
or just clone with :

    git clone https://github.com/HugoTini/DeepBump.git

## Usage

Go to DeepBump folder, then use with :

    python deepbump_cli.py -i <path_input_image> -n <path_generated_normal> -o <overlap>

where `<overlap>` can be `small`, `medium` or `big`.

# License

This repo is under the [GPL license](LICENSE). The training code is currently not available.
