![DeepBump](banner.jpg)

# DeepBump

DeepBump is a tool to generate normal maps from single 
pictures. See this [blog post](https://hugotini.github.io/deepbump) 
for an introduction. It can be used either as a Blender add-on or 
as a command-line program.

# Installation

1) [Download DeepBump as a ZIP](https://github.com/HugoTini/DeepBump/releases).

2) In Blender, go to _Edit -> Preferences -> Add-ons -> Install_ 
and select the downloaded ZIP file. Then enable the add-on.

3) In the add-on preference, click the '_Install dependencies_' button (this 
requires an internet connection and might take a while).

By installing those dependencies, be aware of 
[Microsoft conditions](https://github.com/microsoft/onnxruntime/blob/master/docs/Privacy.md). 
This add-on use available APIs to disable telemetry.


# Usage

Check the [blog post](https://hugotini.github.io/deepbump) first video.

In the Shader Editor, select an _Image Texture Node_ and 
click _Generate Normal Map_ in the right panel under 
the _DeepBump_ tab.

(For advanced usage, see [cli.md](cli.md) for the command-line interface.)

# License

This repo is under the [GPL license](LICENSE). The training code is currently not available.
