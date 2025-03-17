<img src="img/banner.jpg" style="border-radius:8px">

<br>

# DeepBump

DeepBump is a machine-learning driven tool to generate normal & height maps from single pictures. See this [blog post](https://hugotini.github.io/deepbump) for an introduction.

# Installation

1) [Download DeepBump as a ZIP](https://github.com/HugoTini/DeepBump/releases).

2) In Blender, go to **_Edit -> Preferences -> Add-ons -> Top right arrow -> Install from disk_** and select the downloaded ZIP file. Then enable the add-on.

3) In the add-on preference, click the **'_Install dependencies_'** button (this requires an internet connection and might take a while). In case of error, try running Blender as administrator for this step.

By installing those dependencies, be aware of [Microsoft conditions](https://github.com/microsoft/onnxruntime/blob/master/docs/Privacy.md). This add-on uses available APIs to disable telemetry.

# Usage

Check the [blog post](https://hugotini.github.io/deepbump) first video.

In the **_Shader Editor_**, in the side bar under the **_DeepBump tab_**:

<img src="img/location.jpg" width=512px style="border-radius:8px">

<br>

# Tools

## Color (albedo) → Normals

Select a color image node in the shader editor, then click _Generate Normal Map_.

<img src="img/color_to_normals.jpg" width=512px style="border-radius:8px">

## Normals → Height (displacement)

Select the generated normal map image node in the shader editor, then click _Generate Height Map_.

<img src="img/normals_to_height.jpg" width=512px style="border-radius:8px">

## Normals → Curvature

Select the generated normal map image node in the shader editor, then click _Generate Curvature Map_.

<img src="img/normals_to_curvature.jpg" width=512px style="border-radius:8px">

## Low res -> High res (upscale)

Select a color image node in the shader editor, then click _Upscale_.

<img src="img/lowres_to_highres.jpg" width=512px style="border-radius:8px">

<br>

# Command line

For command-line usage, see [cli.md](cli.md).

# License

This repo is under the [GPL license](LICENSE). The training code is currently not available.
