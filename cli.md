# DeepBump CLI

# Installation

1) [Download DeepBump as a ZIP](https://github.com/HugoTini/DeepBump/releases) and extract it.

2) Install the required dependencies :

        pip install numpy onnxruntime imageio

# Usage

See `python3 cli.py -h` for a list of possible arguments. Some examples :

**Color (albedo) → Normals** : 

        python3 cli.py color.jpg normals.jpg color_to_normals

        python3 cli.py color.png normals.png color_to_normals --color_to_normals-overlap MEDIUM

**Normals → Height (displacement)** :

        python3 cli.py normals.png height.png normals_to_height

        python3 cli.py normals.png height.png normals_to_height --normals_to_height-seamless TRUE

**Normals → Curvature** :

        python3 cli.py normals.png curvature.png normals_to_curvature

        python3 cli.py normals.png curvature.png normals_to_curvature --normals_to_curvature-blur_radius SMALLEST