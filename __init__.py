bl_info = {
    'name': 'DeepBump',
    'description': 'Generates normal maps from image textures',
    'author': 'Hugo Tini',
    'version': (3, 0, 0),
    'blender': (3, 0, 0),
    'location': 'Node Editor > DeepBump',
    'category': 'Material',
    'warning': 'Requires installation of dependencies',
    'doc_url': 'https://github.com/HugoTini/DeepBump/blob/master/readme.md'
}


import bpy
from bpy.types import (Panel, Operator, PropertyGroup)
from bpy.props import (EnumProperty, PointerProperty)
import pathlib
import os
import subprocess
import sys
import importlib
from collections import namedtuple


# ------------------------------------------------------------------------
#    Dependencies management utils
# ------------------------------------------------------------------------


# Python dependencies management helpers from :
# https://github.com/robertguetzkow/blender-python-examples/tree/master/add_ons/install_dependencies
Dependency = namedtuple('Dependency', ['module', 'package', 'name'])
dependencies = (Dependency(module='onnxruntime', package=None, name='ort'),
                Dependency(module='numpy', package=None, name='np'))
dependencies_installed = False


def import_module(module_name, global_name=None, reload=True):
    if global_name is None:
        global_name = module_name
    if global_name in globals():
        importlib.reload(globals()[global_name])
    else:
        globals()[global_name] = importlib.import_module(module_name)


def install_pip():
    try:
        # Check if pip is already installed
        subprocess.run([sys.executable, '-m', 'pip', '--version'], check=True)
    except subprocess.CalledProcessError:
        import ensurepip
        ensurepip.bootstrap()
        os.environ.pop("PIP_REQ_TRACKER", None)


def install_and_import_module(module_name, package_name=None, global_name=None):
    if package_name is None:
        package_name = module_name
    if global_name is None:
        global_name = module_name
    # Create a copy of the environment variables and modify them for the subprocess call
    environ_copy = dict(os.environ)
    environ_copy['PYTHONNOUSERSITE'] = '1'
    subprocess.run([sys.executable, '-m', 'pip', 'install', package_name], check=True, env=environ_copy)
    # The installation succeeded, attempt to import the module again
    import_module(module_name, global_name)


# ------------------------------------------------------------------------
#    Scene properties
# ------------------------------------------------------------------------


class DeepBumpProperties(PropertyGroup):

    tiles_overlap_enum: EnumProperty(
        name='Tiles overlap',
        description='More overlap might help reducing some artifacts but takes longer to compute.',
        items=[('SMALL', 'Small', 'Small overlap between tiles.'),
               ('MEDIUM', 'Medium', 'Medium overlap between tiles.'),
               ('LARGE', 'Large', 'Large overlap between tiles.')],
        default='LARGE'
    )


# ------------------------------------------------------------------------
#    Operators
# ------------------------------------------------------------------------


class DEEPBUMP_OT_DeepBumpOperator(Operator):
    bl_idname = 'deepbump.operator'
    bl_label = 'DeepBump'
    bl_description = ('Generates a normal map from an image.' 
                      'Settings are in DeepBump panel (Node Editor)')
    
    progress_started = False
    ort_session = None

    @classmethod
    def poll(self, context):
        if context.active_node is not None :
            selected_node_type = context.active_node.bl_idname
            return (context.area.type == 'NODE_EDITOR') and (selected_node_type == 'ShaderNodeTexImage')
        return False

    def progress_print(self, current, total):
        wm = bpy.context.window_manager
        if self.progress_started:
            wm.progress_update(current)
            print(f'DeepBump : {current}/{total}')
        else:
            wm.progress_begin(0, total)
            self.progress_started = True

    def execute(self, context):
        # Get input image from selected node
        input_node = context.active_node
        input_img = input_node.image
        if input_img is None:
            self.report(
                {'WARNING'}, 'Selected image node must have an image assigned to it.')
            return {'CANCELLED'}

        # Convert to C,H,W numpy array
        width = input_img.size[0]
        height = input_img.size[1]
        channels = input_img.channels
        img = np.array(input_img.pixels)
        img = np.reshape(img, (channels, width, height), order='F')
        img = np.transpose(img, (0, 2, 1))
        # Flip height
        img = np.flip(img, axis=1)

        # Remove alpha & convert to grayscale
        img = np.mean(img[0:3], axis=0, keepdims=True)

        # Split image in tiles
        print('DeepBump : tilling')
        tile_size = 256
        OVERLAP = context.scene.deep_bump_tool.tiles_overlap_enum
        overlaps = {'SMALL': tile_size//6, 'MEDIUM': tile_size//4, 'LARGE': tile_size//2}
        stride_size = tile_size-overlaps[OVERLAP]
        tiles, paddings = infer.tiles_split(img, (tile_size, tile_size),
                                            (stride_size, stride_size))

        # Load model (if not already loaded)
        if self.ort_session is None:
            print('DeepBump : loading model')
            addon_path = str(pathlib.Path(__file__).parent.absolute())
            self.ort_session = ort.InferenceSession(
                addon_path+'/deepbump256.onnx')
            self.ort_session

        # Predict normal map for each tile
        print('DeepBump : generating')
        self.progress_started = False
        pred_tiles = infer.tiles_infer(
            tiles, self.ort_session, progress_callback=self.progress_print)

        # Merge tiles
        print('DeepBump : merging')
        pred_img = infer.tiles_merge(pred_tiles, (stride_size, stride_size), 
                                    (3, img.shape[1], img.shape[2]), paddings)

        # Normalize each pixel to unit vector
        pred_img = infer.normalize(pred_img)

        # Create new image datablock
        img_name = os.path.splitext(input_img.name)
        normal_name = img_name[0] + '_normal' + img_name[1]
        normal_img = bpy.data.images.new(
            normal_name, width=width, height=height)
        normal_img.colorspace_settings.name = 'Non-Color'

        # Flip height
        pred_img = np.flip(pred_img, axis=1)
        # Add alpha channel
        pred_img = np.concatenate(
            [pred_img, np.ones((1, height, width))], axis=0)
        # Flatten to array
        pred_img = np.transpose(pred_img, (0, 2, 1)).flatten('F')
        # Write to image block
        normal_img.pixels = pred_img

        # Create new node for normal map
        normal_node = context.material.node_tree.nodes.new(
            type='ShaderNodeTexImage')
        normal_node.location = input_node.location
        normal_node.location[1] -= input_node.width*1.2
        normal_node.image = normal_img

        # Create normal vector node & link nodes
        normal_vec_node = context.material.node_tree.nodes.new(
            type='ShaderNodeNormalMap')
        normal_vec_node.location = normal_node.location
        normal_vec_node.location[0] += normal_node.width*1.1
        links = context.material.node_tree.links
        links.new(normal_node.outputs['Color'],
                  normal_vec_node.inputs['Color'])

        # If input image was linked to a BSDF, link to BSDF normal slot
        if input_node.outputs['Color'].is_linked:
            if len(input_node.outputs['Color'].links) == 1:
                to_node = input_node.outputs['Color'].links[0].to_node
                if to_node.bl_idname == 'ShaderNodeBsdfPrincipled':
                    links.new(
                        normal_vec_node.outputs['Normal'], to_node.inputs['Normal'])

        print('DeepBump : done')
        return {'FINISHED'}


class DEEPBUMP_OT_install_dependencies(bpy.types.Operator):
    bl_idname = 'deepbump.install_dependencies'
    bl_label = 'Install dependencies'
    bl_description = 'Downloads and installs the required python packages for this add-on.'
    bl_options = {'REGISTER', 'INTERNAL'}

    @classmethod
    def poll(self, context):
        # Deactivate when dependencies have been installed
        return not dependencies_installed

    def execute(self, context):
        try:
            install_pip()
            for dependency in dependencies:
                install_and_import_module(module_name=dependency.module,
                                          package_name=dependency.package,
                                          global_name=dependency.name)
        except (subprocess.CalledProcessError, ImportError) as err:
            self.report({'ERROR'}, str(err))
            return {'CANCELLED'}

        global dependencies_installed
        dependencies_installed = True

        # Register the panels, operators, etc. since dependencies are installed
        register_functionality()

        return {"FINISHED"}


# ------------------------------------------------------------------------
#    UI (DeepBump panel & addon install dependencies button)
# ------------------------------------------------------------------------


class DEEPBUMP_PT_DeepBumpPanel(Panel):
    bl_idname = 'DEEPBUMP_PT_DeepBumpPanel'
    bl_label = 'DeepBump'
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'DeepBump'
    bl_context = 'objectmode'

    @classmethod
    def poll(self, context):
        return context.object is not None

    def draw(self, context):
        layout = self.layout
        deep_bump_tool = context.scene.deep_bump_tool

        layout.label(text='Tiles overlap :')
        layout.prop(deep_bump_tool, 'tiles_overlap_enum', text='')

        layout.separator()
        layout.operator('deepbump.operator', text='Generate Normal Map')


class DEEPBUMP_preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        if dependencies_installed :
            layout.label(text='Required dependencies are installed', icon='CHECKMARK')
        else :
            layout.label(text='Installing dependencies requires internet and might take a few minutes', 
                         icon='INFO')
            layout.operator(DEEPBUMP_OT_install_dependencies.bl_idname, icon='CONSOLE')


# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------


# Classes for the addon actual functionality
classes = (
    DeepBumpProperties,
    DEEPBUMP_OT_DeepBumpOperator,
    DEEPBUMP_PT_DeepBumpPanel
)
# Classes for downloading & installing dependencies
preference_classes = (
    DEEPBUMP_OT_install_dependencies,
    DEEPBUMP_preferences
)


def register():
    global dependencies_installed
    dependencies_installed = False

    for cls in preference_classes:
        bpy.utils.register_class(cls)

    try:
        for dependency in dependencies:
            import_module(module_name=dependency.module, global_name=dependency.name)
    except ModuleNotFoundError:
        # Don't register other panels, operators etc.
        return

    dependencies_installed = True
    register_functionality()


def register_functionality():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.deep_bump_tool = PointerProperty(type=DeepBumpProperties)
    from . import infer
    # Disable MS telemetry
    ort.disable_telemetry_events()


def unregister():
    for cls in preference_classes:
        bpy.utils.unregister_class(cls)

    if dependencies_installed:
        for cls in classes:
            bpy.utils.unregister_class(cls)
            
    del bpy.types.Scene.deep_bump_tool


if __name__ == '__main__':
    register()
