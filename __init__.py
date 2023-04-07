bl_info = {
    'name': 'DeepBump',
    'description': 'Generates normal & height maps from image textures',
    'author': 'Hugo Tini',
    'version': (7, 0, 0),
    'blender': (3, 5, 0),
    'location': 'Node Editor > DeepBump',
    'category': 'Material',
    'warning': 'Make sure dependencies are installed in the preferences panel below',
    'doc_url': 'https://github.com/HugoTini/DeepBump/blob/master/readme.md'
}


import bpy
from bpy.types import (Panel, Operator, PropertyGroup)
from bpy.props import (EnumProperty, BoolProperty, PointerProperty)
import os
import subprocess
import sys
import importlib
from collections import namedtuple
import addon_utils


# ------------------------------------------------------------------------
#    Dependencies management utils
# ------------------------------------------------------------------------


def get_dependencies_path():
    # Dependencies to be installed in same folder as addon
    for mod in addon_utils.modules():
        if mod.bl_info['name'] == "DeepBump":
            return os.path.dirname(mod.__file__)
    return None


# Python dependencies management helpers from :
# https://github.com/robertguetzkow/blender-python-examples/tree/master/add_ons/install_dependencies
Dependency = namedtuple('Dependency', ['module', 'package', 'name'])
dependencies = (Dependency(module='onnxruntime', package=None, name='ort'),
                Dependency(module='numpy', package=None, name='np'))
dependencies_installed = False


def import_module(module_name, global_name=None, reload=True):
    sys.path.append(get_dependencies_path())
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
    # Install dependency with pip in the addon folder
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', package_name, '-t', get_dependencies_path()], 
                            text=True, capture_output=True)
    if result.returncode != 0 :
        raise Exception(f'Dependency install issue : {result}') 

    # The installation succeeded, attempt to import the module again
    import_module(module_name, global_name)


# ------------------------------------------------------------------------
#    Scene properties
# ------------------------------------------------------------------------


class DeepBumpProperties(PropertyGroup):

    # Color -> Normals panel props
    colortonormals_tiles_overlap_enum: EnumProperty(
        name='Tiles overlap',
        description='More overlap might help reducing some artifacts but takes longer to compute',
        items=[('SMALL', 'Small', 'Small overlap between tiles'),
               ('MEDIUM', 'Medium', 'Medium overlap between tiles'),
               ('LARGE', 'Large', 'Large overlap between tiles')],
        default='LARGE'
    )

    # Normals -> Height panel props
    normalstoheight_seamless_bool: BoolProperty(
        description='If input normal map is seamless, keep enabled. Toggle off otherwise',
        default=True
    )

    # Normals -> Curvature panel props
    normalstocurvature_blur_radius_enum: EnumProperty(
        name='Curvature blur radius',
        description='Curvature smoothness',
        items=[('SMALLEST', 'Smallest', 'Smallest blur radius'),
               ('SMALLER', 'Smaller', 'Smaller blur radius'),
               ('SMALL', 'Small', 'Small blur radius'),
               ('MEDIUM', 'Medium', 'Medium blur radius'),
               ('LARGE', 'Large', 'Large blur radius'),
               ('LARGER', 'Larger', 'Larger blur radius'),
               ('LARGEST', 'Largest', 'Largest blur radius')],
        default='SMALL'
    )


# ------------------------------------------------------------------------
#    Operators
# ------------------------------------------------------------------------


class DEEPBUMP_OT_ColorToNormalsOperator(Operator):
    bl_idname = 'deepbump.colortonormals'
    bl_label = 'DeepBump Color → Normals'
    bl_description = bl_label
    
    progress_started = False

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
            print(f'DeepBump Color → Normals : {current}/{total}')
        else:
            wm.progress_begin(0, total)
            self.progress_started = True

    def execute(self, context):
         # Get input image from selected node
        input_node = context.active_node
        input_bl_img = input_node.image
        if input_bl_img is None:
            self.report(
                {'WARNING'}, 'Selected image node must have an image assigned to it.')
            return {'CANCELLED'}

        # Convert image to numpy C,H,W array
        input_img = utils.bl_image_to_np(input_bl_img)

        # Compute normals
        OVERLAP = context.scene.deep_bump_tool.colortonormals_tiles_overlap_enum
        self.progress_started = False
        output_img = module_color_to_normals.apply(input_img, OVERLAP, self.progress_print)

        # Create new image datablock
        input_img_name = os.path.splitext(input_bl_img.name)
        output_img_name = input_img_name[0] + '_normals' + input_img_name[1]
        output_bl_img = bpy.data.images.new(
            output_img_name, width=input_bl_img.size[0], height=input_bl_img.size[1])
        output_bl_img.colorspace_settings.name = 'Non-Color'

        # Convert numpy C,H,W array back to blender image pixels
        output_bl_img.pixels = utils.np_to_bl_pixels(output_img)

        # Create new node for normal map
        output_node = context.material.node_tree.nodes.new(
            type='ShaderNodeTexImage')
        output_node.location = input_node.location
        output_node.location[1] -= input_node.width*1.2
        output_node.image = output_bl_img

        # Create normal vector node & link nodes
        normal_vec_node = context.material.node_tree.nodes.new(
            type='ShaderNodeNormalMap')
        normal_vec_node.location = output_node.location
        normal_vec_node.location[0] += output_node.width*1.1
        links = context.material.node_tree.links
        links.new(output_node.outputs['Color'],
                  normal_vec_node.inputs['Color'])

        # If input image was linked to a BSDF, link to BSDF normal slot
        if input_node.outputs['Color'].is_linked:
            if len(input_node.outputs['Color'].links) == 1:
                to_node = input_node.outputs['Color'].links[0].to_node
                if to_node.bl_idname == 'ShaderNodeBsdfPrincipled':
                    links.new(
                        normal_vec_node.outputs['Normal'], to_node.inputs['Normal'])

        print('DeepBump Color → Normals : done')
        return {'FINISHED'}


class DEEPBUMP_OT_NormalsToHeightOperator(Operator):
    bl_idname = 'deepbump.normalstoheight'
    bl_label = 'DeepBump Normals → Height'
    bl_description = bl_label

    progress_started = False

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
            print(f'DeepBump Normals → Height : {current}/{total}')
        else:
            wm.progress_begin(0, total)
            self.progress_started = True

    def execute(self, context):
        # Get input image from selected node
        input_node = context.active_node
        input_bl_img = input_node.image
        if input_bl_img is None:
            self.report(
                {'WARNING'}, 'Selected image node must have an image assigned to it.')
            return {'CANCELLED'}
        if input_bl_img.colorspace_settings.name != 'Non-Color':
            self.report(
                {'WARNING'}, 'Selected image node must be a normal map in Non-Color colorspace.')
            return {'CANCELLED'}

        # Convert image to numpy C,H,W array
        input_img = utils.bl_image_to_np(input_bl_img)

        # Compute height
        print('DeepBump Normals → Height : computing')
        SEAMLESS = context.scene.deep_bump_tool.normalstoheight_seamless_bool
        self.progress_started = False
        output_img = module_normals_to_height.apply(input_img, SEAMLESS, self.progress_print)

        # Create new image datablock
        input_img_name = os.path.splitext(input_bl_img.name)
        output_img_name = input_img_name[0] + '_height' + input_img_name[1]
        output_bl_img = bpy.data.images.new(
            output_img_name, width=input_bl_img.size[0], height=input_bl_img.size[1])
        output_bl_img.colorspace_settings.name = 'Non-Color'

        # Convert numpy C,H,W array back to blender imaga pixels
        output_bl_img.pixels = utils.np_to_bl_pixels(output_img)

        # Create new node for curvature map
        output_node = context.material.node_tree.nodes.new(
            type='ShaderNodeTexImage')
        output_node.location = input_node.location
        output_node.location[1] -= input_node.width*1.2
        output_node.image = output_bl_img

        print('DeepBump Normals → Height : done')
        return {'FINISHED'}


class DEEPBUMP_OT_NormalsToCurvatureOperator(Operator):
    bl_idname = 'deepbump.normalstocurvature'
    bl_label = 'DeepBump Normals → Curvature'
    bl_description = bl_label
    
    progress_started = False

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
            print(f'DeepBump Normals → Curvature : {current}/{total}')
        else:
            wm.progress_begin(0, total)
            self.progress_started = True

    def execute(self, context):
        # Get input image from selected node
        input_node = context.active_node
        input_bl_img = input_node.image
        if input_bl_img is None:
            self.report(
                {'WARNING'}, 'Selected image node must have an image assigned to it.')
            return {'CANCELLED'}
        if input_bl_img.colorspace_settings.name != 'Non-Color':
            self.report(
                {'WARNING'}, 'Selected image node must be a normal map in Non-Color colorspace.')
            return {'CANCELLED'}

        # Convert image to numpy C,H,W array
        input_img = utils.bl_image_to_np(input_bl_img)

        # Compute curvature
        print('DeepBump Normals → Curvature : computing')
        BLUR_RADIUS = context.scene.deep_bump_tool.normalstocurvature_blur_radius_enum
        self.progress_started = False
        output_img = module_normals_to_curvature.apply(input_img, BLUR_RADIUS, self.progress_print)

        # Create new image datablock
        input_img_name = os.path.splitext(input_bl_img.name)
        output_img_name = input_img_name[0] + '_curvature' + input_img_name[1]
        output_bl_img = bpy.data.images.new(
            output_img_name, width=input_bl_img.size[0], height=input_bl_img.size[1])
        output_bl_img.colorspace_settings.name = 'Non-Color'

        # Convert numpy C,H,W array back to blender imaga pixels
        output_bl_img.pixels = utils.np_to_bl_pixels(output_img)

        # Create new node for curvature map
        output_node = context.material.node_tree.nodes.new(
            type='ShaderNodeTexImage')
        output_node.location = input_node.location
        output_node.location[1] -= input_node.width*1.2
        output_node.image = output_bl_img

        print('DeepBump Normals → Curvature : done')
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
        except BaseException as err:
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


class DEEPBUMP_PT_ColorToNormalsPanel(Panel):
    bl_idname = 'DEEPBUMP_PT_ColorToNormalsPanel'
    bl_label = 'Color → Normals'
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
        row = layout.row()
        row.label(text='Tiles overlap')
        row.prop(deep_bump_tool, 'colortonormals_tiles_overlap_enum', text='')
        layout.operator('deepbump.colortonormals', text='Generate Normal Map')


class DEEPBUMP_PT_NormalsToHeightPanel(Panel):
    bl_idname = 'DEEPBUMP_PT_NormalsToHeight'
    bl_label = 'Normals → Height'
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
        layout.prop(deep_bump_tool, 'normalstoheight_seamless_bool', text='Seamless normals')
        layout.operator('deepbump.normalstoheight', text='Generate Height Map')


class DEEPBUMP_PT_NormalsToCurvaturePanel(Panel):
    bl_idname = 'DEEPBUMP_PT_NormalsToCurvature'
    bl_label = 'Normals → Curvature'
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
        row = layout.row()
        row.label(text='Blur radius')
        row.prop(deep_bump_tool, 'normalstocurvature_blur_radius_enum', text='')
        layout.operator('deepbump.normalstocurvature', text='Generate Curvature Map')


class DEEPBUMP_preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        if dependencies_installed :
            layout.label(text='Required dependencies are installed', icon='CHECKMARK')
            layout.label(text=f'(Dependencies path : {get_dependencies_path()})')
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
    # Color -> Normals operator & panel
    DEEPBUMP_OT_ColorToNormalsOperator,
    DEEPBUMP_PT_ColorToNormalsPanel,
    # Normals -> Height operator & panel
    DEEPBUMP_OT_NormalsToHeightOperator,
    DEEPBUMP_PT_NormalsToHeightPanel,
    # Normals -> Curvature operator & panel
    DEEPBUMP_OT_NormalsToCurvatureOperator,
    DEEPBUMP_PT_NormalsToCurvaturePanel,
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
    # Addon specific imports
    from . import module_color_to_normals
    from . import module_normals_to_height
    from . import module_normals_to_curvature
    from . import utils
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
