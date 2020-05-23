import bpy
from bpy.types import (Panel, Operator, PropertyGroup)
from bpy.props import (EnumProperty, PointerProperty)
import pathlib
import os

from . import infer
if "bpy" in locals():
    import importlib
    importlib.reload(infer)

bl_info = {
    'name': 'DeepBump',
    'description': 'Generates normal maps from image textures',
    'author': 'Hugo Tini',
    'version': (0, 1, 0),
    'blender': (2, 82, 0),
    'location': 'Node Editor > DeepBump',
    'category': 'Material',
    'warning': 'Extra Python dependencies must be installed first, check the readme.'
}

# ------------------------------------------------------------------------
#    Scene Properties
# ------------------------------------------------------------------------


class DeepBumpProperties(PropertyGroup):

    tiles_overlap_enum: EnumProperty(
        name='Tiles overlap',
        description='More overlap might help reducing some artifacts but takes longer to compute.',
        items=[('SMALL', 'Small', 'Small overlap between tiles.'),
               ('MEDIUM', 'Medium', 'Medium overlap between tiles.'),
               ('BIG', 'Big', 'Big overlap between tiles.')]
    )

# ------------------------------------------------------------------------
#    Operators
# ------------------------------------------------------------------------


class WM_OT_DeepBumpOperator(Operator):
    '''Generates a normal map from an image. Settings in DeepBump panel (Node Editor)'''

    bl_label = 'DeepBump'
    bl_idname = 'wm.deep_bump'
    progress_started = False
    ort_session = None

    @classmethod
    def poll(self, context):
        selected_node_type = context.active_node.bl_idname
        return (context.area.type == 'NODE_EDITOR') and (selected_node_type == 'ShaderNodeTexImage')

    def progress_print(self, current, total):
        wm = bpy.context.window_manager
        if self.progress_started:
            wm.progress_update(current)
            print('{}/{}'.format(current, total))
        else:
            wm.progress_begin(0, total)
            self.progress_started = True

    def execute(self, context):
        # make sure dependencies are installed
        try:
            import numpy as np
            import onnxruntime as ort
            from . import infer
        except ImportError as e:
            self.report({'WARNING'}, 'Dependencies missing, check readme.')
            print(e)
            return {'CANCELLED'}

        # get input image from selected node
        input_node = context.active_node
        input_img = input_node.image
        if input_img == None:
            self.report(
                {'WARNING'}, 'Selected image node must have an image assigned to it.')
            return {'CANCELLED'}

        # convert to C,H,W numpy array
        width = input_img.size[0]
        height = input_img.size[1]
        channels = input_img.channels
        img = np.array(input_img.pixels)
        img = np.reshape(img, (channels, width, height), order='F')
        img = np.transpose(img, (0, 2, 1))
        # flip height
        img = np.flip(img, axis=1)

        # remove alpha & convert to grayscale
        img = np.mean(img[0:3], axis=0, keepdims=True)

        # split image in tiles
        tile_size = (256, 256)
        OVERLAP = context.scene.deep_bump_tool.tiles_overlap_enum
        overlaps = {'SMALL': 20, 'MEDIUM': 50, 'BIG': 124}
        stride_size = (tile_size[0]-overlaps[OVERLAP],
                       tile_size[1]-overlaps[OVERLAP])
        print('tilling')
        tiles, paddings = infer.tiles_split(img, tile_size, stride_size)

        # load model (if not already loaded)
        if self.ort_session == None:
            print('loading model')
            addon_path = str(pathlib.Path(__file__).parent.absolute())
            self.ort_session = ort.InferenceSession(
                addon_path+'/deepbump256.onnx')

        # predict normal map for each tile
        print('generating')
        self.progress_started = False
        pred_tiles = infer.tiles_infer(
            tiles, self.ort_session, progress_callback=self.progress_print)

        # merge tiles
        print('merging')
        pred_img = infer.tiles_merge(
            pred_tiles, stride_size, (3, img.shape[1], img.shape[2]), paddings)

        # normalize each pixel to unit vector
        pred_img = infer.normalize(pred_img)

        # create new image datablock
        img_name = os.path.splitext(input_img.name)
        normal_name = img_name[0] + '_normal' + img_name[1]
        normal_img = bpy.data.images.new(
            normal_name, width=width, height=height)
        normal_img.colorspace_settings.name = 'Non-Color'

        # flip height
        pred_img = np.flip(pred_img, axis=1)
        # add alpha channel
        pred_img = np.concatenate(
            [pred_img, np.ones((1, height, width))], axis=0)
        # flatten to array
        pred_img = np.transpose(pred_img, (0, 2, 1)).flatten('F')
        # write to image block
        normal_img.pixels = pred_img

        # create new node for normal map
        normal_node = context.material.node_tree.nodes.new(
            type='ShaderNodeTexImage')
        normal_node.location = input_node.location
        normal_node.location[1] -= input_node.width*1.2
        normal_node.image = normal_img

        # create normal vector node & link nodes
        normal_vec_node = context.material.node_tree.nodes.new(
            type='ShaderNodeNormalMap')
        normal_vec_node.location = normal_node.location
        normal_vec_node.location[0] += normal_node.width*1.1
        links = context.material.node_tree.links
        links.new(normal_node.outputs['Color'],
                  normal_vec_node.inputs['Color'])

        # if input image was linked to a BSDF, link to BSDF normal slot
        if input_node.outputs['Color'].is_linked:
            if len(input_node.outputs['Color'].links) == 1:
                to_node = input_node.outputs['Color'].links[0].to_node
                if to_node.bl_idname == 'ShaderNodeBsdfPrincipled':
                    links.new(
                        normal_vec_node.outputs['Normal'], to_node.inputs['Normal'])

        return {'FINISHED'}

# ------------------------------------------------------------------------
#    Panel in Object Mode
# ------------------------------------------------------------------------


class OBJECT_PT_DeepBumpPanel(Panel):
    bl_label = 'DeepBump'
    bl_idname = 'OBJECT_PT_DeepBumpPanel'
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
        layout.operator('wm.deep_bump', text='Generate Normal Map')

# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------


classes = (
    DeepBumpProperties,
    WM_OT_DeepBumpOperator,
    OBJECT_PT_DeepBumpPanel
)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    bpy.types.Scene.deep_bump_tool = PointerProperty(type=DeepBumpProperties)


def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
    del bpy.types.Scene.deep_bump_tool


if __name__ == '__main__':
    register()
