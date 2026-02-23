"""
Steps:
1. add obj coord to “” run cell get pos_obj.json
2. get pos_cam.json from get_cam_pos.py
3. run cell in unreal_to_mitsuba.ipynb to get all xml in sequences
4. Change texture to white -> using reset_texture.py
5. Get images for masks using render_and_save_imgs.py
6. Change texture_back using reset_te0xture.pye

"""

import unreal
import json
import hydra
import xml.etree.ElementTree as ET
import mitsuba as mi
from glob import glob
import cv2
import os

mi.set_variant('cuda_ad_rgb')

def get_pos_obj(obj_pos, num_frames,  obj_pos_json_name):


    obj_data = []
    for i in range(num_frames):
        obj_data.append(
            
        {
            "Location.X": obj_pos["Location.X"],
            "Location.Y": obj_pos["Location.Y"],
            "Location.Z": obj_pos["Location.Z"],
            "Rotation.X": obj_pos["Rotation.X"],
            "Rotation.Y": obj_pos["Rotation.Y"],
            "Rotation.Z": obj_pos["Rotation.Z"],
            "Scale.X": obj_pos["Scale.X"],
            "Scale.Y": obj_pos["Scale.Y"],
            "Scale.Z": obj_pos["Scale.Z"],
            
        })

    with open( obj_pos_json_name, "w") as file:
        json.dump(obj_data, file, indent=4)


def get_camera_positions(cam_pos_json_name, seq_label):

    EAS = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = EAS.get_all_level_actors()
    print(actors)

    sequence_actor = unreal.EditorFilterLibrary.by_actor_label(actors, seq_label)[0]

    level_sequence = sequence_actor.get_sequence()
    object_bindings = level_sequence.get_bindings()

    seq_start = level_sequence.get_playback_start()
    seq_end = level_sequence.get_playback_end()
    print("playback start : ", seq_start)
    print("playback end : ", seq_end)

    seq_length = seq_end - seq_start
    print("Sequence length : ", seq_length)

    print("Object bindings")
    print(object_bindings)
    transform_track = None

    cam_keys_info = {}

    for binding in object_bindings:
        if binding.get_display_name() == "CineCameraActor":
            print("Found camera binding")
            for track in binding.get_tracks():
                    print("track info")
                    if isinstance(track, unreal.MovieScene3DTransformTrack):
                        sections = track.get_sections()
                        channels = sections[0].get_all_channels()
                        #  print(channels)
                        for channel in channels:
                            # print(channel.get_name())
                            keys = channel.get_keys()
                            print(len(keys))
                            channel_keys = []
                            for k in keys:
                                # print(k.get_interpolation_mode())
                                # print(k.get_time().frame_number.value)
                                # print(k.get_value())
                                
                                channel_keys.append({"interpolation" : k.get_interpolation_mode(),
                                                        "frame_number":k.get_time().frame_number.value,
                                                        "value":  k.get_value()})
                                # print("----------------")
                            print(channel_keys)
                            cam_keys_info[channel.get_name().split("_")[0]] = channel_keys
            if transform_track:
                pass
            
    cam_positions = [{} for i in range(seq_length)] 


    print(cam_keys_info)

    for metric in cam_keys_info:
        print("Processing metric: ", metric)
        keyframes = [x["frame_number"] for x in cam_keys_info[metric]] 
        keyvalues = [x["value"] for x in cam_keys_info[metric]]

        print(keyframes)
        print(keyvalues)

        k = 0
        for i in range(seq_length): 
            cam_positions[i][metric] = (i/keyframes[-1]) * (keyvalues[-1]-keyvalues[0]) + keyvalues[0]

    with open( cam_pos_json_name, "w") as file:
        json.dump(cam_positions, file, indent=4)

def parse_json(json_path):
    transform_data = json.load(open(json_path,"r" ,encoding="utf-8"))
    return transform_data

def convert_values(cam_dict, obj_dict, mi_unreal_ratio):
    new_camx = (cam_dict["Location.X"] - obj_dict["Location.X"])/(mi_unreal_ratio * obj_dict["Scale.X"])
    new_camy = -(cam_dict["Location.Y"] - obj_dict["Location.Y"])/(mi_unreal_ratio * obj_dict["Scale.Y"])
    new_camz = (cam_dict["Location.Z"] - obj_dict["Location.Z"])/(mi_unreal_ratio * obj_dict["Scale.Z"])

    if cam_dict["Rotation.Y"] != 90 and cam_dict != -90:

        cam_rotx = cam_dict["Rotation.X"]
        cam_roty = -cam_dict["Rotation.Y"]
        cam_rotz = -cam_dict["Rotation.Z"]
    
    else:
        cam_rotx = 0
        cam_roty = -cam_dict["Rotation.Y"]
        cam_rotz = -(cam_dict["Rotation.Z"] - cam_dict["Rotation.X"])


    obj_rotx = obj_dict["Rotation.X"]
    obj_roty = -obj_dict["Rotation.Y"]
    obj_rotz = -obj_dict["Rotation.Z"]

    cam_rot = [cam_rotx, cam_roty, cam_rotz]
    obj_rot = [obj_rotx, obj_roty, obj_rotz]
    t = [new_camx, new_camy, new_camz]

    return cam_rot, obj_rot, t


def write_xml( cam_rot, obj_rot, t , frame_num, parent_dir, orig_scene_path):
    rot_camx, rot_camy, rot_camz = cam_rot
    rot_objx, rot_objy, rot_objz = obj_rot
    t_x, t_y, t_z = t
    ## rotate to align with unreal orientation
    t1 = mi.Transform4f().rotate(axis=[0,1,0], angle=90)
    t12 = mi.Transform4f().rotate(axis=[1,0,0], angle=90)

    #now x and z are aligned, and y is -y

    # rotate roll , then pitch, then yaw
    # roll
    t2 = mi.Transform4f().rotate(axis=[1,0,0], angle=rot_camx)
    #pitch
    t3 = mi.Transform4f().rotate(axis=[0,1,0], angle=rot_camy)
    # yaw
    t4 = mi.Transform4f().rotate(axis=[0,0,1], angle=rot_camz)

    # translation
    t5 = mi.Transform4f().translate([t_x,t_y,t_z])

    T = t5@t4@t3@t2@t12@t1
    print(T)

    o11 = mi.Transform4f().rotate(axis=[1,0,0], angle=90)
    o12 = mi.Transform4f().rotate(axis=[0,0,1], angle=90)
    o21 =  mi.Transform4f().rotate(axis=[0,0,1], angle=rot_objx)
    o22 =  mi.Transform4f().rotate(axis=[0,0,1], angle=rot_objy)
    o23 = mi.Transform4f().rotate(axis=[0,0,1], angle=rot_objz)
    o3 = mi.Transform4f().translate([0,0,0])
    # o2 = mi.Transform4f().scale([5, 5, 5])
    objT = o3@o23@o22@o21@o12@o11

    print("obj")
    print(objT)

    tree = ET.parse(orig_scene_path)
    root = tree.getroot()
    mat_str = ""
    for row in T.matrix:
        for col in row:
            mat_str = mat_str + " "+ str(round(col[0],4))
    mat_str = mat_str[1:]
    sensor = root.find('sensor')
    print(sensor)
    to_world = sensor.find('transform')
    matrix = to_world.find('matrix')
    matrix.set("value", mat_str)


    mat_str = ""
    for row in objT.matrix:
        for col in row:
            mat_str = mat_str + " "+ str(round(col[0],4))
    mat_str = mat_str[1:]
    sensor = root.findall('''.//*[@id='plane']''')[0]
    to_world = sensor.find('transform')
    matrix = to_world.find('matrix')

    matrix.set("value", mat_str)


    tree.write(f'{parent_dir}\\frame_{frame_num}.xml')

def get_seq_xml(seq_xml_dir, cam_transforms, obj_transforms, mi_unreal_ratio, orig_scene_path):
    for i in range(len(cam_transforms)):
        cam_rot, obj_rot, t = convert_values(cam_transforms[i], obj_transforms[i], mi_unreal_ratio)
        write_xml(cam_rot, obj_rot, t , i, seq_xml_dir, orig_scene_path)

def change_texture(xml_dir, texture_path):
    mi_files = sorted(glob(f"{xml_dir}\\*.xml"))
    for file in mi_files:
        print(file)
        tree = ET.parse(file)
        root = tree.getroot()

        obj = root.findall('''.//*[@id="plane"]''')[0]

        # obj_path = obj.find('.//string[@name="filename"]')
        # obj_path.set("value", obj_file)

        mat = obj.findall("bsdf")[0].findall("texture")[0]
        tex = mat.find('.//string[@name="filename"]')
        tex.set("value", texture_path)

        # light = root.findall('''.//*[@id="light"]''')[0]
        # # print(light)
        # transform = light.find("transform").find("matrix")
        # transform.set("value", light_translation)

        tree.write(file)

def get_mi_imgs(seq_xml_dir, dest_folder):
    mi_files = sorted(glob(f"{seq_xml_dir}\\*.xml"))
    for file in mi_files:
        scene_img = mi.Bitmap(mi.render( mi.load_file(file), spp=512))
        scene_img = scene_img.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True)

        xml_name = file.split(".xml")[0].split("\\")[-1]

        scene_img.write(dest_folder+"//"+xml_name+".png")

def binarize_img(img_path):
    mi_files = sorted(glob(img_path+"//*")) 
    for file in mi_files[:2]:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
        cv2.imwrite( file, binary)

@hydra.main(version_base=None, config_path="config", config_name="config")
def initialize(cfg):
    unreal.log("Starting initialization script...")

    # store object positions
    
    ## Load the object position from the config
    obj_pos = cfg.object_position
    obj_pos_json_name = cfg.object_position_json_path
    ## Get the object position and save it to a JSON file
    get_pos_obj(obj_pos, cfg.num_frames, obj_pos_json_name)

    # store camera positions

    cam_json = cfg.camera_json_path
    seq_label = cfg.sequence_label

    # Get camera positions
    get_camera_positions(cam_json, seq_label)

    # generate sequence xml files
    
    seq_xml_dir = cfg.sequence_xml_dir
    orig_scene_path = cfg.original_scene_path
    mi_scale = cfg.mitsuba_scale
    unreal_scale = cfg.unreal_scale
    mi_unreal_ratio = unreal_scale/mi_scale

    cam_transforms = parse_json(cam_json)
    obj_transforms = parse_json(obj_pos_json_name)

    os.makedirs(seq_xml_dir, exist_ok=True)

    get_seq_xml(seq_xml_dir, cam_transforms, obj_transforms, mi_unreal_ratio, orig_scene_path)

    # change texture to white
    white_texture_path = cfg.white_texture_path
    change_texture(seq_xml_dir, white_texture_path)

    # get images from mitsuba
    mask_dest_folder = cfg.mask_destination_folder
    os.makedirs(mask_dest_folder, exist_ok=True)
    
    get_mi_imgs(seq_xml_dir, mask_dest_folder)
    binarize_img(mask_dest_folder)

    #changetexture to desired
    desired_texture_path = cfg.desired_texture_path
    change_texture(seq_xml_dir, desired_texture_path)

    print("------------------------------------------------------")
    print("Initialization completed successfully.")
    print("sequence xml files are stored in : ", seq_xml_dir)
    print("Mask images are stored in : ", mask_dest_folder)
    print("------------------------------------------------------")

    unreal.log("Initialization script completed successfully.")

if __name__ == "__main__":
    initialize()