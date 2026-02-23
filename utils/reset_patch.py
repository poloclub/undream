import unreal
texture = '/Game/Fab/bin_eps5_a1.bin_eps5_a1'
material_instance_name = '/Game/Fab/CustomMaterial_Inst.CustomMaterial_Inst'
material_instance = unreal.EditorAssetLibrary.load_asset(material_instance_name)
texture_asset = unreal.load_asset(texture)
print(texture_asset)
print(unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(material_instance, "Param3", 
                                                                                texture_asset ))

