from glob import glob
import xml.etree.ElementTree as ET

texture_path = "..\\sequences\\sequence_bin1\\bin_eps5_a1.png"
# obj_file  = "" 
# light_translation = "30 0 0 0 0 30 0 0 0 0 1 15 0 0 0 1"

mi_files = sorted(glob("..\\sequences\\sequence_bin1\\xmls\\*.xml"))
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
