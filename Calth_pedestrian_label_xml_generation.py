from lxml import etree
import os
import re


def create_object_xml_node(root, xmin, ymin, xmax, ymax):
    node_object = etree.SubElement(root, "object")
    # ----- properties below belongs to node: object ----- # this is only a template
    node_name = etree.SubElement(node_object, "name") # give default value: pedestrian
    node_name.text = "pedestrian"
    node_pose = etree.SubElement(node_object, "pose") # give default value: unspecified
    node_pose.text = "unspecified"
    node_truncated = etree.SubElement(node_object, "truncated") # give default value: 0
    node_truncated.text = "0"
    node_difficult = etree.SubElement(node_object, "difficult") # give default value: 1
    node_difficult.text = "0"
    node_bndbox = etree.SubElement(node_object, "bndbox") # give default value: 1
    #  ----- properties below belongs to node: bndbox ----- #
    node_xmin = etree.SubElement(node_bndbox, "xmin")
    node_xmin.text = str(xmin)
    node_ymin = etree.SubElement(node_bndbox, "ymin")
    node_ymin.text = str(ymin)
    node_xmax = etree.SubElement(node_bndbox, "xmax")
    node_xmax.text = str(xmax)
    node_ymax = etree.SubElement(node_bndbox, "ymax")
    node_ymax.text = str(ymax)
    # ----- properties above belongs to node: bndbox ----- #
    return root


def create_annotation_for_one_image(image_file_name):
    root = etree.Element("annotation")

    node_folder = etree.SubElement(root, "folder") # give default value: caltech pedestrian
    node_folder.text = "VOC2007"

    node_filename = etree.SubElement(root, "filename") # one parameter
    node_filename.text = image_file_name

    node_source = etree.SubElement(root, "source")
    # ----- properties below belongs to node: source ----- #
    node_database = etree.SubElement(node_source, "database") # give default value: caltech pedestrian
    node_database.text = "The VOC2007 Database"

    node_annotation = etree.SubElement(node_source, "annotation") # give default value: caltech pedestrian
    node_annotation.text = "PASCAL VOC2007"

    node_image = etree.SubElement(node_source, "image") # give default value: caltech_pedestrian
    node_image.text = "caltech pedestrian"
    node_flickerid = etree.SubElement(node_source, "flickerid") # give default value: 32767
    node_flickerid.text = "0000000000"
    # ----- properties above belongs to node: source ----- #

    node_owner = etree.SubElement(root, "owner") # give default value: Celtech
    # ----- properties below belongs to node: owner ----- #
    node_owner_flickerid = etree.SubElement(node_owner, "flickerid") # give default value: GO_BLUE
    node_owner_flickerid.text = "GO BLUE"
    node_owner_name = etree.SubElement(node_owner, "name") # give default value: UMD_ISL
    node_owner_name.text = "UMD_ISL"
    # ----- properties above belongs to node: owner ----- #

    node_size = etree.SubElement(root, "size")
    # ----- properties below belongs to node: source ----- #
    node_width = etree.SubElement(node_size, "width")
    node_width.text = "640"
    node_height = etree.SubElement(node_size, "height")
    node_height.text = "480"
    node_depth = etree.SubElement(node_size, "depth")
    node_depth.text = "3"
    # ----- properties above belongs to node: source ----- #

    node_segmentation = etree.SubElement(root, "segmentation") # give default value: 0
    node_segmentation.text = "0"

    return root


def main():
    # dir folder
    output_path = '/home/zhans/zhans/Data/Caltech_Pedestrian/after/output/annotations'
    # traverse root directory, and list directories as dirs and files as files
    for root_path, dirs, files in os.walk("/home/zhans/zhans/Data/Caltech_Pedestrian/after/annot_format_change"):
        for file in files:
            print(root_path, file)
            # use regular expression to extract the set index.
            m = re.search('[0-9]+', root_path)
            if m:
                dir_index = (m.group(0)) # may have bug here.
                print(dir_index)

                n = re.search('[0-9]+', file)
                if n:
                    file_index = (n.group(0))
                    print(file_index)

                    file_path = root_path + '/' + file
                    with open(file_path, 'r') as f:
                        old_frame_index = -1
                        lines = f.readlines()

                        for i in range(0, len(lines)):
                            one_line_data = lines[i][:-2] # ignore the last return symbol
                            [frame_index, x_min, y_min, width, height] = one_line_data.split(' ')
                            x_max = int(float(x_min) + float(width))
                            y_max = int(float(y_min) + float(height))
                            x_min = int(float(x_min))
                            y_min = int(float(y_min))

                            frame_index = int(frame_index)

                            if old_frame_index != frame_index: # this means it's a new image
                                if old_frame_index != -1: # means it's not the first labeled frame
                                    annotation_xml_file_name = output_path + '/' + dir_index + file_index + (5 - len(str(old_frame_index - 1))) * "0" + str(old_frame_index - 1) + '.xml'
                                    with open(annotation_xml_file_name, "wb") as f:
                                        f.write(etree.tostring(root, pretty_print=True))
                                old_frame_index = frame_index

                                image_file_name = dir_index + file_index + (5 - len(str(frame_index - 1))) * "0" + str(frame_index - 1) + '.jpg'
                                root = create_annotation_for_one_image(image_file_name)
                                root = create_object_xml_node(root, x_min, y_min, x_max, y_max)
                            else:
                                # only create another object node in the xml file
                                root = create_object_xml_node(root, x_min, y_min, x_max, y_max)
                else:
                    continue
            else:
                continue

if __name__ == "__main__": main()

