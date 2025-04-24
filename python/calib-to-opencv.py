#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:43:40 2022

@author: lusk
"""
# %% Imports
import os
import json
import argparse
from datetime import datetime
import xml.etree.ElementTree as e

# JSON vals to string
def dict_to_str(obj, common_val=None):
    """
    Iterates through a dict and returns a string of the values and a count of
    the values.
    """
    count = 0
    out = ""
    for i, key in enumerate(obj):
        if common_val:
            if obj[key][common_val] != 0 and obj[key][common_val] != 1:
                count += 1
                out += str(obj[key][common_val]) + " "
        else:
            if obj[key] != 0 and obj[key] != 1:
                count += 1
                out += str(obj[key]) + " "

    return count, out.strip()


# Extract attributes and generate XML
def to_xml(filepath, xml_dir):
    """
    Takes a Calib.io JSON file and relevant distortion coefficients and writes
    an OpenCV XML file to disk.
    """

    # Open JSON file from filepath provided
    with open(filepath, "r") as f:
        jd = json.load(f)

    # Get file creation timestamp
    created = datetime.fromtimestamp(os.path.getmtime(filepath))
    created = created.strftime("%a %b %-d %H:%M:%S %Y")

    r = e.Element("opencv_storage")  # parent element
    e.SubElement(r, "calibration_Time").text = '"' + created + '"'
    cams = jd["Calibration"]["cameras"]

    for i, c in enumerate(cams):
        data = cams[i]["model"]["ptr_wrapper"]["data"]

        # Parameters
        ptrs = data["parameters"]
        p_list = []
        for param in ptrs:
            p_list.append(ptrs[param]["val"])

        # Now adust the parameter list order to match Open CV's dist_coeff
        # matrix ordering.
        # [f, ar, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tauX, tauY]
        opencv_order = [0, 1, 2, 3, 4, 5, 10, 11, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17]
        p_list[:] = [p_list[i] for i in opencv_order]

        img_size = data["CameraModelCRT"]["CameraModelBase"]["imageSize"]

        ##############################
        # Image width / height
        ##############################
        image_Width = e.SubElement(r, "image_Width")
        image_Width.text = str(img_size["width"])
        image_Height = e.SubElement(r, "image_Height")
        image_Height.text = str(img_size["height"])

        ##############################
        # Camera Matrix
        ##############################
        cam_matrix = e.SubElement(r, "Camera_Matrix", type_id="opencv-matrix")
        e.SubElement(cam_matrix, "rows").text = "3"
        e.SubElement(cam_matrix, "cols").text = "3"
        e.SubElement(cam_matrix, "dt").text = "d"

        cam_matrix_str = str(ptrs["f"]["val"]) + " 0. " + str(ptrs["cx"]["val"])
        cam_matrix_str += " 0. " + str(ptrs["f"]["val"]) + " " + str(ptrs["cy"]["val"])
        cam_matrix_str += " 0. 0. 1."
        e.SubElement(cam_matrix, "data").text = cam_matrix_str

        ##############################
        # Distortion Coefficients
        ##############################
        dist_coeff_ele = e.SubElement(
            r, "Distortion_Coefficients", type_id="opencv-matrix"
        )

        dist_coeff_str = ""
        num_coeffs = 0

        for j, val in enumerate(p_list[4:9]):
            # If we're going to have params beyond k1, ensure that k2, p1, p2 are
            # included, even if they're 0 (opencv needs them since the order is k1, k2,
            # p1, p2, k3, ...)
            num_coeffs += 1
            dist_coeff_str += str(val) + " "

        e.SubElement(dist_coeff_ele, "rows").text = str(num_coeffs)
        e.SubElement(dist_coeff_ele, "cols").text = "1"
        e.SubElement(dist_coeff_ele, "dt").text = "d"
        e.SubElement(dist_coeff_ele, "data").text = dist_coeff_str.strip()

        ##############################
        # Translation and Rotation matrices if more than one camera
        ##############################
        r_ct = 0  # rotation iteration
        t_ct = 0  # translation iteration

        if i > 0:
            rot = cams[i]["transform"]["rotation"]
            trans = cams[i]["transform"]["translation"]

            # Get rotation and translation strings and lengths
            rot_len, rot_str = dict_to_str(rot)
            trans_len, trans_str = dict_to_str(trans)

            # R tag
            r_tag = "R" if r_ct == 0 else "R" + str(r_ct)
            r_ele = e.SubElement(r, r_tag, type_id="opencv-matrix")
            e.SubElement(r_ele, "rows").text = str(rot_len)
            e.SubElement(r_ele, "cols").text = "1"
            e.SubElement(r_ele, "dt").text = "d"
            e.SubElement(r_ele, "data").text = rot_str

            # T tag
            t_tag = "T" if t_ct == 0 else "T" + str(t_ct)
            t_ele = e.SubElement(r, t_tag, type_id="opencv-matrix")
            e.SubElement(t_ele, "rows").text = str(trans_len)
            e.SubElement(t_ele, "cols").text = "1"
            e.SubElement(t_ele, "dt").text = "d"
            e.SubElement(t_ele, "data").text = trans_str

    a = e.ElementTree(r)  # Save full XML tree

    # Write XML file
    dir_path = os.path.dirname(os.path.abspath(filepath))

    if xml_dir == "":
        xml_dir = os.getcwd() + "/xml/"
        if not os.path.exists(os.getcwd() + "/xml"):
            os.makedirs(os.getcwd() + "/xml")

    xml_dir = xml_dir if xml_dir.endswith("/") else xml_dir + "/"

    out_filename = os.path.basename(filepath)
    out_filename = os.path.splitext(out_filename)[0] + ".xml"
    out_filepath = xml_dir + out_filename
    a.write(out_filepath, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    # Check the help parameters to understand arguments
    parser = argparse.ArgumentParser(
        description="Calib.io calibration (JSON) to OpenCV calibration (XML) (lusk@uni-potsdam.de)"
    )

    parser.add_argument(
        "--json_dir",
        type=str,
        required=False,
        default="",
        help="The location of the Calib.io JSON file(s)",
    )

    parser.add_argument(
        "--xml_dir",
        type=str,
        required=False,
        default="",
        help="The location where the XML file(s) should be saved. If empty, it/they will be saved in"
        " a new folder in the CWD called 'xml'.",
    )

    args = parser.parse_args()

    # Assign json_dir to user input or else CWD
    json_dir = args.json_dir if not args.json_dir == "" else os.getcwd()
    json_dir = json_dir if json_dir.endswith("/") else json_dir + "/"

    for file in os.listdir(json_dir):
        if file.endswith("json"):
            to_xml(json_dir + file, args.xml_dir)
        else:
            continue
