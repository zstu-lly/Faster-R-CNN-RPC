import os
import argparse

parser = argparse.ArgumentParser(
    description='Meshroom command line script')
parser.add_argument('--project', default='project.mg', type=str, help='project name')
parser.add_argument('--input', default='input', type=str, help='directory contains all rgb images')
parser.add_argument('--output', default='output', type=str, help='directory to save output obj file,mtl file '
                                                                 'and texture file')
args = parser.parse_args()

if not os.path.exists(args.project):
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    os.system("/home/lilinye/reconstruct/Meshroom-2019.1.0/meshroom_photogrammetry "
              f"--input {args.input} --output {args.output} --save {args.project}")

os.system(f"/home/lilinye/reconstruct/Meshroom-2019.1.0/meshroom_compute {args.project} --toNode Publish_1 --forceStatus")
