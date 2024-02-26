
import bagpy
from bagpy import bagreader
import pandas as pd
import os.path
import numpy as np
from typing import List, Iterator
from scipy.spatial.transform import Rotation as R
import re
import io
import json
import struct
#from pypcd import pypcd
from pypcd4 import PointCloud
from dataclasses import dataclass
import yaml
import cv2

ODOMETRY_LOCATIONS = ['pose.pose.position.x',
       'pose.pose.position.y', 'pose.pose.position.z']

ODOMETRY_QUAT = ['pose.pose.orientation.x', 'pose.pose.orientation.y',
       'pose.pose.orientation.z', 'pose.pose.orientation.w']

ODOMETRY_RPY = ['roll', 'pitch', 'yaw']

ODOMETRY_OUTPUT_NAMES = [
    'x', 'y', 'z', 'rx', 'ry', 'rx', 'rw'
]


def replace_char(string: str, sub: str, wanted: str, n: int) -> str:
    where = [m.start() for m in re.finditer(sub, string)][n-1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    new_string= before + after
    return new_string

def build_csv(path: str, topic_name: str, bag: bagreader) -> str:
    if topic_name.count('/') > 1:
        topic_file_name = path + replace_char(topic_name, '/', '-', 2) + '.csv'
    else:
        topic_file_name = path + topic_name + '.csv'
    if not os.path.isfile(topic_file_name):
        topic_file_name = bag.message_by_topic(topic_name)
    return topic_file_name

def load_full_csv(topic_file_name: str) -> pd.DataFrame:
    topic_csv = pd.read_csv(topic_file_name)
    return topic_csv

def load_chunk_csv(topic_file_name: str, chunk_size: int, num_rows: int):
    skip_rows = 0
    while skip_rows < num_rows:
        df = pd.read_csv(topic_file_name, skiprows=skip_rows, nrows=chunk_size)
        yield df
        skip_rows += chunk_size

def load_chunk_csv_pandas(topic_file_name: str, chunk_size: int):

    with pd.read_csv(topic_file_name, chunksize=chunk_size) as reader:
        print(reader)
        for chunk in reader:
            yield chunk

def load_chunk_csv_pandas_multi(topic_file_names: List[str], chunk_sizes: List[int], num_chunks: int):

    readers = [pd.read_csv(name, iterator=True) for name, chunk_size in zip(topic_file_names)]
    for _ in range(num_chunks):
        yield [reader.get_chunk(chunk_size) for reader, chunk_size in zip(readers, chunk_sizes)]

def jsonify(strings):
    string = '{'
    for s in strings:
        string += '"' + s.replace(':', '":') + ', '
    string = string[:-2] + '}'
    return string

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def varify(strings):
    return [s.split(': ')[1] for s in strings]

def pc_to_str(pc):
    good_idx = ~np.any(np.isnan(pc), axis=1)
    pc = pc[good_idx, :]

    return ''.join([' '.join([str(x) for x in xx]) + '\n' for xx in pc[:, :3]])

@dataclass
class FieldDataClass(object):
    name: str
    offset: int
    datatype: int
    count: int
    
    def __init__(self, name: str, offset: int, datatype: int, count: int) -> None:
        self.name = name.strip('"')
        self.count = int(count)
        self.offset = int(offset)
        self.datatype = int(datatype)

if __name__ == '__main__':

    # Paths
    base_path = '/home/brendan/realsense_data/'
    bag_name = 'doncey2'
    dataset_dir = '/home/brendan/realsense_data/dataset/raw/'
    data_dir = 'doncey2/'
    point_cloud_topic_name = '/d400/throttled_point_cloud'
    odom_topic_name = '/t265/odom/sample'
    tf_topic_name = '/tf'


    # Read bagfile
    bag = bagreader(base_path + bag_name + '.bag')


    # Build .csv files
    tt = bag.topic_table.set_index('Topics')
    num_point_clouds = tt.loc['/d400/throttled_point_cloud']['Message Count']
    text_output_filename = base_path + bag_name + '.log'

    # camera_info_name = build_csv(base_path + bag_name, camera_info_topic_name, bag)
    # raw_name = build_csv(base_path + bag_name, raw_topic_name, bag)
    # tf_name = build_csv(base_path + bag_name, tf_topic_name, bag)
    # depth_name = build_csv(base_path + bag_name, depth_topic_name, bag)

    point_cloud_file_name = build_csv(base_path + bag_name, point_cloud_topic_name, bag)
    tf_file_name = build_csv(base_path + bag_name, tf_topic_name, bag)
    odom_file_name = build_csv(base_path + bag_name, odom_topic_name, bag)


    # Load .csv files into memory
    transforms = load_full_csv(tf_file_name)
    odometry = load_full_csv(odom_file_name)
    

    odometry[['roll', 'pitch', 'yaw']] = R.from_quat(odometry[ODOMETRY_QUAT].to_numpy()).as_rotvec()


    pc_path = os.path.join(dataset_dir, data_dir, 'point_clouds/')
    os.makedirs(pc_path, exist_ok=True)
    transforms_path = os.path.join(dataset_dir, data_dir, 'transforms/')
    os.makedirs(transforms_path, exist_ok=True)
    odom_path = os.path.join(dataset_dir, data_dir, 'odometry/')
    os.makedirs(odom_path, exist_ok=True)
    # # Loop over all point clouds
    with open(text_output_filename, 'w') as file:

        # num_chunks, remaining = divmod(num_point_clouds)
        # pc_chunks = [100 for _ in range(num_chunks)] + [remaining]
        # octo_chunks = [num_octompa_in // num_chunks for _ in range(num_chunks)] + [num_octompa_in % num_chunks]

        # print(len(pc_chunks), len(octo_chunks))
        # print(pc_chunks, octo_chunks)
        # exit()

        # Iterate over 100 row chunks of point cloud csv
        #for point_clouds in load_chunk_csv_pandas(octomap_in_file_name, 100):
        for chunk in load_chunk_csv_pandas(point_cloud_file_name, 100):
            # Iterate over each row
            for index, row in chunk.iterrows():


                
                print(row)
                #img = row['data'].encode().decode('unicode-escape').encode('ISO-8859-1')[2:-1]
                # print(img)
                # img = np.frombuffer(row['data'].encode('utf-8').decode('unicode-escape').encode('ISO-8859-1')[2:-1], dtype=np.uint8)
                # img = img.reshape(row['height'], row['width'], 2)
                # print(img.shape)
                # img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
                # cv2.imshow('image', img)
                # img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
                # cv2.imshow('image', img.reshape(row['height'], row['width']))

                # Find closest odometry message to current point cloud message
                odometry_index = np.argmin(np.abs(odometry['Time'].to_numpy() - row['Time']))

                odom = [odometry.iloc[odometry_index][k] for k in ODOMETRY_LOCATIONS]
                odom += [odometry.iloc[odometry_index][k] for k in ODOMETRY_RPY]
                odom_dict = {k: v for k, v in zip(ODOMETRY_OUTPUT_NAMES, np.array(odom))}
                # Find closest transform message to currect point cloud message

                tf_indicies = np.argsort(np.abs(transforms['Time'].to_numpy() - row['Time']))
                
                tf_string = transforms.iloc[tf_indicies[0]]['transforms'][1:-1].replace(', ', '\n')
                tf = yaml.safe_load(tf_string)
            
                # Reformat string representation of fields into list of dataclasses
                fields = [FieldDataClass(*varify(s)) for s in chunker(re.split('\n|, ', row['fields'][1:-1]), 4)]
                row['fields'] = fields

                fields = [
                    ('x', np.float32),
                    ('y', np.float32),
                    ('z', np.float32),
                    ('rgb', np.bytes_)
                    # Add more fields as needed
                ]

                # Create a NumPy dtype for the structured array
                dtype = np.dtype(fields)
                num_points = len(row['data']) // dtype.itemsize
                # Use the struct module to unpack the binary data into a NumPy array
                pc = np.frombuffer(row['data'].encode('utf-8'), dtype=dtype, count=row['width'])
                print(pc)
                print(pc.shape)
                pc = np.array([[x.x, x.y, x.z] for x in pc])
                print(pc.shape)
                # cloud = row['data'].encode('utf-8').decode('unicode-escape').encode('ISO-8859-1')[2:-1]
                # row['data'] = np.frombuffer(cloud)
                # pc = PointCloud.from_msg(row)
                

                # Save point clouds, odometry, and transforms
                pc.save(dataset_dir + data_dir + 'point_clouds/' + str(index).zfill(4) + '.pcd')
                with open(dataset_dir + data_dir + 'transforms/' + str(index).zfill(4) + '.yaml', 'w') as f:
                    try:
                        yaml.safe_dump(tf, f)
                    except:
                        print(tf)
                        exit()
                np.save(dataset_dir + data_dir + 'odometry/' + str(index).zfill(4), np.array(odom))
                
                # Write point clouds and odometry to .log file for octomap
                file.write(f'NODE {" ".join([str(x) for x in odom])}\n')
                file.write(pc_to_str(pc.numpy()))
                if index % 100 == 0:
                    print("{:.2f}".format(index / num_point_clouds))
            
            
