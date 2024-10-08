
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
#import open3d as o3d
from dataclasses import dataclass
import yaml
#import sensor_msgs.point_cloud2 as pc2
import cv2
from pypcd4 import PointCloud
from scipy.spatial.transform import Rotation
from SceneSense import utils
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


def byte_string_to_pc(data):
    format_string = '<fffccc'

    # Calculate the size of each point in bytes
    point_size = struct.calcsize(format_string)

    # Use struct to unpack the byte string into a list of tuples
    points_list = struct.unpack_from(format_string * (len(data) // point_size), data)

    # Convert the list of tuples to a NumPy array
    return np.array(points_list).reshape(-1, 6)
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

        # Iterate over 100 row chunks of point cloud csv
        #for point_clouds in load_chunk_csv_pandas(octomap_in_file_name, 100):
        for chunk in load_chunk_csv_pandas(point_cloud_file_name, 100):
            # Iterate over each row
            for index, row in chunk.iterrows():

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

                tf_indices = np.argsort(np.abs(transforms['Time'].to_numpy() - row['Time']))


                tfs = []
                rot_index = 0
                for i in tf_indices:

                    if len(tfs) >= 2:
                        break
                    tf = yaml.safe_load(transforms.iloc[tf_indices[i]]['transforms'][1:-1].replace(', ', '\n'))
                    if tf['header']['frame_id'] == 't265_odom_frame' and (len(tfs) == 0 or tfs[0]['header']['frame_id'] != 't265_odom_frame'):
                        tfs.append(tf)
                    elif tf['header']['frame_id'] == '/t265_link' and (len(tfs) == 0 or tfs[0]['header']['frame_id'] != '/t265_link'):
                        rot_index = len(tfs)
                        tfs.append(tf)

                # tf_string = transforms.iloc[tf_indices[0]]['transforms'][1:-1].replace(', ', '\n')
                # tf = yaml.safe_load(tf_string)
                # Reformat string representation of fields into list of dataclasses
                fields = [FieldDataClass(*varify(s)) for s in chunker(re.split('\n|, ', row['fields'][1:-1]), 4)]
                row['fields'] = fields



                row['data'] = row['data'].encode().decode('unicode-escape').encode('ISO-8859-1')[2:-1]
                pc = PointCloud.from_msg(row)
                print(pc.fields)

                rot = tfs[rot_index]['transform']['rotation']
                rot = [rot[k] for k in ['x', 'y', 'z', 'w']]
                rotation_obj = Rotation.from_quat(rot)
                hm_tx_mat = utils.homogeneous_transform([0, 0, 0], rotation_obj.as_quat())
                #transform the pcd into the robot frame

                pc = pc.numpy()

                exit()
                t = utils.inverse_homogeneous_transform(hm_tx_mat)
                pc = t @ pc
                pc = PointCloud.from_points(pc)
                house_points = np.asarray(pc.points)
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


