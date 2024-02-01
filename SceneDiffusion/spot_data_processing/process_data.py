
import bagpy
from bagpy import bagreader
import pandas as pd
import os.path
import numpy as np
from scipy.spatial.transform import Rotation as R
import re
import io
import json
#from pypcd import pypcd
from pypcd4 import PointCloud
from dataclasses import dataclass
import yaml

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


# f = open("explore_test/sample_octomap_running.txt", "a")
# write_str = "NODE " + str(depth_cam_pos[0]) +" " + str(depth_cam_pos[1]) + " " + str(depth_cam_pos[2]) + " " + str(depth_cam_rot[0]) + " " + str(depth_cam_rot[1])  + " " + str(depth_cam_rot[2])  + "\n"
# # print(write_str)
# f.write(write_str)
# f.close()
# #create pointcloud from depth image
# print(observations["depth"].shape)
# pointcloud = depth_to_pc(observations["depth"][:,:,0])
# print(pointcloud.shape)
# #save the pointclouda
# with open("explore_test/sample_octomap_running.txt", "ab") as f:
#     np.savetxt(f, pointcloud)
# f.close()

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
    base_path = '/home/brendan/spot_data/'
    bag_name = 'CSEL'
    dataset_dir = '/home/brendan/spot_data/dataset/raw/'
    data_dir = 'CSEL/'
    odometry_topic_name = '/D02/throttled_odometry'
    #point_cloud_topic_name = '/D02/throttled_point_cloud'
    tf_topic_name = '/throttled_tf'
    octomap_topic_name = '/D02/merged_map'
    octomap_in_topic_name = '/D02/throttled_octomap_in'


    # Read bagfile
    bag = bagreader(base_path + bag_name + '.bag')


    # Build .csv files
    num_point_clouds = bag.topic_table.iloc[2]['Message Count']
    text_output_filename = base_path + bag_name + '.log'
    tf_file_name = build_csv(base_path + bag_name, tf_topic_name, bag)
    odometry_file_name = build_csv(base_path + bag_name, odometry_topic_name, bag)
    #point_cloud_file_name = build_csv(base_path + bag_name, point_cloud_topic_name, bag)
    octomap_file_name = build_csv(base_path + bag_name, octomap_topic_name, bag)
    octomap_in_file_name = build_csv(base_path + bag_name, octomap_in_topic_name, bag)

    # Load .csv files into memory
    transforms = load_full_csv(tf_file_name)
    odometry = load_full_csv(odometry_file_name)


    import bagpy
    from bagpy import bagreader
    import matplotlib.pyplot as plt
    
 
    # Get information about the bag
    # info_dict = bag.message_by_topic(octomap_topic_name, start_time=0, stop_time=1)
    
    # # Extract Octomap messages
    # octomap_msgs = bag.message_by_topic(octomap_topic_name, start_time=0, stop_time=info_dict['Time'][0])
    
    # Save Octomap data to .bt file
    # output_bt_file = '/home/brendan/spot_data/output_octomap.bt'
    # with open(output_bt_file, 'wb') as bt_file:
    #     for msg in octomap_msgs:
    #         bt_file.write(msg[1].data)
    # with open(file_path, 'r') as file:
    #     # Create a CSV reader
    #     csv_reader = csv.reader(file)
    
    #     # Iterate over the lines in reverse order
    #     for row in reversed(list(csv_reader)):
    #         # 'row' now contains the last row of the CSV file
    #         final_map = row
    #         break  # Exit the loop after getting the last row

    #print(type(final_map), len(final_map))
    odometry[['roll', 'pitch', 'yaw']] = R.from_quat(odometry[ODOMETRY_QUAT].to_numpy()).as_rotvec()


    pc_path = os.path.join(dataset_dir, data_dir, 'point_clouds/')
    os.makedirs(pc_path, exist_ok=True)
    transforms_path = os.path.join(dataset_dir, data_dir, 'transforms/')
    os.makedirs(transforms_path, exist_ok=True)
    odom_path = os.path.join(dataset_dir, data_dir, 'odometry/')
    os.makedirs(odom_path, exist_ok=True)
    # Loop over all point clouds
    with open(text_output_filename, 'w') as file:
        # Iterate over 100 row chunks of point cloud csv
        for point_clouds in load_chunk_csv_pandas(octomap_in_file_name, 100):
            # Iterate over each row
            for index, row in point_clouds.iterrows():

                # Find closest odometry message to current point cloud message
                odometry_index = np.argmin(np.abs(odometry['Time'].to_numpy() - row['Time']))
                odom = [odometry.iloc[odometry_index][k] for k in ODOMETRY_LOCATIONS]
                odom += [odometry.iloc[odometry_index][k] for k in ODOMETRY_RPY]
                #odom_dict = {k: v for k, v in zip(ODOMETRY_OUTPUT_NAMES, odom.to_numpy())}
                # Find closest transform message to currect point cloud message

                tf_indicies = np.argsort(np.abs(transforms['Time'].to_numpy() - row['Time']))
                
                tf_string = transforms.iloc[tf_indicies[0]]['transforms'][1:-1].replace(', ', '\n')
                tf = yaml.safe_load(tf_string)
            
                # Reformat string representation of fields into list of dataclasses
                fields = [FieldDataClass(*varify(s)) for s in chunker(re.split('\n|, ', row['fields'][1:-1]), 4)]
                row['fields'] = fields

                # Encode str of binary represenation of point cloud into bytes var
             
                cloud = row['data'].encode().decode('unicode-escape').encode('ISO-8859-1')[2:-1]
                row['data'] = np.frombuffer(cloud)
                pc = PointCloud.from_msg(row)
                

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
            
            
