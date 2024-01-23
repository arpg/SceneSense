
import bagpy
from bagpy import bagreader
import pandas as pd
import os.path
import numpy as np 
import re
import io
import json
#from pypcd import pypcd
from pypcd4 import PointCloud
from dataclasses import dataclass

ODOMETRY_COLUMNS = ['pose.pose.position.x',
       'pose.pose.position.y', 'pose.pose.position.z',
       'pose.pose.orientation.x', 'pose.pose.orientation.y',
       'pose.pose.orientation.z']

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

    base_path = '/home/brendan/spot_data/'
    bag_name = 'IRL_lab_and_below'
    odometry_topic_name = '/D02/throttled_odometry'
    point_cloud_topic_name = '/D02/throttled_point_cloud'
    tf_topic_name = '/throttled_tf'
    bag = bagreader(base_path + bag_name + '.bag')
    num_point_clouds = bag.topic_table.iloc[2]['Message Count']
    
    text_output_filename = base_path + bag_name + '.txt'
    tf_file_name = build_csv(base_path + bag_name, tf_topic_name, bag)
    odometry_file_name = build_csv(base_path + bag_name, odometry_topic_name, bag)
    point_cloud_file_name = build_csv(base_path + bag_name, point_cloud_topic_name, bag)
    
    odometry = load_full_csv(odometry_file_name)
    #print(odometry.min(), odometry.max())
    tf = load_full_csv(tf_file_name)
    #print(tf.iloc[0]['transforms'])
    with open(text_output_filename, 'w') as file:
        for point_clouds in load_chunk_csv_pandas(point_cloud_file_name, 100):
            for index, row in point_clouds.iterrows():
                odometry_index = np.argmin(np.abs(odometry['Time'].to_numpy() - row['Time']))
                odom = odometry.iloc[odometry_index][ODOMETRY_COLUMNS]
                
                tf_index = np.argmin(np.abs(tf['Time'].to_numpy() - row['Time']))
                #print(point_clouds.columns)
                
                fields = [FieldDataClass(*varify(s)) for s in chunker(re.split('\n|, ', row['fields'][1:-1]), 4)]
                #print(fields)
                row['fields'] = fields
                cloud = row['data'].encode().decode('unicode-escape').encode('ISO-8859-1')[2:-1]
                row['data'] = np.frombuffer(cloud)
                #print(PointCloud.from_fileobj(io.StringIO(row['data'])))
                pc = PointCloud.from_msg(row)
                #print(pc.numpy().shape[0])
                file.write(f'NODE {" ".join([str(x) for x in odom])}\n')
                file.write(pc_to_str(pc.numpy()))
                if index % 100 == 0:
                    print("{:.2f}".format(index / num_point_clouds))
                #print(row['data'])
                #print(odometry_index, tf_index)
            

    #point_clouds = load_csv(base_path + bag_name, point_cloud_topic_name, bag)
    #print(point_clouds.head())