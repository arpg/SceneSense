import bagpy
from bagpy import bagreader
import pandas as pd
import os.path

import re

def replace_char(string: str, sub: str, wanted: str, n: int) -> str:
    where = [m.start() for m in re.finditer(sub, string)][n-1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    new_string= before + after
    return new_string

def build_csv(path: str, topic_name: str, bag: bagreader) -> str:
    topic_file_name = path + replace_char(topic_name, '/', '-', 2) + '.csv'
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
        print(df.head())
        yield df
        skip_rows += chunk_size
    


if __name__ == '__main__':

    base_path = '/home/brendan/spot_data/'
    bag_name = 'IRL_lab_and_below'
    odometry_topic_name = '/D02/throttled_odometry'
    point_cloud_topic_name = '/D02/throttled_point_cloud'
    point_cloud_file_name = base_path + bag_name + replace_char(point_cloud_topic_name, '/', '-', 2) + '.csv'
    bag = bagreader(base_path + bag_name + '.bag')
    num_point_clouds = bag.topic_table.iloc[2]['Message Count']
    
    odometry_file_name = build_csv(base_path + bag_name, odometry_topic_name, bag)
    odometry = load_full_csv(odometry_file_name)
    for point_clouds in load_chunk_csv(point_cloud_file_name, 100, num_point_clouds):
        print(point_clouds.head())

    #point_clouds = load_csv(base_path + bag_name, point_cloud_topic_name, bag)
    #print(point_clouds.head())