import os
import shutil
# import loadlog

# Don't use until exist log_2018_5
# This program is to create log only 5people

def get_dir_size(path='.'):
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(entry.path)
    return total

def create_log_direc_path(direc_num):
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    path_by_env = './log_cedec2018/{:03}/'.format(direc_num)
    return os.path.join(this_file_path, path_by_env)

def create_new_log_direc_path(direc_num):
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    path_by_env = './log_2018_5/{:03}/'.format(direc_num)
    return os.path.join(this_file_path, path_by_env)

num_5 = 0
num_15 = 0

for i in range(1, 203+1):
    log_direc_path = create_log_direc_path(i)
    dir_size = get_dir_size(log_direc_path)
    print(i, dir_size)
    if dir_size < 1e6:
        new_direc_path = create_new_log_direc_path(num_5)
        shutil.copytree(log_direc_path, new_direc_path)
        num_5 += 1
    else:
        num_15 += 1
print(num_5, num_15)
