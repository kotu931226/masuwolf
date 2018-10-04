import os
import re
import pickle
import random
import numpy as np



class LoadLog:
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    relative_path2log = './log/log_2018_5/'
    relative_path = os.path.join(this_file_path, relative_path2log)
    people_num = 5
    search_CO = ["SEER", "WEREWOLF", "POSSESSED"]

    def __init__(self):
        self.count_lines = 0
        self.line_counter = []
        self.wolf_vec = []
        self.log_wolf_vec = []

    def see_all_file(self):
        with os.scandir(self.relative_path) as its:
            for entrys in its:
                with os.scandir(entrys.path) as it:
                    for entry in it:
                        # This is carved file
                        print(entry.path)
                        self.count_lines = 0
                        self.capture_log(entry.path)
                        self.line_counter.append(self.count_lines)

    def generate_file(self, path):
        with os.scandir(path) as its:
            for entrys in its:
                with os.scandir(entrys.path) as it:
                    for entry in it:
                        with open(entry.path, 'r') as f:
                            print(entry.path)
                            yield [[w for w in l.split(',')] for l in f.read().splitlines()]

    def emit_onehot_t(self):
        onehot_t = []
        for lines in self.generate_file(self.relative_path):
            dir_t = [0 for _ in range(self.people_num)]
            for line in lines:
                # Example line ['1', 'talk', '8', '1', '2', 'DIVINED Agent[01] HUMAN']
                # Example line [0:day, 1:type, 2:talk_id, 3:turn_num, 4:Agent_num, 5:talk_content]
                if line[0] == '0':
                    if line[3] == 'WEREWOLF':
                        dir_t[int(line[2])-1] = 1
            onehot_t.append(dir_t)
        return onehot_t

    def create_log_path(self, direc_num, log_num):
        path_by_env = f'{direc_num:03}/{log_num:03}.log'
        return os.path.join(self.relative_path, path_by_env)

    def capture_log(self, log_path):
        with open(log_path, 'r') as log_file:
            lines = [l for l in log_file.read().splitlines()]

        for line in lines:
            word = [w for w in line.split(',')]
            self.find_something(word)

        self.log_wolf_vec.append(self.wolf_vec)
        self.wolf_vec = []

    # Edit this function!!
    def find_something(self, word):
        if not any(map(lambda x: x in word[2], ('talk', 'status'))):
            return
        if ('Skip' in word[5] or 'Over' in word[5]):
            return
        if any(map(lambda x: x in word[5], ('COMINGOUT', 'VOTE', 'DIVINE', 'ESTIMATE'))):

            # print(word)
            self.word2wolf_vec(word)

            return
        self.count_lines += 1

    def emit_onehot_y(self):
        for lines in self.generate_file(self.relative_path):
            self.wolf_vec = []
            for line in lines[:-7]:
                self.word2wolf_vec(line)
            self.log_wolf_vec.append(self.wolf_vec)
        return

    def word2wolf_vec(self, word):
        # Example ['1', 'talk', '8', '1', '2', 'DIVINED Agent[01] HUMAN']
        # Example [0:day, 1:type, 2:talk_id, 3:turn_num, 4:Agent_num, 5:talk_content]
        if 'talk' in word[1]:
            if any(map(lambda x: x in word[5], ('Skip', 'Over'))):
                return
            if not any(map(lambda x: x in word[5], ('COMINGOUT', 'VOTE', 'DIVINE', 'ESTIMATE'))):
                return
            print(word)
            target_num = self.extract_target(word[5])
            if 'COMINGOUT' in word[5]:
                for i, co_role in enumerate(self.search_CO):
                    if co_role in word[5]:
                        self.wolf_vec.append(self.emit_id('CO', i+1, int(word[4])))
            elif 'VOTE' in word[5]:
                self.wolf_vec.append(self.emit_id('VOTE', int(word[4]), target_num))
            elif 'DIVINE' in word[5]:
                if 'HUMAN' in word[5]:
                    self.wolf_vec.append(self.emit_id('DIVINE', int(word[4]), target_num))
                elif 'WEREWOLF' in word[5]:
                    self.wolf_vec.append(self.emit_id('DIVINE_1', int(word[4]), target_num))
        if 'status' in word[1]:
            if 'DEAD' in word[4]:
                target_num = int(word[2])
                self.wolf_vec.append(self.emit_id('DEAD', 1, target_num))

    def extract_target(self, talk_content):
        return int(re.search(r"\d{2}", talk_content).group())


    def emit_id(self, state, number, my_number=1):
        until_DEAD = 1
        until_CO = until_DEAD + len(self.search_CO)
        until_VOTE = until_CO + self.people_num
        until_DIVINE = until_VOTE + self.people_num
        total_id_num = (until_DIVINE + self.people_num) * (my_number - 1)
        # total_id_num is 19

        if state == 'DEAD':
            return total_id_num + 1
        if state == 'CO':
            return total_id_num + until_DEAD + number
        if state == 'VOTE':
            return total_id_num + until_CO + number
        if state == 'DIVINE':
            return total_id_num + until_VOTE + number
        if state == 'DIVINE_1':
            return total_id_num + until_DIVINE + number


class ConvertWolfLog(object):
    wolf_log_path = 'log_wolf_vec.bf'
    def __init__(self):
        self.wolf_vec = []
        self.load_wolf_log()
        self.elm_num = 8
        self.wolf_vec_certian_elm = []

    def load_wolf_log(self):
        with open(self.wolf_log_path, 'rb') as f:
            self.wolf_vec = pickle.load(f)

    def convert_certian_elm(self):
        for one_match in self.wolf_vec:
            self.wolf_vec_certian_elm.append(self.emit_certian_elm(one_match))
        


    def emit_certian_elm(self, one_match):
        diff_num = len(one_match) - self.elm_num
        if diff_num < 0:
            for _ in range(abs(diff_num)):
                one_match.insert(0, 0)
            return one_match
        elif diff_num > 0:
            dir_diff = diff_num * random.random()
            # Be carefull random == 1.0?
            if dir_diff == diff_num:
                dir_diff = 0
            return one_match[int(dir_diff) : int(dir_diff + self.elm_num)]
        return one_match


def conv_onehot(label_matrix, max_num=96):
    ndim0 = label_matrix.shape[0]
    ndim1 = label_matrix.shape[1]
    onehot_matrix = np.zeros((ndim0, ndim1, max_num), dtype=np.int32)
    for idx_0, word_ids in enumerate(label_matrix):
        for idx_1, word_id in enumerate(word_ids):
            onehot_matrix[idx_0, idx_1, word_id] = 1
    return onehot_matrix
    

def main():
    load_log = LoadLog()
    # load_log.emit_onehot_y()
    onehot_y = np.array(load_log.emit_onehot_t())
    
    # load_log.see_all_file()
    # print(load_log.log_wolf_vec[0])

    # # print(load_log.line_counter)
    # # print(sum(load_log.line_counter))
    # with open('log_wolf_vec.bf', 'wb') as f:
    #     pickle.dump(load_log.log_wolf_vec, f)

    convert_wolf_log = ConvertWolfLog()
    convert_wolf_log.convert_certian_elm()

    wolf_labels = np.array(convert_wolf_log.wolf_vec_certian_elm)
    wolf_onehot = conv_onehot(wolf_labels)

    print(onehot_y.shape, wolf_onehot.shape)

    np.save('onehot_x.npy', wolf_onehot)
    np.save('onehot_y.npy', onehot_y)

    # print(len(onehot_t), len(wolf_onehot))
if __name__ == '__main__':
    main()
