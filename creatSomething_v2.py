import os
import re
import pickle
import random
import math
import numpy as np



class LoadLog:
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    relative_path2log = './log/log_2018_5/'
    relative_path = os.path.join(this_file_path, relative_path2log)
    people_num = 5
    search_CO = ["SEER", "WEREWOLF", "POSSESSED"]

    def __init__(self):
        self.wolf_vec = []
        self.log_wolf_vec = []

    def generate_file(self, path):
        with os.scandir(path) as its:
            for entrys in its:
                with os.scandir(entrys.path) as it:
                    for entry in it:
                        with open(entry.path, 'r') as f:
                            print(entry.path)
                            yield [[w for w in l.split(',')] for l in f.read().splitlines()]

    # Example line ['1', 'talk', '8', '1', '2', 'DIVINED Agent[01] HUMAN']
    # Example line [0:day, 1:type, 2:talk_id, 3:turn_num, 4:Agent_num, 5:talk_content]
    def emit_onehot_t(self):
        onehot_t = []
        for lines in self.generate_file(self.relative_path):
            dir_t = [0 for _ in range(self.people_num)]
            for line in lines:
                if line[0] == '0':
                    if line[3] == 'WEREWOLF':
                        dir_t[int(line[2])-1] = 1
            onehot_t.append(dir_t)
        return onehot_t

    def emit_wolf_vec(self):
        for lines in self.generate_file(self.relative_path):
            self.wolf_vec = []
            for line in lines[:-7]:
                self.word2wolf_vec(line)
            self.log_wolf_vec.append(self.wolf_vec)
            print(self.wolf_vec)
        return

    def word2wolf_vec(self, word):
        # Example ['1', 'talk', '8', '1', '2', 'DIVINED Agent[01] HUMAN']
        # Example [0:day, 1:type, 2:talk_id, 3:turn_num, 4:Agent_num, 5:talk_content]
        if 'talk' in word[1]:
            if not any(map(lambda x: x in word[5], ('Skip', 'Over', 'COMINGOUT', 'VOTE', 'DIVINE'))):
                self.wolf_vec.append(self.emit_id_112('undefined_talk', 1, int(word[4])))
                return
            if 'Over' in word[5]:
                self.wolf_vec.append(self.emit_id_112('Over', 1, int(word[4])))
                return
            if 'Skip' in word[5]:
                self.wolf_vec.append(self.emit_id_112('Skip', 1, int(word[4])))
                return
            target_num = self.extract_target(word[5])
            if 'COMINGOUT' in word[5]:
                for i, co_role in enumerate(self.search_CO):
                    if co_role in word[5]:
                        self.wolf_vec.append(self.emit_id_112('CO', i+1, int(word[4])))
            elif 'VOTE' in word[5]:
                self.wolf_vec.append(self.emit_id_112('VOTE', int(word[4]), target_num))
            elif 'DIVINE' in word[5]:
                if 'HUMAN' in word[5]:
                    self.wolf_vec.append(self.emit_id_112('DIVINE', int(word[4]), target_num))
                elif 'WEREWOLF' in word[5]:
                    self.wolf_vec.append(self.emit_id_112('DIVINE_1', int(word[4]), target_num))
        if 'status' in word[1]:
            if 'DEAD' in word[4]:
                self.wolf_vec.append(self.emit_id_112('DEAD', 1, int(word[2])))

    def extract_target(self, talk_content):
        return int(re.search(r"\d{2}", talk_content).group())

    def emit_id_112(self, state, number, my_number=1):
        until_DEAD = 4
        until_CO = until_DEAD + len(self.search_CO)
        until_VOTE = until_CO + self.people_num
        until_DIVINE = until_VOTE + self.people_num
        until_DIVINE_1 = until_DIVINE + self.people_num
        solo_id_num = until_DIVINE_1
        total_id_num = solo_id_num * (my_number - 1)
        # total_id_num is 19

        if state == 'undefined_talk':
            return total_id_num + 1
        if state == 'Over':
            return total_id_num + 2
        if state == 'Skip':
            return total_id_num + 3
        if state == 'DEAD':
            return total_id_num + 4
        if state == 'CO':
            return total_id_num + until_DEAD + number
        if state == 'VOTE':
            return total_id_num + until_CO + number
        if state == 'DIVINE':
            return total_id_num + until_VOTE + number
        if state == 'DIVINE_1':
            return total_id_num + until_DIVINE + number


class ConvertWolfLog(object):
    wolf_log_path = 'log_wolf_vec_v2.bf'
    elm_num = 48
    def __init__(self):
        self.wolf_vec = []
        self.load_wolf_log()
        self.wolf_vec_certian_elm = []

    def load_wolf_log(self):
        with open(self.wolf_log_path, 'rb') as f:
            self.wolf_vec = pickle.load(f)

    def convert_certian_elm(self):
        for one_match in self.wolf_vec:
            self.wolf_vec_certian_elm.append(self.emit_certian_elm(one_match))
        


    def emit_certian_elm(self, one_match):
        random_pos = math.floor(len(one_match) * random.random())
        over_length = random_pos + self.elm_num - len(one_match)
        if over_length > 0:
            fix_length = one_match[random_pos:]
            for _ in range(over_length):
                fix_length.insert(0, 0)
            return fix_length
        else:
            return one_match[random_pos : random_pos + self.elm_num]


def conv_onehot(label_matrix, max_num=112):
    ndim0 = label_matrix.shape[0]
    ndim1 = label_matrix.shape[1]
    onehot_matrix = np.zeros((ndim0, ndim1, max_num), dtype=np.int32)
    for idx_0, word_ids in enumerate(label_matrix):
        for idx_1, word_id in enumerate(word_ids):
            onehot_matrix[idx_0, idx_1, word_id] = 1
    return onehot_matrix
    

def main():
    load_log = LoadLog()
    onehot_y = np.array(load_log.emit_onehot_t())

    # load_log.emit_wolf_vec()
    # with open('log_wolf_vec_v2.bf', 'wb') as f:
    #     pickle.dump(load_log.log_wolf_vec, f)

    convert_wolf_log = ConvertWolfLog()
    convert_wolf_log.convert_certian_elm()

    wolf_labels = np.array(convert_wolf_log.wolf_vec_certian_elm)
    onehot_x = conv_onehot(wolf_labels)

    print(onehot_x.shape, onehot_y.shape)

    # np.save('onehot_x_112.npy', onehot_x)
    # np.save('onehot_y_112.npy', onehot_y)

    # print(len(onehot_t), len(wolf_onehot))
if __name__ == '__main__':
    main()
