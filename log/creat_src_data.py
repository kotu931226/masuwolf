import os
import re
import pickle
import random
import math
import csv
import numpy as np



class LoadLog:
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    relative_path2log = './log_2018_5/'
    relative_path = os.path.join(this_file_path, relative_path2log)
    people_num = 5
    search_CO = ["SEER", "WEREWOLF", "POSSESSED"]

    def __init__(self):
        self.wolf_vec = []
        self.log_wolf_vec = []

    def generate_file(self, path):
        # with os.scandir(path) as its:
        #     for entrys in its:
        #         with os.scandir(entrys.path) as it:
        #             for entry in it:
        #                 with open(entry.path, 'r') as f:
        #                     print(entry.path)
        #                     yield [[w for w in l.split(',')] for l in f.read().splitlines()]
        for entries in os.scandir(path):
            for entry in os.scandir(entries.path):
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
                    if line[3] == 'POSSESSED':
                        dir_t[int(line[2])-1] = 1
            onehot_t.append(dir_t)
        return onehot_t

    def emit_t(self):
        tgt = []
        for lines in self.generate_file(self.relative_path):
            dir_t = []
            for line in lines:
                if line[0] == '0':
                    if line[3] == 'VILLAGER':
                        dir_t.append(line[2])
            tgt.append(dir_t)
        return tgt

    def emit_wolf_vec(self):
        for lines in self.generate_file(self.relative_path):
            self.wolf_vec = []
            for line in lines:
                self.word2wolf_vec_v2(line)
            self.log_wolf_vec.append(self.wolf_vec)
            # print(self.wolf_vec)
        return



    def word2wolf_vec_v2(self, word):
        if word[0] != '1':
            return
        if not 'talk' in word[1]:
            return 
        talk_info = word[5].split()
        if not any(map(lambda x: x in word[5], ('Skip', 'Over', 'COMINGOUT', 'VOTE', 'DIVINED', 'ESTIMATE'))):
            self.wolf_vec.append(self.emit_id_320(word[4], 'MIX', mix_api='unk'))
            return
        if talk_info[0] in ('Skip', 'Over'):
            self.wolf_vec.append(self.emit_id_320(word[4], 'MIX', mix_api=talk_info[0]))
            return
        tgt_num = self.extract_target(word[5])
        if talk_info[0] in ('VOTE',):
            self.wolf_vec.append(self.emit_id_320(word[4], talk_info[0], tgt_num=tgt_num))
            return
        if talk_info[0] in ('COMINGOUT', 'ESTIMATE', 'DIVINED'):
            self.wolf_vec.append(self.emit_id_320(word[4], talk_info[0], talk_info[2], tgt_num))
            return
        # Example ['1', 'talk', '8', '1', '2', 'DIVINED Agent[01] HUMAN']
        # Example [0:day, 1:type, 2:talk_id, 3:turn_num, 4:Agent_num, 5:talk_content]
        

    @classmethod
    def emit_id_320(cls, src_num, thing=None, role='HUMAN', tgt_num=1, mix_api=None):
        src_num = int(src_num)
        tgt_num -= 1
        things = {'COMINGOUT':0, 'ESTIMATE':20, 'DIVINED':40, 'VOTE':50, 'MIX':55}
        roles = {'SEER':0, 'WEREWOLF':1, 'POSSESSED':2, 'VILLAGER':3, 'HUMAN':0}
        wolf_api = {'pad':0, 'unk':1, 'sos':2, 'eos':3}
        MIX_API = {'Skip':0, 'Over':1, 'unk':4}
        off_set = src_num*60-40
        if mix_api is not None:
            tgt_num = MIX_API[mix_api]

        if src_num == 0:
            if thing == 'role_info':
                return roles[role] + 5
            if thing in wolf_api:
                return wolf_api[thing]
        if 0 < src_num <= 5:
            if thing in things:
                return off_set + things[thing] + roles[role]*5 + tgt_num
        raise 'undifind'


    def word2wolf_vec(self, word):
        if word[0] != '1':
            return
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
        # if 'status' in word[1]:
        #     if 'DEAD' in word[4]:
        #         self.wolf_vec.append(self.emit_id_112('DEAD', 1, int(word[2])))

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
    

def encode_padding_id_dir(id_list, pad_num, padding_len=8):
    # zero padding
    if len(id_list) < padding_len:
        while len(id_list) < padding_len:
            id_list.append(pad_num)
    elif padding_len < len(id_list):
        id_list = id_list[:padding_len]
    return id_list

def load_csv(relative_path):
    with open(relative_path, 'r') as f:
        for l in f.read().splitlines():
            yield [w for w in l.split(',')]


def main():
    load_log = LoadLog()

    # print(load_log.emit_id_320(2, 'ESTIMATE', 'POSSESSED', 3))
    # onehot_y = np.array(load_log.emit_onehot_t())

    # print(onehot_y)
    # tgt = []
    # for line in onehot_y:
    #     for i, elm in enumerate(line):
    #         if elm == 1:
    #             tgt.append(str(i+1))
    # print(tgt)

    # with open('tgt_poss.csv', 'w') as f:
    #     f.write('\n'.join(tgt))


    # tgt_vill = load_log.emit_t()

    # with open('tgt_vill.csv', 'w') as f:
    #     writer = csv.writer(f, lineterminator='\n')
    #     writer.writerows(tgt_vill)

    # vill_line1 = []
    # for vill_line in load_csv('tgt_vill.csv'):
    #     vill_line1.append(vill_line[1])
    
    # with open('tgt_vill2.csv', 'w') as f:
    #     f.write('\n'.join(vill_line1))
    


    load_log.emit_wolf_vec()
    print(load_log.log_wolf_vec)

    with open('src_wolf_320.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(load_log.log_wolf_vec)

    
    # count_token = {}
    # with open('src_wolf_320.csv', 'r') as f:
    #     for line in f.read().splitlines():
    #         for elm in line.split(','):
    #             elm = int(elm)
    #             if elm in count_token:
    #                 count_token[elm] += 1
    #             else:
    #                 count_token[elm] = 1
    # [print(i) for i in sorted(count_token.items(), key=lambda x: x[0])]

    # print(len(count_token))

    # add_wolf_info = []
    # pad_src_wolf_add_wolf = []
    # for tgt_line, src_line in zip(load_csv('tgt_poss.csv'), load_csv('src_wolf_320.csv')):
    #     add_wolf_info = list(str(int(tgt_line[0])+4)) + src_line
    #     pad_src_wolf_add_wolf.append(encode_padding_id_dir(add_wolf_info, '0', padding_len=66))


    # add_wolf_info = []
    # pad_src_wolf_add_wolf = []
    # for tgt_seer, tgt_poss, tgt_vill1, tgt_vill2, src_line in zip(
    #     load_csv('tgt_seer.csv'), load_csv('tgt_poss.csv'),
    #     load_csv('tgt_vill1.csv'), load_csv('tgt_vill2.csv'),
    #     load_csv('src_wolf_320.csv')
    #     ):
    #     add_wolf_info = (
    #     # [str(int(tgt_seer[0])+4)]
    #     # [str(int(tgt_poss[0])+9)]
    #     [str(int(tgt_vill1[0])+14)]
    #     #  + [str(int(tgt_vill2[0])+14)] 
    #      + src_line
    #      )
    #     pad_src_wolf_add_wolf.append(encode_padding_id_dir(add_wolf_info, '0', padding_len=66))


    # pad_src_wolf = []
    # for line in load_csv('./src_wolf_320.csv'):
    #     pad_src_wolf.append(encode_padding_id_dir(line, '0', padding_len=65))
    #     print(line)

    # with open('src_wolf_320_pad_add_vill1.csv', 'w') as f:
    #     writer = csv.writer(f, lineterminator='\n')
    #     writer.writerows(pad_src_wolf_add_wolf)



    # print(len(tgt), len(load_log.log_wolf_vec))

    #############################

    # with open('log_wolf_vec_v2.bf', 'wb') as f:
    #     pickle.dump(load_log.log_wolf_vec, f)

    # convert_wolf_log = ConvertWolfLog()
    # convert_wolf_log.convert_certian_elm()

    # wolf_labels = np.array(convert_wolf_log.wolf_vec_certian_elm)
    # onehot_x = conv_onehot(wolf_labels)

    # print(onehot_x.shape, onehot_y.shape)

    # np.save('onehot_x_112.npy', onehot_x)
    # np.save('onehot_y_112.npy', onehot_y)

    # print(len(onehot_t), len(wolf_onehot))
if __name__ == '__main__':
    main()
