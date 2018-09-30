import os
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

    def create_log_path(self, direc_num, log_num):
        path_by_env = f'{direc_num:03}/{log_num:03}.log'
        return os.path.join(self.relative_path, path_by_env)

    def capture_log(self, log_path):
        with open(log_path, 'r') as log_file:
            lines = [l for l in log_file.read().splitlines()]

        for line in lines:
            word = [w for w in line.split(',')]
            self.find_something(word)
            # self.word2wolf_vec(word, self.people_num)
    

    # Edit this function!!
    def find_something(self, word):
        # if not 'talk' in word[1]:
        #     return
        if 'talk' in word[1]:
            if ('Skip' in word[5] or 'Over' in word[5]):
                return
        print(word)
        self.count_lines += 1


    def word2wolf_vec(self, word):
        if 'COMINGOUT' in word[5]:
            for i, co_role in enumerate(self.search_CO):
                if co_role in word[5]:
                    # append id
                    pass
    
    def emit_id(self, state, number, my_number=1):
        until_CO = len(self.search_CO)
        until_VOTE = until_CO + self.people_num
        until_DIVINE = until_VOTE + self.people_num
        total_id_num = (until_DIVINE + self.people_num) * (my_number - 1)

        if state == 'CO':
            return total_id_num + number
        if state == 'VOTE':
            return total_id_num + until_CO + number
        if state == 'DIVINE':
            return total_id_num + until_VOTE + number
        if state == 'DIVINE-1':
            return total_id_num + until_DIVINE + number


def main():
    load_log = LoadLog()
    load_log.see_all_file()


if __name__ == '__main__':
    main()
