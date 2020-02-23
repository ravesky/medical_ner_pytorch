import os
from collections import Counter

class TransferData:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        print(cur)
        self.label_dict = {
                      '检查和检验': 'CHECK',
                      '症状和体征': 'SIGNS',
                      '疾病和诊断': 'DISEASE',
                      '治疗': 'TREATMENT',
                      '身体部位': 'BODY'
        }
        self.origin_path = os.path.join(cur, 'data_origin')
        self.train_filepath = os.path.join(cur, 'medical.train')
        return


    def transfer(self):
        f = open(self.train_filepath, 'w+')
        count = 0
        for root,dirs,files in os.walk(self.origin_path):
            print('root',root)
            print('dirs',dirs)
            print('files',files)
            for file in files:
                filepath = os.path.join(root, file)
                if 'original' not in filepath:
                    continue
                label_filepath = filepath.replace('.txtoriginal','')
                print(filepath, '\t\t', label_filepath)
                content = open(filepath,encoding='utf-8').read().strip()
                print('content',content)
                res_dict = {}
                for line in open(label_filepath,encoding='utf-8'):
                    res = line.strip().split('	')
                    start = int(res[1])
                    end = int(res[2])
                    label = res[3]
                    label_id = self.label_dict.get(label)
                    for i in range(start, end+1):
                        if i == start:
                            # label_cate = label_id + '-B'
                            label_cate = 'B-' + label_id
                        elif i == end:
                            label_cate = 'E-' + label_id
                        else:
                            # label_cate = label_id + '-I'
                            label_cate = 'I-' + label_id
                        res_dict[i] = label_cate

                for indx, char in enumerate(content):
                    char_label = res_dict.get(indx, 'O')
                    print(char, char_label)
                    f.write(char + '\t' + char_label + '\n')
                f.write('end\n')
        f.close()
        return



if __name__ == '__main__':
    handler = TransferData()
    train_datas = handler.transfer()
    print(train_datas)
