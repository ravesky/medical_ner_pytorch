# 返回实体内容，实体起始位置和实体内容，实体的标签
def format_result(result, text, tag): 
    entities = [] 
    for i in result:
        if len(i) == 1:
            begin = end = i[0]
        else:
            begin,end = i
        entities.append({ 
            "start":begin, 
            "stop":end,
            "word":text[begin] if len(i) == 1 else text[begin:end+1],
            "type":tag
        }) 
    return entities

# 返回指定实体（比如tag=ORG）中在句子中索引位置，[0,2]表示第一个到第三个是一个实体的位置
def get_tags(path, tag, tag_map):
    begin_tag = tag_map.get("B-" + tag)
    mid_tag = tag_map.get("I-" + tag)
    end_tag = tag_map.get("E-" + tag)
    single_tag = tag_map.get("S")
    o_tag = tag_map.get("O")
    begin = -1
    end = 0
    tags = []
    last_tag = 0
    onlyOne = False # 用来判断是否是单个字符
    for index, tag in enumerate(path):
        # if tag == begin_tag and index == 0:
        #     begin = 0
        if tag == begin_tag and onlyOne == False:
            begin = index
            onlyOne = True
        elif tag == begin_tag and onlyOne == True:
            tags.append([begin])
            begin = index
            onlyOne = False
        elif tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index
            tags.append([begin, end])
            onlyOne = False
        elif tag == o_tag and onlyOne == True:
            tags.append([begin])
            onlyOne = False
        elif tag == o_tag and onlyOne == False:
            begin = -1
        last_tag = tag
    return tags

# 计算每个batch中实体的召回率和准确率和F1
def f1_score(tar_path, pre_path, tag, tag_map):
    origin = 0.
    found = 0.
    right = 0.
    for fetch in zip(tar_path, pre_path):
        # tar原始句子索引 pre预测的句子索引
        tar, pre = fetch
        # 原始的句子的实体索引，get_tags()方法：如果这个句子存在实体，就把实体的下标位置标记出来
        tar_tags = get_tags(tar, tag, tag_map)
        # 预测的句子的实体索引
        pre_tags = get_tags(pre, tag, tag_map)

        origin += len(tar_tags) # 原来的实体数
        found += len(pre_tags) # 识别的实体数，正不正确未知

        for p_tag in pre_tags:
            if p_tag in tar_tags:
                right += 1 # 正确数

    # recall = 0. if origin == 0 else (right / origin)
    # precision = 0. if found == 0 else (right / found)
    # f1 = 0. if recall+precision == 0 else (2*precision*recall)/(precision + recall)
    # print("\t{}\torgin:{}\tfound:{}\tright:{}".format(tag, origin, found, right))
    # print("\t\t\trecall:{:.6f}\tprecision:{:.6f}\tf1:{:.6f}".format(recall, precision, f1))
    # return origin, found, right, recall, precision, f1
    return origin, found, right