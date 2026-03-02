#!/user/bin/env python3
# -*- coding : utf-8 -*-

import jieba
import jieba.posseg as pseg

#加入自定义词典
jieba.load_userdict('userdict.txt')

def get_content():
    content = "我来自浙江省舟山市，我的名字叫黄丹阳，我的儿子是王绍嘉"
    resultone = jieba.cut(content, cut_all= False)
    #print(f'resultone: {resultone}')

    for word in resultone:
        yield word

#get_content()
for w in get_content():
    print(w)

def get_content2():
    content = "我来自浙江省舟山市，我的名字叫黄丹阳，我的儿子是王绍嘉"
    result_2 = jieba.lcut(content, cut_all=True)
    print(result_2)

get_content2()

def get_content3():
    content = "我来自浙江省舟山市，我的名字叫黄丹阳，我的儿子是王绍嘉"
    result_3 = jieba.lcut_for_search(content)
    print(result_3)

get_content3()

print(pseg.lcut('我爱北京天安门'))