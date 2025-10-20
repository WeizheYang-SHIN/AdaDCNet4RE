# coding: utf-8
# _*_ coding: utf-8 _*_
# @Time : 2023/2/16 16:45 
# @Author : wz.yang 
# @File : sort_reference.py
# @desc :
n = 0
with open("reference.txt", 'r', encoding='utf-8') as r:
    lines = r.readlines()
    for line in lines:
        result = ''
        if '``' in line and '.,' in line:
            names = line[0:line.index('``')].strip().split(". and ")
            raw_names = line[0:line.index('``')].strip()
            # print("修改前：", raw_names)
            title = line[line.index('``'):].strip()
            n += 1
            last = names[-1].split(',')[1].strip() + ' ' + names[-1].split(',')[0] + ','
            if len(names) == 2:
                pri_names = names[0].split('.,')
                for i in pri_names:
                    i = i.strip().split(',')
                    name = i[1].strip() + '. ' + i[0].strip() + ', '
                    result += name
                result = result + 'and ' +  last
            else:
                result = last
            # print("原  句：",line.strip())
            # print("标题：",title)
            print(result + ' ' + title)
        else:
            print(line.strip())


print(n,'条参考文献')