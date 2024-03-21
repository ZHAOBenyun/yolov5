"""
使用python xml解析树解析xml文件，批量修改xml文件里object节点下name节点的text
"""


import glob
import xml.etree.ElementTree as ET
path = '/media/kemove/2T/datasets/SUT-Crack-detection/xml'   # xml文件夹路径
i = 0
for xml_file in glob.glob(path + '/*.xml'):
    # print(xml_file)
    tree = ET.parse(xml_file) 
    obj_list = tree.getroot().findall('object') 
    for per_obj in obj_list:
        if per_obj[0].text == 'Crack':    # 错误的标签“33”
            per_obj[0].text = 'crack'    # 修改成“44”
            i = i+1

    tree.write(xml_file)    # 将改好的文件重新写入，会覆盖原文件
print('共完成了{}处替换'.format(i))
