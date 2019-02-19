# -*- coding: utf-8 -*-'
'''
Created on 2018年3月8日
插值尝试
@author: 杨帆
'''
import numpy as np
import os
from PIL import Image
from random import shuffle
# dir="/media/fan/H/imgFeature_center16"

def generateNpy(dir):
    result_arr=[]
    label_list=[]
    map={}
    map_file_result={}
    map_file_label={}
    map_new={}
    ret_arr=[]
    count_label=0
    count=0
    vowels = ['a1','a2','a3','a4','o1','o2','o3','o4','e1','e2','e3','e4','i1','i2','i3','i4','u1','u2','u3','u4']
#     vowels = ['a1','a3','o1','o3','e1','e3','i1','i3','u1','u3']
#     vowels = ['a1','a2','a3','a4','ta1','ta2','ta3','ta4',
#               'o1','o2','o3','o4','fo1','fo2','fo3','fo4',
#               'e1','e2','e3','e4','te1','te2','te3','te4',
#               'i1','i2','i3','i4','ti1','ti2','ti3','ti4',
#               'u1','u2','u3','u4','tu1','tu2','tu3','tu4']
    file_list = []
    for vowel in vowels:
        files= os.listdir(dir + '/' + vowel)
        for file in files:
            file_list.append(dir + '/' + vowel + '/' + file)

    for file in file_list:
        file_path=file
        vowel = file.split("/")[-2]
        label = vowel
#         label = vowel[0:len(vowel)-1]
#         label=file.split("/")[-2][0]
        map[file]=label
        if label not in label_list:
            print '+',label,count_label
            label_list.append(label)
            map_new[label]=count_label
            count_label=count_label+1
        ##csv数据
#         data = np.loadtxt(file_path,delimiter = ',')
#         result= data.reshape((24,24,1))
        ##end
        ##图像
        img=Image.open(file_path)
        result=np.array([])
        r,g,b=img.split()

        r_arr=np.array(r).reshape(32*32*16)
        g_arr=np.array(g).reshape(32*32*16)
        b_arr=np.array(b).reshape(32*32*16)                                                             
        img_arr=np.concatenate((r_arr,g_arr,b_arr))
        result=np.concatenate((result,img_arr))
        #result=result.reshape((3,112,112))
        result=result.reshape((32*4,32*4,3))
        ##end
        map_file_result[file]=result
        result_arr.append(result)
        count=count+1
    for file in file_list:
        map_file_label[file]=map_new[map[file]]
    each_list=[]
    for file in file_list:
        each_list ={}
        label_one_zero=np.zeros(count_label)
        result=map_file_result[file]
        label=map_file_label[file]
        label_one_zero[label]=1.0
    
        each_list={'feature':result,'label':label_one_zero}
        ret_arr.append(each_list)
        shuffle(ret_arr)
    np.save('/media/fan/H1/npy-Feature/aoeiu1234_train.npy', ret_arr)
    return ret_arr

from random import shuffle
import shutil
def trainTestDivision(path):
    desTrainPath = 'E:\\depthFeature\\trainPointFeature'
    desTestPath = 'E:\\depthFeature\\testPointFeature'
    files = os.listdir(path)
    files = np.array(files)
    start = 0
    end = 0
    delenum = 0
    i = 0
    while end < len(files):
        while end<len(files) and files[start].split('-')[0] == files[end].split('-')[0]:
            end = end + 1
#         print start,end
        arrayList = range(start,end)
        if files[start].split('-')[0]!='a' and files[start].split('-')[0] !='ji' and files[start].split('-')[0] !='cong':
            print files[start].split('-')[0]
            i = i +1
            delenum = delenum + len(arrayList)
            start = end
            continue
        shuffle(arrayList)
        trainList = files[arrayList[0:len(arrayList)*8/10]]
        testList = files[arrayList[len(arrayList)*8/10:len(arrayList)]]
        for trainItem in trainList:
            shutil.copyfile(path + '\\' + trainItem ,desTrainPath + '\\' + trainItem)
        for testItem in testList:
            shutil.copyfile(path + '\\' + testItem ,desTestPath + '\\' + testItem)
        start = end
    print i,delenum
    
if __name__=="__main__":
    ret_arr=generateNpy('/media/fan/H1/imgFeatureTrain')
#     trainTestDivision('E:\\depthFeature\\pointFeature')
#     betch1 = np.load('/media/fan/H/npy-Feature/aeiu.npy')
#     print len(betch1)
#     betch2 = np.load('E:\\depthFeature\\test_point_feature.npy')
#     print betch1[0]['label']
#     print len(betch1[0]['label'])
#     print betch1[0]['csv']
#     print len(betch1[0]['csv'])
#     print len(betch2[0]['label'])

    