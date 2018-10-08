# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 13:26:30 2018

@author: LvLab_ZR
This part include all function defined that will be used in 2p analysis
"""
#%%首先是遍历文件夹的tif文件
import os
 
def tif_name(file_dir): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if root == file_dir:#只遍历根目录，不操作子目录的文件
                if os.path.splitext(file)[1] == '.tif':
                    L.append(os.path.join(root, file))
    return L
#test = tif_name(r'F:\datatemp\180508_L14\Run02_spon\1-002')
#%%第二个功能是创建目录，如果目录已存在就不创建，并返回
def mkdir(path):
    # 引入模块
    import os
    isExists=os.path.exists(path)
    if isExists:
        # 如果目录存在则不创建，并提示目录已存在
        print('Folder',path,'already exists!')
        return False
    else:
        os.mkdir(path)
        return True
#%%第三个功能是对齐，输入当前图像和平均帧，对齐之后计算出与平均帧偏移的X和Y量
        #原来的写法不太直观，于是改成另一种标准的写法，结果一样。
def bias_correlation(temp_tif,averaged_tif):
    import numpy as np
    temp_tif = np.float64(temp_tif)
    averaged_tif = np.float64(averaged_tif)#转数据类型
    target = temp_tif[100:412,100:412] #对齐图片留了100像素的边框,大小312*312
    tample = averaged_tif[60:452,60:452]#模板图片，保留了60像素边框，大小392*392
    target_pad = np.pad(np.rot90(target,2),((0,391),(0,391)),'constant')
    tample_pad = np.pad(tample,((0,311),(0,311)),'constant')#把两个矩阵分别扩充为312+392-1维
#    target_fft = np.fft.fft2(np.rot90(target,2),[400,400])
#    tample_fft = np.fft.fft2(tample,[400,400])
    target_fft = np.fft.fft2(target_pad)
    tample_fft = np.fft.fft2(tample_pad)
    conv2 = np.real(np.fft.ifft2(target_fft*tample_fft))#np里*是点乘,用这种方法可以得到卷积，效率更高
    find_location = conv2[331:371,331:371]#由于对齐中心应该是(351，351),这里选取它上下20像素。
    y_bias = np.where(find_location ==np.max(find_location))[0][0] -20# 得到偏移的y量。
    x_bias = np.where(find_location ==np.max(find_location))[1][0] -20# 得到偏移的x量。
    return[x_bias,y_bias]
#%% 第四个功能是生成二维维高斯函数掩膜，用于进行高斯模糊用来得到细胞图。
def normalized_gauss2D(shape,sigma): #e.g. shape = [7,7],sigma = 1.5
    import numpy as np
    m,n = [(ss-1.)/2. for ss in shape] #得到横纵方向的半高宽
    y,x = np.ogrid[-m:m+1,-n:n+1]    #左闭右开，得到每一个取值。
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0 #小于精度的置为零。
    sumh = h.sum() # 归一化高斯掩模，使整个模的和为1
    if sumh != 0:
        h /= sumh
    return h
#%% 第五个功能是将细胞编号画出来，把连通区域编号画在区域中心上。
def show_cell(base_graph_path,cell_group):
    from PIL import ImageFont
    from PIL import Image
    from PIL import ImageDraw
    font = ImageFont.truetype('arial.ttf',11)
    im = Image.open(base_graph_path)
    for N in range(0,len(cell_group)):
        y,x = cell_group[N].centroid
        draw = ImageDraw.Draw(im)
        draw.text((x*2,y*2),str(N),(0,255,100),font = font,align = 'center')#图像放大一倍
    save_path = base_graph_path[0:(len(base_graph_path)-4)]+'_Labeled.tif'
    im.save(save_path)
#%% 第六个功能是归一化，把输入向量归一化成为0-1的数组。
def normalize_vector(vector):
    import numpy as np
    max_num = np.max(vector)
    min_num = np.min(vector)
    vector = (vector-min_num)/(max_num-min_num)
    return vector
#%% 第七个功能是计算一帧上,这个细胞面积的亮度加和。
def sum_a_frame(frame,cell_index):#输入这一帧和cell_group[i]
    cell_location = cell_index.coords#
    x_list = cell_location[:,1] #这个细胞的全部X坐标
    y_list = cell_location[:,0] #这个细胞的全部Y坐标
    frame_sum = 0
    for i in range(0,len(x_list)):
        frame_sum = frame_sum +frame[y_list[i],x_list[i]]
    return frame_sum
#%% 第八个功能是把特定细胞的位置坐标得到，并以列表形式返回。返回值先x后y
def cell_location(cell_index):#输入cell_group[i]
    cell_location = cell_index.coords#
    x_list = cell_location[:,1] #这个细胞的全部X坐标
    y_list = cell_location[:,0] #这个细胞的全部Y坐标
    return x_list,y_list