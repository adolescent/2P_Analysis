# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 13:26:30 2018

@author: LvLab_ZR
This part include all function defined that will be used in 2p analysis
通用函数，在这里作为一个模块方便调用
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
#%% 第四个功能是生成高斯矩阵的掩模，用来进行滤波模糊运算。
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
#%% 第六个功能是计算一帧上,这个细胞面积的亮度加和。
def sum_a_frame(frame,cell_index):#输入这一帧和cell_group[i]
    cell_location = cell_index.coords#
    x_list = cell_location[:,1] #这个细胞的全部X坐标
    y_list = cell_location[:,0] #这个细胞的全部Y坐标
    frame_sum = 0
    for i in range(0,len(x_list)):
        frame_sum = frame_sum +frame[y_list[i],x_list[i]]
    return frame_sum
#%% 第七个功能是把特定细胞的位置坐标得到，并以列表形式返回。返回值先x后y
def cell_location(cell_index):#输入cell_group[i]
    cell_location = cell_index.coords#
    x_list = cell_location[:,1] #这个细胞的全部X坐标
    y_list = cell_location[:,0] #这个细胞的全部Y坐标
    return x_list,y_list
#%% 第八个功能来自https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/，是scipy聚类画图的美化。
def fancy_dendrogram(*args, **kwargs):
    import scipy.cluster.hierarchy as clus_h
    import matplotlib.pyplot as plt
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = clus_h.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata
#%%
#第九个功能，用pickle保存文件。
import pickle
def save_variable(variable,name):
    fw = open(name,'wb')
    pickle.dump(variable,fw)#保存细胞连通性质的变量。 
    fw.close()
#%%    
#功能10-读取变量。    
def read_variable(name):#读取变量用的题头，希望这个可以在后续删掉
    with open(name, 'rb') as file:
        variable = pickle.load(file)
    file.close()
    return variable