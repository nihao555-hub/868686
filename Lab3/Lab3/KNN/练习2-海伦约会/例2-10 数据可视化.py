# -*- coding: utf-8 -*-

from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def showdatas(datingDataMat, datingLabels):
    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False,
                            sharey=False, figsize=(13,8))
    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('red')
        if i == 2:
            LabelsColors.append('green')
        if i == 3:
            LabelsColors.append('blue')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、
    # 第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1],
                      color=LabelsColors,s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(
        u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',
        FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',
                                            FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占',
                                            FontProperties=font)
    plt.setp(axs0_title_text, size=10, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=10, weight='bold', color='black')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、
    # 第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2],
                      color=LabelsColors,s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(
        u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',
                                            FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',
                                            FontProperties=font)
    plt.setp(axs1_title_text, size=10, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=10, weight='bold', color='black')
    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、
    # 第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2],
                      color=LabelsColors,s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(
        u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(
        u'玩视频游戏所消耗时间占比',FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(
        u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs2_title_text, size=10, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=10, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='green', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='blue', marker='.',
                      markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    # 调整子图间距
    plt.subplots_adjust(wspace =0.2, hspace =0.4)
    # 显示图片
    plt.show()

showdatas(datingDataMat, datingLabels)