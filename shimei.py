# -*- encoding: utf-8 -*-
#-------------------------------------------------------------------------------
# Purpose:     txt转换成Excel
# Author:      ShowS
# Created:     2018-03-03
# update:      2018-03-03
#-------------------------------------------------------------------------------

import xlwt #需要的模块
import pandas as pd

def txt2xls(filename,xlsname):  #文本转换成xls的函数，filename 表示一个要被转换的txt文本，xlsname 表示转换后的文件名
    print("converting " + xlsname + " xls ...")
    f = open(filename)   #打开txt文本进行读取
    x = 0                #在excel开始写的位置（y）
    y = 0                #在excel开始写的位置（x）
    xls=xlwt.Workbook()
    sheet = xls.add_sheet('sheet1',cell_overwrite_ok=True) #生成excel的方法，声明excel
    while True:  #循环，读取文本里面的所有内容
        line = f.readline() #一行一行读取
        if not line:  #如果没有内容，则退出循环
            break
        for i in line.split():#读取出相应的内容写到x
            item=i.strip()
            sheet.write(x,y,item)
            y += 1 #另起一列
        x += 1 #另起一行
        y = 0  #初始成第一列
    f.close()
    xls.save(xlsname+'.xls') #保存

'''
def xls_to_csv_pd():
    data_xls = pd.read_excel(xlsname + '.xls', index_col=0)
    data_xls.to_csv(xlsname + '.csv', encoding='utf-8')
'''

if __name__ == "__main__":
    Indexs = ['annotations00735','annotations03665','annotations04015','annotations04043','annotations04048','annotations04126','annotations04746','annotations04908','annotations04936','annotations05091','annotations05121','annotations05261','annotations06426','annotations06453','annotations06995','annotations07162','annotations07859','annotations07879','annotations07910','annotations08215','annotations08219','annotations08378','annotations08405','annotations08434','annotations08455']
    for index in range(0,25):
        singlename = Indexs[index]
        filename = "S:\\MIT相关文件\\MIT数据采样点\\师妹\\" + singlename + ".txt"
        xlsname  = singlename
        txt2xls(filename, xlsname)
        #xls_to_csv_pd()