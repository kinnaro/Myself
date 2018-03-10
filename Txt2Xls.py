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
    xls.save("S:\\MIT相关文件\\MIT数据采样点\\SamplePoint\Excel\\" + xlsname+'.xls') #保存

'''
def xls_to_csv_pd():
    data_xls = pd.read_excel(xlsname + '.xls', index_col=0)
    data_xls.to_csv(xlsname + '.csv', encoding='utf-8')
'''


'''
if __name__ == "__main__":
    Indexs = ['100','104','108','113','117','122','201','207','212','217','222','231','101','105','109','114','118','123','202','208','213','219','223','232','102','106','111','115','119','124','203','209','214','220','228','233','103','107','112','116','121','200','205','210','215','221','230','234']
    for index in range(0,48):
        singlename = Indexs[index]
        filename = "S:\\MIT相关文件\\MIT数据采样点\\SamplePoint\Txt\\" + singlename + ".txt"
        filename = "S:\\MIT相关文件\\MIT数据采样点\\SamplePoint\\Txt\\100_AtrResult.txt"
        xlsname  = singlename 
        txt2xls(filename, xlsname)
        #xls_to_csv_pd()
'''

if __name__ == "__main__":
    singlename = "100"
    filename = "S:\\MIT相关文件\\MIT数据采样点\\SamplePoint\\Txt\\100_AtrResult.txt"
    xlsname  = singlename 
    txt2xls(filename, xlsname)
