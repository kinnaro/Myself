# -*- encoding: utf-8 -*-
#-------------------------------------------------------------------------------
# Purpose:     每一个采样点
# Author:      ShowS
# Created:     2018-03-08
# update:      2018-03-08
#-------------------------------------------------------------------------------

import pandas as pd


def dealSimplePoint(simplePoint):
    simplePoint = simplePoint.drop(['Elapsed', 'MLII'], axis=1)
    #simplePoint = simplePoint.drop(['Elapsed', 'V5'], axis=1)
    simplePoint = simplePoint.drop([0])
    finalPoint = simplePoint['time']
    finalPoint.to_csv("S:\\MIT相关文件\\MIT数据采样点\\SamplePoint\\FinalSample\\" + singlename + ".csv", index=False, sep=',')

def dealSimplePoint2(simplePoint):
    #simplePoint = simplePoint.drop(['Elapsed', 'MLII'], axis=1)
    simplePoint = simplePoint.drop(['Elapsed', 'V5'], axis=1)
    simplePoint = simplePoint.drop([0])
    finalPoint = simplePoint['time']
    finalPoint.to_csv("S:\\MIT相关文件\\MIT数据采样点\\SamplePoint\\FinalSample\\" + singlename + ".csv", index=False, sep=',')    

if __name__ == "__main__":
    Indexs = ['100','108','113','117','122','201','207','212','217','222','231','101','105','109','114','118','123','202','208','213','219','223','232','106','111','115','119','124','203','209','214','220','228','233','103','107','112','116','121','200','205','210','215','221','230','234']
    Indexs2 = ['102','104']
    for index in range(0,46):
        singlename = Indexs[index]
        simplePoint = pd.read_excel("S:\\MIT相关文件\\MIT数据采样点\\SamplePoint\\Excel\\" + singlename + ".xls")
        simplePoint = dealSimplePoint(simplePoint)
    for index in range(0,2):
        singlename = Indexs2[index]
        simplePoint = pd.read_excel("S:\\MIT相关文件\\MIT数据采样点\\SamplePoint\\Excel\\" + singlename + ".xls")
        simplePoint = dealSimplePoint2(simplePoint)
        
        
        