#!/usr/bin/python
# -*- conding:utf-8 -*-
import sys,getopt
from fcn_dropout_CE_GDO import mnistdataclassical_dropout_CE_GDO
from fcn_dropout_CE_AO import mnistdataclassical_dropout_CE_AO

def dnn_train_main(argv):
    selectNN='fcn_dropout_CE_GDO'
    ArgvDict={"mnistDataDir":"MNIST_data","batchSize":200, "learningRate":0.01, "epochSize":50, "dropout": 0.7}
    try:
        opts,args=getopt.getopt(argv,"hs:d:b:l:e:D:",["select=","DataDir=","batchSize=","learningRate=","epochSize=","dropout="])
    except getopt.GetoptError:
        print('python dnnrun.py -s <select> -d <datasetDir> -b <batch> -l <learningRate> -e <epoch> -D <Dropout>\n')
        print("selectable option: 1>'fcn_dropout_CE_GDO'.\n2>'fcn_dropout_CE_GDO'.\nDefault is fcn_dropout_CE_GDO\n")
        sys.exit(2)
    for opt,arg in opts:
        if opt =='-h':
            print('python dnnrun.py -s <select> -d <datasetDir> -b <batch> -l <learningRate> -e <epoch> -D <Dropout>\n')
            print("selectable option: 1>'fcn_dropout_CE_GDO'.\n2>'fcn_dropout_CE_GDO'.\nDefault is fcn_dropout_CE_GDO\n")
            sys.exit()
        elif opt in ("-s", "--select"):
            selectNN=arg
        elif opt in ("-d","--DataDir"):
            if arg!=ArgvDict["mnistDataDir"]:
                ArgvDict["mnistDataDir"]=arg
        elif opt in ("-b","--batchSize"):
            if arg!=ArgvDict["batchSize"]:
                ArgvDict["batchSize"]=int(arg)
        elif opt in ("-l","--learningRate"):
            if arg!=ArgvDict["learningRate"]:
                ArgvDict["learningRate"]=float(arg)
        elif opt in ("-e","--epochSize"):
            if arg!=ArgvDict["epochSize"]:
                ArgvDict["epochSize"]=int(arg)
        elif opt in ("-D","--dropout"):
            if arg!=ArgvDict["dropout"]:
                ArgvDict["dropout"]=float(arg)

    if selectNN=='fcn_dropout_CE_GDO':
        print(selectNN)
        mnistdataclassical_dropout_CE_GDO(**ArgvDict)
    elif selectNN=='fcn_dropout_CE_AO':
        print(selectNN)
        mnistdataclassical_dropout_CE_AO(**ArgvDict)
    else:
        print('invalid params\n')
        sys.exit()

if __name__=="__main__":
    dnn_train_main(sys.argv[1:])