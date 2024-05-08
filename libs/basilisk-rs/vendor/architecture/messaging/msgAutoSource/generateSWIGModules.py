import sys

if __name__ == "__main__":
     moduleOutputPath = sys.argv[1]
     headerinputPath = sys.argv[2]
     structType = sys.argv[3].split('Payload')[0]
     baseDir = sys.argv[4]
     generateCInfo = sys.argv[5] == 'True'

     swigTemplateFile = 'msgInterfacePy.i.in'
     swigCTemplateFile = 'cMsgCInterfacePy.i.in'     

     swigFid = open(swigTemplateFile, 'r')
     swigTemplateData = swigFid.read()
     swigFid.close()

     swigCFid = open(swigCTemplateFile, 'r')
     swigCTemplateData = swigCFid.read()
     swigCFid.close()

     moduleFileOut = open(moduleOutputPath, 'w')
     moduleFileOut.write(swigTemplateData.format(type=structType, baseDir=baseDir))
     if(generateCInfo):
         moduleFileOut.write(swigCTemplateData.format(type=structType))
     moduleFileOut.close()


