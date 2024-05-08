import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path + '/../../../../../Basilisk/src/architecture/messaging/msgAutoSource')

if __name__ == "__main__":
    moduleOutputPath = sys.argv[1]
    isExist = os.path.exists(moduleOutputPath)
    if not isExist:
        os.makedirs(moduleOutputPath, exist_ok=True)
    mainImportFid = open(moduleOutputPath + '/__init__.py', 'w')
    for i in range(2, len(sys.argv)):
        headerInputPath = sys.argv[i]
        for filePre in os.listdir(headerInputPath):
            if(filePre.endswith(".h") or filePre.endswith(".hpp")):
                className = os.path.splitext(filePre)[0]
                msgName = className.split('Payload')[0]
                mainImportFid.write('from Basilisk.architecture.messaging.' + className + ' import *\n')
    mainImportFid.close()
    setOldPath = moduleOutputPath.split('messaging')[0] + '/cMsgCInterfacePy'
    os.symlink(moduleOutputPath, setOldPath)
