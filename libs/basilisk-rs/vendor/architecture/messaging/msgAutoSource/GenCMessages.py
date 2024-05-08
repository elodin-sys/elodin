# Generate C code messages that are compatible with cpp functor based system
import argparse
import errno
import os
import shutil
from sys import platform

import parse


class GenerateMessages:

    """
    A class to generate C and C++ messages using a defined template before the build.

    Attributes:
        pathToExternalModules: string
                path to add messages for external module (default empty string)

    """
    def __init__(self, pathToExternalModules):
        self.messageTemplate = ""
        self.headerTemplate = ""
        self.autoSourceDestDir = '../../../../dist3/autoSource/'
        self.destinationDir = os.path.join(self.autoSourceDestDir, 'cMsgCInterface/')
        self.pathToExternalModules = pathToExternalModules
        with open('./cMsgCInterfacePy.i.in', 'r') as f:
            self.swig_template_block = f.read()
        self.swigTemplate = ""
        self.messagingAutoData = list()


    def __createMessageAndHeaderTemplate(self):
        """
        A method which reads the license and README add create messageTemplate and headerTemplate
        """
        licenseREADME = list()
        with open("../../../../LICENSE", 'r') as f:
            licenseREADME.extend(["/*", f.read(),"*/\n\n"])
        with open('./README.in', 'r') as r:
            licenseREADME.append(r.read())
        self.messageTemplate = ''.join(licenseREADME)
        self.headerTemplate = ''.join(licenseREADME)
        with open('./msg_C.cpp.in', 'r') as f:
            self.messageTemplate += f.read()
        with open('./msg_C.h.in', 'r') as f:
            self.headerTemplate += f.read()

    def __recreateDestinationDirectory(self):
        """
        Method to delete the existing destination directory and recreate it.
        """
        if os.path.exists(self.autoSourceDestDir):
            shutil.rmtree(self.autoSourceDestDir, ignore_errors=True)
        try:
            os.makedirs(os.path.dirname(self.autoSourceDestDir))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
        print(self.destinationDir)
        os.makedirs(os.path.dirname(self.destinationDir))

    def __generateMessagingHeaderInterface(self):
        """
        Method to generate message header interface. It is empty for windows.
        """
        messaging_header_i_template = ""
        if platform == "linux" or platform == "linux2":
            messaging_header_i_template = "#define SWIGWORDSIZE64\n"
        with open(self.autoSourceDestDir + 'messaging.header.auto.i', 'w') as w:
            w.write(messaging_header_i_template)

    def __createMessageC(self,parentPath, external=False):
        """
        Method to add C messages to messaging.auto.i file which will be used to create swig interface for messaging
        """
        if external:
            messaging_i_template = ""
            relativePath = os.path.relpath(self.pathToExternalModules, "../../../architecture").replace("\\",
                                                                                                        "/")
        else:
            messaging_i_template = "//C messages:"
        for file in os.listdir(f"{parentPath}/msgPayloadDefC"):
            if file.endswith(".h"):
                msgName = (os.path.splitext(file)[0])[:-7]
                if external:
                    messaging_i_template += f"\nINSTANTIATE_TEMPLATES({msgName}, {msgName}Payload, {relativePath}/msgPayloadDefC)"
                else:
                    messaging_i_template += f"\nINSTANTIATE_TEMPLATES({msgName}, {msgName}Payload, msgPayloadDefC)"
        with open(self.autoSourceDestDir + 'messaging.auto.i', 'a') as w:
            w.write(messaging_i_template)

    def __createMessageCpp(self,parentPath, external=False):
        """
        Method to add Cpp messages to messaging.auto.i file which will be used to create swig interface for messaging
        """
        if external:
            messaging_i_template = ""
        else:
            messaging_i_template = "\n\n//C++ messages:"
        for file in os.listdir(f"{parentPath}/msgPayloadDefCpp"):
            if file.endswith(".h"):
                msgName = (os.path.splitext(file)[0])[:-7]
                if external:
                    relativePath = os.path.relpath(self.pathToExternalModules, "../../../architecture").replace("\\",
                                                                                                               "/")
                    messaging_i_template += f"\nINSTANTIATE_TEMPLATES({msgName}, {msgName}Payload, {relativePath}/msgPayloadDefCpp)"
                else:
                    messaging_i_template += f"\nINSTANTIATE_TEMPLATES({msgName}, {msgName}Payload, msgPayloadDefCpp)"
        with open(self.autoSourceDestDir + 'messaging.auto.i', 'a') as w:
            w.write(messaging_i_template)

    def __generateMessages(self):
        """
        Method which call create messages methods for c and Cpp messages for basilisk as well as external messages
        """
        # append all C msg definitions to the dist3/autoSource/messaging.auto.i file that is imported into messaging.auto.i
        self.__createMessageC("../..")
        if self.pathToExternalModules and os.path.exists(os.path.join(self.pathToExternalModules,"msgPayloadDefC")):
            self.__createMessageC(self.pathToExternalModules,True)

        with open(self.autoSourceDestDir + 'messaging.auto.i', 'r') as fb:
            self.messagingAutoData = fb.readlines()
        # The following cpp message definitions must be included after the `self.messagingAutoData` variable is set above.
        # We only need to create Python interfaces to C++ messages, not C wrappers.
        self.__createMessageCpp("../..")
        if self.pathToExternalModules and os.path.exists(
                os.path.join(self.pathToExternalModules, "msgPayloadDefCpp")):

            self.__createMessageCpp(self.pathToExternalModules, True)

    def __toMessage(self, structData):
        """
        Method to generate Cpp wrapper for C messages.
        """
        if structData:
            structData = structData.replace(' ', '').split(',')
            structName = structData[0]
            sourceHeaderFile = f"{structData[2]}/{structName}Payload.h"
            definitions = self.messageTemplate.format(type=structName)
            header = self.headerTemplate.format(type=structName, structHeader=sourceHeaderFile)
            self.swigTemplate.write(self.swig_template_block.format(type=structName))
            file_name = os.path.join(self.destinationDir, structName + '_C')
            definitionsFile = file_name + '.cpp'
            header_file = file_name + '.h'
            with open(definitionsFile, 'w') as w:
                w.write(definitions)
            with open(header_file, 'w') as w:
                w.write(header)

    def initialize(self):
        """
        Method to initialize class members.
        """
        self.__createMessageAndHeaderTemplate()
        self.__recreateDestinationDirectory()
        self.__generateMessagingHeaderInterface()

    def run(self):
        """
        Method to run the generation workflow
        """
        self.__generateMessages()
        # create swig file for C-msg C interface methods
        self.swigTemplate = open(self.autoSourceDestDir + 'cMsgCInterfacePy.auto.i', 'w')
        templateCall = 'INSTANTIATE_TEMPLATES({:dat})'
        for line in self.messagingAutoData:
            parse.parse(templateCall, line.strip(), dict(dat=self.__toMessage))
        self.swigTemplate.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure generated Messages")
    # define the optional arguments
    parser.add_argument("--pathToExternalModules", help="External Module path", default="")
    args = parser.parse_args()
    generateMessages = GenerateMessages(args.pathToExternalModules)
    generateMessages.initialize()
    generateMessages.run()