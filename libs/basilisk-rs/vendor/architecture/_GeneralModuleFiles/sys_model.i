
%module cSysModel
%{
   #include "sys_model.h"
%}

%pythoncode %{
from Basilisk.architecture.swig_common_model import *
%}
%include "std_string.i"
%include "swig_conly_data.i"
%include "architecture/utilities/bskLogging.h"

%include "sys_model.h"

%pythonbegin %{
from typing import Union, Iterable
%}

%extend SysModel
{
    %pythoncode %{
        def logger(self, variableNames: Union[str, Iterable[str]], recordingTime: int = 0):
            if isinstance(variableNames, str):
                variableNames = [variableNames]

            logging_functions = {
                variable_name: lambda _, vn=variable_name: getattr(self, vn)
                for variable_name in variableNames
            }

            for variable_name, log_fun in logging_functions.items():
                try:
                    log_fun(0)
                except AttributeError:
                    raise ValueError(f"Cannot log {variable_name} as it is not a "
                                    f"variable of {type(self).__name__}")

            from Basilisk.utilities import pythonVariableLogger
            return pythonVariableLogger.PythonVariableLogger(logging_functions, recordingTime)
    %}
}
