
%module(directors="1") sysModel
%{
   #include "sys_model.h"
%}

%pythoncode %{
import sys
import traceback
from Basilisk.architecture.swig_common_model import *
%}
%include "std_string.i"
%include "swig_conly_data.i"
%include "architecture/utilities/bskLogging.h"

%feature("director") SysModel;
%feature("pythonappend") SysModel::SysModel %{
    self.__super_init_called__ = True%}
%rename("_SysModel") SysModel;
%include "sys_model.i"

%pythoncode %{
class SuperInitChecker(type):

    def __call__(cls, *a, **kw):
        rv = super(SuperInitChecker, cls).__call__(*a, **kw)
        if not getattr(rv, "__super_init_called__", False):
            error_msg = (
               "Need to call parent __init__ in SysModel subclasses:\n"
               f"class {cls.__name__}(sysModel.SysModel):\n"
               "    def __init__(...):\n"
               "        super().__init__()"
            )
            raise SyntaxError(error_msg)
        return rv

def logError(func):
    """Decorator that prints any exceptions that happen when
    the original function is called, and then raises them again."""
    def inner(*arg, **kwargs):
        try:
            return func(*arg, **kwargs)
        except Exception:
            traceback.print_exc()
            raise
    return inner

class SysModel(_SysModel, metaclass=SuperInitChecker):
    bskLogger: BSKLogger = None

    def __init_subclass__(cls):
        # Make it so any exceptions in UpdateState and Reset
        # print any exceptions before returning control to
        # C++ (at which point exceptions will crash the program)
        cls.UpdateState = logError(cls.UpdateState)
        cls.Reset = logError(cls.Reset)
%}
