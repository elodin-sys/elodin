/*
 ISC License

 Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

 Permission to use, copy, modify, and/or distribute this software for any
 purpose with or without fee is hereby granted, provided that the above
 copyright notice and this permission notice appear in all copies.

 THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

 */

%module swig_c_wrap
%{
    #include "architecture/_GeneralModuleFiles/sys_model.h"
    #include <architecture/utilities/bskLogging.h>
    #include <memory>
    #include <type_traits>
%}
%include <std_string.i>

%include "sys_model.i"

%inline %{

template <typename TConfig,
          void (*updateStateFun)(TConfig*, uint64_t, int64_t),
          void (*selfInitFun)(TConfig*, int64_t),
          void (*resetFun)(TConfig*, uint64_t, int64_t)
          >
class CWrapper : public SysModel {
    static_assert(std::is_default_constructible_v<TConfig>,
                  "The wrapped config must be default constructible (all 'Config' struct used in C "
                  "should be).");

  public:
    CWrapper() : config{std::make_unique<TConfig>()} {};
    CWrapper(TConfig* config) : config{config} {}; // We take ownership of config

    void SelfInit(){
        selfInitFun(this->config.get(), this->moduleID);};

    void UpdateState(uint64_t currentSimNanos){
        updateStateFun(this->config.get(), currentSimNanos, this->moduleID);};

    void Reset(uint64_t currentSimNanos){
        resetFun(this->config.get(), currentSimNanos, this->moduleID);};

    // Allows accesing the elements of the config from the wrapper in Python
    // Similar to how the smart pointers are implemented in SWIG
    TConfig* operator->() const { return this->config.get(); }

    TConfig& getConfig() { return *this->config.get(); }

  private:
    std::unique_ptr<TConfig> config; //!< class variable
};

%}

%define %c_wrap_3(moduleName, configName, functionSuffix)
    // This macro expects the header file to be "[moduleName].h"
    // the 'Config' structure to be called [configName],
    // and the implementation functions to follow the pattern:
    //    Update_[functionSuffix]
    //    SelfInit_[functionSuffix]
    //    Reset_[functionSuffix]

    %ignore Update_ ## functionSuffix;
    %ignore SelfInit_ ## functionSuffix;
    %ignore Reset_ ## functionSuffix;

    /*
    We define the Reset method for the given moduleName as empty.
    We make this method templated, so that it has lower priority
    in overload resolution than methods using the explicit type.

    This means that:
    
    Reset_hillPoint(hillPointConfig*, uint64_t, int64_t) { ... }

    will always be chosen before:

    template <typename T> Reset_hillPoint(T, uint64_t, int64_t) {}

    which effectively means that the empty method will only be used if
    users do not provide their own Reset method.
    */
    %inline %{
      template <typename T> inline void Reset_ ## functionSuffix(T, uint64_t, int64_t) {}
    %}

    // The constructor CWrapper(TConfig* config) takes ownership of the given pointer
    // We don't want the Python object for this config to also think it owns the memory
    %pythonappend CWrapper::CWrapper %{
        if (len(args)) > 0:
            args[0].thisown = False
    %}
    
    %include "moduleName.h"

    %template(moduleName) CWrapper<configName,Update_ ## functionSuffix,SelfInit_ ## functionSuffix,Reset_ ## functionSuffix>;

    %extend configName {
      %pythoncode %{
        def createWrapper(self):
            return moduleName(self)
      %}
    }

%enddef

%define %c_wrap_2(moduleName, configName)
    // This macro expects the header file to be "[moduleName].h"
    // the 'Config' structure to be called [configName],
    // and the implementation functions to follow the pattern:
    //    Update_[moduleName]
    //    SelfInit_[moduleName]
    //    Reset_[moduleName]
    %c_wrap_3(moduleName, configName, moduleName)
%enddef

%define %c_wrap(moduleName)
    // This macro expects the header file to be "[moduleName].h"
    // the 'Config' structure to be called [moduleName]Config,
    // and the implementation functions to follow the pattern:
    //    Update_[moduleName]
    //    SelfInit_[moduleName]
    //    Reset_[moduleName]

    %c_wrap_2(moduleName, moduleName ## Config)
%enddef
