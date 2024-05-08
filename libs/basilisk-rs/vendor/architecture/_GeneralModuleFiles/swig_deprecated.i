/*
 ISC License

 Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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
%module swig_deprecated

/** Used to deprecate a function in C++ that is exposed to Python through SWIG.

'function' is the SWIG identifier of the function. If it is a standalone function,
then this is simply its name. If it's a class function, then this should be
[CLASS_NAME]::[FUNCTION_NAME]

'removal_date' is the expected removal date in the format 'YYYY/MM/DD'. Think of
an amount of time that would let users update their code, and then add that duration
to today's date to find a reasonable removal date.

'message' is a text that is directly shown to the users. Here, you may explain
why the function is deprecated, alternative functions, links to documentation
or scenarios that show how to translate deprecated code...

See src\architecture\utilitiesSelfCheck\swigDeprecatedCheck.i
*/
%define %deprecated_function(function, removal_date, message)
%pythonprepend function %{
    from Basilisk.utilities import deprecated
    deprecated.deprecationWarn(f"{__name__}.function".replace("::","."), `removal_date`, `message`)
%}
%enddef

/** Used to deprecate a public class variable in C++ that is exposed to Python through SWIG.

'class' is the SWIG identifier of the class.

'variable' is the name of the variable.

'removal_date' is the expected removal date in the format 'YYYY/MM/DD'. Think of
an amount of time that would let users update their code, and then add that duration
to today's date to find a reasonable removal date.

'message' is a text that is directly shown to the users. Here, you may explain
why the variable is deprecated, alternative variables, links to documentation
or scenarios that show how to translate deprecated code...

See src\architecture\utilitiesSelfCheck\swigDeprecatedCheck.i
*/
%define %deprecated_variable(class, variable, removal_date, message)
%extend class {
    %pythoncode %{
    from Basilisk.utilities import deprecated
    variable = deprecated.DeprecatedProperty(`removal_date`, `message`, variable)
    %}
}
%enddef
