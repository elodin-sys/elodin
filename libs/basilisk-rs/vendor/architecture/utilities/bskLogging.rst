Executive Summary
-----------------
This support class ``bskLogger`` enables C++ ``bskLog()`` and ANSI-C ``_bskLog()`` method to be used to log various debug, information, warning and error messages. The verbosity, i.e. what level of messages are printed to the terminal, can be set from the Basilisk python script.

.. table:: Verbosity Level Options
        :widths: 25 25 100

        +-----------------------+---------------------------------+---------------------------------------------------+
        | Level                 | Description                                                                         |
        +=======================+=================================+===================================================+
        | BSK_DEBUG             | Can be used for debug information logging.  Such ``bskLog`` statement should not be |
        |                       | left in the final Basilisk code.                                                    |
        +-----------------------+---------------------------------+---------------------------------------------------+
        | BSK_INFORMATION       | General information messages                                                        |
        +-----------------------+---------------------------------+---------------------------------------------------+
        | BSK_WARNING           | Warnings about unexpected behavior, but not outright errors.                        |
        +-----------------------+---------------------------------+---------------------------------------------------+
        | BSK_ERROR             | Erroneous behavior that needs to be fixed.                                          |
        +-----------------------+---------------------------------+---------------------------------------------------+
        | BSK_SILENT            | This level is used to silence all `bskLog` statements.  This should never be used   |
        |                       | with the `bskLog` method within the C++ or C code.                                  |
        +-----------------------+---------------------------------+---------------------------------------------------+


Class Assumptions and Limitations
----------------------------------
The ``bskLogger`` class is intended to be used primarily within the BSK modules.  All the modules must have been initialized before the ``bskLog`` becomes effective.  Thus, if ``bskLog`` is used during initialization to print a warning, this will not function as expected.

For utility libraries such as ``linearAlgebra.c/h`` etc., this logging capabilities is not applicable as these libraries don't have access to the ``bskLogger`` instance.  Rather, in such cases use the ``BSK_PRINT()`` macro instead.



Using ``bskLogger`` From Python
-------------------------------
For an example of how to set the verbosity from Python, see :ref:`scenarioBskLog`.
The default verbosity is set to the lowest level ``BSK_DEBUG`` such that any ``bskLog`` method print out the associated message string.  If this is the desired behavior, then no further actions are required.

If the verbosity level is to be changed for a particular Basilisk script, then the following instructions explain how this can be done.  At the top of the Basilisk python scrip be sure to include the ``bskLogging`` support package::

    from Basilisk.architecture import bskLogging

Setting Verbosity Globally for all BSK Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``bskLog`` verbosity can be modified for all Basilisk modules by using::

    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

The verbosity options are listed in the table above.  Note that this command must be included at the very beginning of
the Basilisk simulation script, certainly before the call for ``SimulationBaseClass.SimBaseClass()``.

Changing Verbosity for a Particular BSK Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is possible to override the global verbosity setting and specify a different verbosity for a particular module.  Assume we want to have a unique verbosity level for the ``simpleNav`` module.  This can be done through::

    sNavObject = simpleNav.SimpleNav()
    scSim.AddModelToTask(simTaskName, sNavObject)
    logger = bskLogging.BSKLogger()
    logger.setLogLevel(bskLogging.BSK_INFORMATION)
    sNavObject.bskLogger = logger

Another option is to use the ``BSKLogger()`` constructor to provide the verbosity directly through::

    sNavObject = simpleNav.SimpleNav()
    sNavObject.bskLogger = bskLogging.BSKLogger(bskLogging.BSK_INFORMATION)

Unlike change the global verbosity level, the module specific verbosity can be changed later on in the Basilisk
python script as the corresponding module is created and configured.

Using ``bskLog`` in C++ Basilisk Modules
----------------------------------------
The first step is to include the ``bskLogging`` support file with the module `*.h` file using:

.. code-block:: cpp

    #include "architecture/utilities/bskLogging.h"

Next, the module class must contain the following public variable:

.. code-block:: cpp

    BSKLogger bskLogger;

Within the ``*.cpp`` file, the ``bskLog()`` method can be called with:

.. code-block:: cpp

    bskLogger.bskLog(BSK_INFORMATION, "%d %d", arg1, arg2);


Using ``_bskLog`` in C Basilisk Modules
---------------------------------------
The first step is to include the ``bskLogging`` support file with the module ``*.h`` file using:

.. code-block:: c

    #include "architecture/utilities/bskLogging.h"

The C-module configuration structure must contain a pointer to the ``BSKLogger`` type using:

.. code-block:: c

    BSKLogger *bskLogger;

The ``_bskLog`` only accepts char*/string, so the formatting must be done before logging call.  If it is a simple message without any variables being included, then you can use:

.. code-block:: c

    _bskLog(configData->bskLogger, BSK_INFORMATION, "Fixed String");

If you want to print variables to the logging string, this must be done before calling ``_bskLog``, such as in this example:

.. code-block:: c

   char info[MAX_LOGGING_LENGTH];
   sprintf(info, "Variable is too large (%d). Setting to max value.", variable);
   _bskLog(configData->bskLogger, BSK_ERROR, info);
