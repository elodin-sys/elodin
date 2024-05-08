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

/// \cond DO_NOT_DOCUMENT

#include <stdio.h>
#include <stdarg.h>
#include "architecture/utilities/bskLogging.h"

logLevel_t LogLevel = BSK_DEBUG;

/*! This method sets the default logging verbosity
    @param logLevel
 */
void setDefaultLogLevel(logLevel_t logLevel)
{
    LogLevel = logLevel;
}

/*! This method gets the default logging verbosity */
logLevel_t getDefaultLogLevel()
{
    return LogLevel;
}

/*! This method prints the default logging verbosity */
void printDefaultLogLevel()
{
    std::map<int, const char*> logLevelMap
    {
        {0, "BSK_DEBUG"},
        {1, "BSK_INFORMATION"},
        {2, "BSK_WARNING"},
        {3, "BSK_ERROR"},
        {4, "BSK_SILENT"}
    };
    const char* defaultLevelStr = logLevelMap[LogLevel];
    printf("Default Logging Level: %s\n", defaultLevelStr);
}

/*! The constructor initialises the logger for a module and uses default verbostiy level for logging */
BSKLogger::BSKLogger()
{
    //Default print verbosity
    this->_logLevel = getDefaultLogLevel();
}

/*! The constructor initialises the logger for a module and uses a user defined verbostiy level for logging
    @param logLevel
 */
BSKLogger::BSKLogger(logLevel_t logLevel)
{
    this->_logLevel = logLevel;
}

/*! This method changes the logging verbosity after construction
 */
void BSKLogger::setLogLevel(logLevel_t logLevel)
{
    this->_logLevel = logLevel;
}

/*! This method reads the current logging verbosity */
void BSKLogger::printLogLevel()
{
    const char* currLevelStr = this->logLevelMap[this->_logLevel];
    printf("Current Logging Level: %s\n", currLevelStr);
}

/*! Get the current log level value */
int BSKLogger::getLogLevel()
{
    return this->_logLevel;
}

/*! This method logs information. The current behavior is to simply print out the message and the targeted logging level.
    This should be the main method called in user code.
    @param targetLevel
    @param info
*/
void BSKLogger::bskLog(logLevel_t targetLevel, const char* info, ...)
{
    if(targetLevel >= this->_logLevel)
    {
        const char* targetLevelStr = this->logLevelMap[targetLevel];
        char formatMessage[MAX_LOGGING_LENGTH];
        va_list args;
        va_start (args, info);
        vsnprintf(formatMessage, sizeof(formatMessage), info, args);
        printf("%s: %s\n", targetLevelStr, formatMessage);
    }
}

/*! Section contains C interfacing to C++ object */
EXTERN BSKLogger* _BSKLogger(void)
{
    return new BSKLogger();
}

EXTERN void _BSKLogger_d(BSKLogger* bskLogger)
{
    delete bskLogger;
}

/*! This method reads the current logging verbosity */
EXTERN void _printLogLevel(BSKLogger* bskLogger)
{
    bskLogger->printLogLevel();
}

/*! This method changes the logging verbosity after construction
    @param bskLogger
    @param logLevel
 */
EXTERN void _setLogLevel(BSKLogger* bskLogger, logLevel_t logLevel)
{
    bskLogger->setLogLevel(logLevel);
}

/*! This method logs information. The current behavior is to simply print out the message and the targeted logging level.
    This should be the main method called in user code.
*/
EXTERN void _bskLog(BSKLogger* bskLogger, logLevel_t logLevel, const char* info)
{
    bskLogger->bskLog(logLevel, info);
}

/// \endcond
