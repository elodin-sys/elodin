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

#ifndef msgHeader_h
#define msgHeader_h

/*! @brief message system 2 header information structure */
typedef struct {
    int64_t isLinked;       //!< flag if the message has is connected to another message
    int64_t isWritten;      //!< flag if the message conntent has ever been written
    uint64_t timeWritten;   //!< [ns] time the message was written
    int64_t moduleID;       //!< ID of the module who wrote the message, negative value for Python module, non-negative for C/C++ modules
}MsgHeader;

#endif /* msgHeader_h */
