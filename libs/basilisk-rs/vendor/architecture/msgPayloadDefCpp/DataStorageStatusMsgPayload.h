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

#include <vector>

#ifndef BASILISK_DATASTORAGESTATUSSIMMSG_H
#define BASILISK_DATASTORAGESTATUSSIMMSG_H


/*! @brief Message to store current storage unit stored data, storage capacity, and received data.*/
typedef struct
//@cond DOXYGEN_IGNORE
DataStorageStatusMsgPayload
//@endcond
{
    double storageLevel; //!< [b] Storage unit stored data in bits.
    double storageCapacity; //!< [b] Maximum data storage unit capacity.
    double currentNetBaud; //!< [baud] Current data written to or removed from the storage unit net power.
    std::vector<std::string> storedDataName;    //!< [] vector of data name strings
    std::vector<double> storedData;             //!< [] vector of stored data amount for each data name group
}DataStorageStatusMsgPayload;

#endif //BASILISK_DATASTORAGESTATUSSIMMSG_H
