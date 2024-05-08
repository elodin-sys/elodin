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

#ifndef COMMON_UTILS_SEMAPHORE_H
#define COMMON_UTILS_SEMAPHORE_H
// https://riptutorial.com/cplusplus/example/30142/semaphore-cplusplus-11

#include <mutex>
#include <condition_variable>


/*! Basilisk semaphore class */
class BSKSemaphore
{
    std::mutex mutex;
    std::condition_variable cv;
    size_t count;

public:
    /*! method description */
    BSKSemaphore(int count_in = 0)
        : count(count_in)
    {
    }
    
    /*! release the lock */
    inline void release()
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            ++count;
            //notify the waiting thread
        }
        cv.notify_one();
    }
    
    /*! aquire the lock */
    inline void acquire()
    {
        std::unique_lock<std::mutex> lock(mutex);
        while (count == 0)
        {
            //wait on the mutex until notify is called
            cv.wait(lock);
        }
        --count;
    }
};

#endif
