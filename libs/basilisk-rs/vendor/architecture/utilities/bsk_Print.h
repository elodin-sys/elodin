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

#ifndef _BSK_PRINT_
#define _BSK_PRINT_

#include <stdio.h>

typedef enum {
    MSG_DEBUG,
    MSG_INFORMATION,
    MSG_WARNING,
    MSG_ERROR,
    MSG_SILENT          // the coder should never use this flag when using BSK_PRINT().  It is used to turn off all BSK_PRINT()
} msgLevel_t_;

/* specify the BSK printing verbosity level.
 */
#define MSG_LEVEL MSG_DEBUG


#define EXPAND(x) x

#define BSK_MESSAGE(...) { printf(__VA_ARGS__); }

#ifdef _WIN32

#define BSK_PRINT(X, _fmt, ...) if (EXPAND(X) >= MSG_LEVEL) {printf(#X ": " _fmt "\n", __VA_ARGS__);}
#define BSK_PRINT_BRIEF(X, _fmt, ...) if (EXPAND(X) >= MSG_LEVEL) {printf(#X ": " _fmt "\n", __VA_ARGS__);}


#else       /* macOS and Linux */

#define WHERESTR "[FILE : %s, FUNC : %s, LINE : %d]:\n"
#define WHEREARG __FILE__,__func__,__LINE__
#define BSK_PRINT(X, _fmt, ...) if(X >= MSG_LEVEL) \
                                BSK_MESSAGE(WHERESTR #X ": " _fmt "\n", WHEREARG, ##__VA_ARGS__)
#define BSK_PRINT_BRIEF(X, _fmt, ...) if(X >= MSG_LEVEL) \
                                BSK_MESSAGE(#X ": " _fmt "\n", ##__VA_ARGS__)

#endif

#endif /* BSK_PRINT_ */
