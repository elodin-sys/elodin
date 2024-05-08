/*
 ISC License

 Copyright (c) 2023, Laboratory for Atmospheric and Space Physics, University of Colorado at Boulder

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

#ifndef UNITTESTCOMPARATORS_H
#define UNITTESTCOMPARATORS_H

#include <math.h>

int isEqual(double a, double b, double accuracy)
{
    if(fabs(a - b) > accuracy) {
        return 0;
    }
    return 1;
}

int isEqualRel(double a, double b, double accuracy)
{
    if(fabs(a - b)/fabs(a) > accuracy) {
        return 0;
    }
    return 1;
}

#endif //UNITTESTCOMPARATORS_H
