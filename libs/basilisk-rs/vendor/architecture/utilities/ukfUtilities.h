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

#ifndef _UKF_UTILITIES_H_
#define _UKF_UTILITIES_H_

#include <stdint.h>
#include <string.h>
#include<architecture/utilities/bskLogging.h>

#define UKF_MAX_DIM 20


#ifdef __cplusplus
extern "C" {
#endif

	void ukfQRDJustR(
		double *sourceMat, int32_t nRow, int32_t nCol, double *destMat);
	int32_t ukfLInv(
		double *sourceMat, int32_t nRow, int32_t nCol, double *destMat);
	int32_t ukfUInv(
		double *sourceMat, int32_t nRow, int32_t nCol, double *destMat);
	int32_t ukfLUD(double *sourceMat, int32_t nRow, int32_t nCol,
		double *destMat, int32_t *indx);
	int32_t ukfLUBckSlv(double *sourceMat, int32_t nRow, int32_t nCol,
		int32_t *indx, double *bmat, double *destMat);
	int32_t ukfMatInv(double *sourceMat, int32_t nRow, int32_t nCol,
		double *destMat);
	int32_t ukfCholDecomp(double *sourceMat, int32_t nRow, int32_t nCol,
		double *destMat);
    int32_t ukfCholDownDate(double *rMat, double *xVec, double beta, int32_t nStates,
                         double *rOut);

#ifdef __cplusplus
}
#endif


#endif
