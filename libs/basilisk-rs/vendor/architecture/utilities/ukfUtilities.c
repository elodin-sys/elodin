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

#include "architecture/utilities/ukfUtilities.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/bsk_Print.h"
#include <math.h>

void ukfQRDJustR(
	double *inMat, int32_t nRow, int32_t nCol, double *destMat)
{
	int32_t i, j, k, dimi, dimj;
	double qMat[UKF_MAX_DIM*UKF_MAX_DIM];
	double sourceMat[UKF_MAX_DIM*UKF_MAX_DIM];
	double sum;

	mSetZero(qMat, UKF_MAX_DIM, UKF_MAX_DIM);
	mSetZero(sourceMat, UKF_MAX_DIM, UKF_MAX_DIM);
	mSetZero(destMat, (size_t) nCol, (size_t) nCol);
	mCopy(inMat, (size_t) nRow, (size_t) nCol, sourceMat);

	for (i = 0; i<nCol; i++)
	{
		dimi = i*nCol + i;
		for (j = 0; j<nRow; j++)
		{
			dimj = nCol * j;
			destMat[dimi] += sourceMat[dimj + i] * sourceMat[dimj + i];
		}
		destMat[dimi] = sqrt(destMat[dimi]); /* Sum of squares, can't be negative */
		for (j = 0; j<nRow; j++)
		{
			qMat[j*nCol + i] =
				sourceMat[j*nCol + i] / destMat[dimi];
		}

		for (j = i + 1; j<nCol; j++)
		{
			sum = 0;
			dimj = nCol * j;
			for (k = 0; k<nRow; k++)
			{
				sum += sourceMat[k*nCol + j] * qMat[k*nCol + i];
			}
			destMat[i*nCol + j] = sum;
			for (k = 0; k<nRow; k++)
			{
				sourceMat[k*nCol + j] -= sum*qMat[k*nCol + i];
			}
		}
	}

	return;
}

int32_t ukfLInv(double *sourceMat, int32_t nRow, int32_t nCol, double *destMat)
{
	int i, j, k, mat_dim;

	mSetZero(destMat, (size_t) nRow, (size_t) nCol);
	if (nRow != nCol)
	{
		BSK_PRINT(MSG_WARNING, "Can't get a lower-triangular inverse of non-square matrix in ukfLInv.");
		return -1;
	}
	mat_dim = nRow;
	for (i = mat_dim - 1; i >= 0; i--)
	{
        if (sourceMat[i*mat_dim + i] == 0)
        {
            BSK_PRINT(MSG_WARNING, "Can't invert zero matrix.");
            return -1;
        }
		destMat[mat_dim*i + i] = 1.0 / sourceMat[i*mat_dim + i];
		for (j = mat_dim - 1; j >= i + 1; j--)
		{
			destMat[mat_dim*j + i] = 0.0;
			for (k = i + 1; k <= j; k++)
			{
				destMat[j*mat_dim + i] -= sourceMat[mat_dim*k + i] *
					destMat[j*mat_dim + k];
			}
			destMat[j*mat_dim + i] *= destMat[mat_dim*i + i];
		}
	}


	return 0;
}

int32_t ukfUInv(double *sourceMat, int32_t nRow, int32_t nCol, double *destMat)
{
	int i, j, k, mat_dim;

	mSetZero(destMat, (size_t) nRow, (size_t) nCol);
	if (nRow != nCol)
	{
		BSK_PRINT(MSG_WARNING, "Can't get a lower-triangular inverse of non-square matrix in ukfLInv.");
		return -1;
	}
	mat_dim = nRow;
	for (i = mat_dim - 1; i >= 0; i--)
	{
        if (sourceMat[i*mat_dim + i] == 0)
        {
            BSK_PRINT(MSG_WARNING, "Can't invert zero matrix.");
            return -1;
        }
		destMat[mat_dim*i + i] = 1.0 / sourceMat[i*mat_dim + i];
		for (j = mat_dim - 1; j >= i + 1; j--)
		{
			destMat[mat_dim*i + j] = 0.0;
			for (k = i + 1; k <= j; k++)
			{
				destMat[i*mat_dim + j] -= sourceMat[mat_dim*i + k] *
					destMat[k*mat_dim + j];
			}
			destMat[i*mat_dim + j] *= destMat[mat_dim*i + i];
		}
	}
	return 0;
}

int32_t ukfLUD(double *sourceMat, int32_t nRow, int32_t nCol,
	double *destMat, int32_t *indx)
{
	double vv[UKF_MAX_DIM];
	int32_t rowIndicator, i, j, k, imax;
	double big, dum, sum, temp;
	double TINY = 1.0E-14;

	mSetZero(destMat, (size_t) nRow, (size_t) nCol);
	for(i=0; i<nRow; i++)
    {
	    indx[i] = i;
    }
	if (nRow != nCol)
	{
		BSK_PRINT(MSG_WARNING, "Can't get a lower-triangular inverse of non-square matrix in ukfLUD.");
		return -1;
	}
	mCopy(sourceMat, (size_t) nRow, (size_t) nCol, destMat);
	vSetZero(vv, (size_t) nRow);
	rowIndicator = 1;
	for (i = 0; i < nRow; i++)
	{
		big = 0.0;
		for (j = 0; j < nRow; j++)
		{
			temp = fabs(destMat[i*nRow + j]);
			big = temp > big ? temp : big;
		}
		if (big < TINY)
		{
			BSK_PRINT(MSG_WARNING, "Singlular matrix encountered in ukfLUD LU decomposition.");
			return -1;
		}
		vv[i] = 1.0 / big;
	}
	for (j = 0; j < nRow; j++)
	{
		for (i = 0; i < j; i++)
		{
			sum = destMat[i*nRow + j];
			for (k = 0; k < i; k++)
			{
				sum -= destMat[i*nRow + k] * destMat[k*nRow + j];
			}
			destMat[i*nRow + j] = sum;
		}
		big = 0.0;
		imax = j;
		for (i = j; i < nRow; i++)
		{
			sum = destMat[i*nRow + j];
			for (k = 0; k < j; k++)
			{
				sum -= destMat[i*nRow + k] * destMat[k*nRow + j];
			}
			destMat[i*nRow + j] = sum;
			dum = vv[i] * fabs(sum);
			if (dum >= big)
			{
				big = dum;
				imax = i;
			}
		}
		if (j != imax)
		{
			for (k = 0; k < nRow; k++)
			{
				dum = destMat[imax*nRow + k];
				destMat[imax*nRow + k] = destMat[j*nRow + k];
				destMat[j*nRow + k] = dum;
			}
			rowIndicator += 1;
			vv[imax] = vv[j];
		}
		indx[j] = imax;
		if (destMat[j*nRow + j] == 0.0)
		{
			destMat[j*nRow + j] = TINY;
		}
		if (j != nRow-1)
		{
			dum = 1.0 / destMat[j*nRow + j];
			for (i = j + 1; i < nRow; i++)
			{
				destMat[i*nRow + j] *= dum;
			}
		}
	}
	return 0;
}

int32_t ukfLUBckSlv(double *sourceMat, int32_t nRow, int32_t nCol,
	int32_t *indx, double *bmat, double *destMat)
{
	int i, ip, j;
	int ii = -1;
	double sum;

	vSetZero(destMat, (size_t) nRow);
	if (nRow != nCol)
	{
		BSK_PRINT(MSG_WARNING, "Can't get a linear solution of non-square matrix in ukfLUBckSlv.");
		return -1;
	}
	vCopy(bmat, (size_t) nRow, destMat);

	for (i = 0; i < nRow; i++)
	{
		ip = indx[i];
		sum = destMat[ip];
		destMat[ip] = destMat[i];
		if (ii >= 0)
		{
			for (j = ii; j <= i-1; j++)
			{
				sum -= sourceMat[i*nRow + j] * destMat[j];
			}
		}
		else if ((int) sum)
		{
			ii = i;
		}
		destMat[i] = sum;
	}
	for (i = nRow - 1; i >= 0; i--)
	{
		sum = destMat[i];
		for (j = i + 1; j < nRow; j++)
		{
			sum -= sourceMat[i*nRow + j] * destMat[j];
		}
        if (sourceMat[i*nRow + i] == 0){
            BSK_PRINT(MSG_WARNING, "Dividing by zero.");
            return -1;}
		destMat[i] = sum / sourceMat[i*nRow + i];
	}
    return 0;
}

int32_t ukfMatInv(double *sourceMat, int32_t nRow, int32_t nCol,
	double *destMat)
{
	double LUMatrix[UKF_MAX_DIM*UKF_MAX_DIM];
	double invCol[UKF_MAX_DIM];
	double colSolve[UKF_MAX_DIM];
	int indx[UKF_MAX_DIM];
	int32_t i, j, badCall;

	mSetZero(destMat, (size_t) nRow, (size_t) nCol);
	if (nRow != nCol)
	{
		BSK_PRINT(MSG_WARNING, "Can't invert a non-square matrix in ukfMatInv.");
		return -1;
	}
	ukfLUD(sourceMat, nRow, nCol, LUMatrix, indx);
	for (j = 0; j < nRow; j++)
	{
		vSetZero(colSolve, (size_t) nRow);
		colSolve[j] = 1.0;
        badCall = ukfLUBckSlv(LUMatrix, nRow, nCol, indx, colSolve, invCol);
		for (i = 0; i < nRow; i++)
		{
			destMat[i*nRow + j] = invCol[i];
		}
	}
    return badCall;
}

int32_t ukfCholDecomp(double *sourceMat, int32_t nRow, int32_t nCol,
	double *destMat)
{
	int32_t i, j, k;
	double sigma;

	mSetZero(destMat, (size_t) nRow, (size_t) nCol);
	if (nRow != nCol)
	{
		BSK_PRINT(MSG_WARNING, "Can't get a lower-triangular inverse of non-square matrix in ukfCholDecomp.");
		return -1;
	}

	for (i = 0; i<nRow; i++)
	{
		for (j = 0; j <= i; j++)
		{
			sigma = sourceMat[nRow * i + j];
			for (k = 0; k <= (j - 1); k++)
			{
				sigma -= destMat[nRow * i + k] * destMat[nRow * j + k];
			}
			if (i == j)
			{
                if (sigma<0){
                    BSK_PRINT(MSG_WARNING, "Invalid SQRT in ukfCholDecomp, skipping value.");
                    return -1;}
				destMat[nRow * i + j] = sqrt(sigma);
			}
			else
			{
                if (destMat[nRow * j + j] == 0){
                    BSK_PRINT(MSG_WARNING, "Dividing by zero.");
                    return -1;}
				destMat[nRow * i + j] = sigma / (destMat[nRow * j + j]);
			}
		}
	}
    return 0;
}

int32_t ukfCholDownDate(double *rMat, double *xVec, double beta, int32_t nStates,
	double *rOut)
{
	int i, j;
	double wVec[UKF_MAX_DIM];
    double rEl2, bParam, gamma;
	
    vCopy(xVec, (size_t) nStates, wVec);
    mSetZero(rOut, (size_t) nStates, (size_t) nStates);

    bParam = 1.0;
	for (i = 0; i < nStates; i++)
	{
        rEl2 = rMat[i*nStates+i] * rMat[i*nStates+i];
        if (rEl2 + beta/bParam * wVec[i]*wVec[i]<0){
            BSK_PRINT(MSG_WARNING, "Invalid SQRT in ukfCholDownDate, skipping value.");
            return -1;}
        rOut[i*nStates + i] = sqrt(rEl2 + beta/bParam * wVec[i]*wVec[i]);
        gamma = rEl2*bParam + beta * wVec[i]*wVec[i];
        for(j=i+1; j<nStates; j++)
        {
            wVec[j] = wVec[j] - wVec[i]/rMat[i*nStates + i]*rMat[j*nStates+i];
            rOut[j*nStates +i] = rOut[i*nStates + i]/rMat[i*nStates + i] *
                rMat[j*nStates + i];
            rOut[j*nStates +i] += rOut[i*nStates + i]*beta*wVec[j]*wVec[i]/gamma;
        }
        bParam += beta * wVec[i]*wVec[i]/rEl2;
	}
	

	return 0;
}
