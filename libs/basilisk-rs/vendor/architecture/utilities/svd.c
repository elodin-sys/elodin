/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#include "svd.h"
#include <stdlib.h>
#include "architecture/utilities/linearAlgebra.h"
#include <math.h>
#include <stdio.h>

/*
 * Originally Authored by Hanspeter Schaub. Sourced from: Recipes in C
 * Revised by Henry Macanas to eliminate dynamic memory allocation, operate based
 * on 0 based indexing.
 * Sourced from: https://gist.github.com/sasekazu/32f966816ad6d9244259.
 */

#define SIGN(a,b) ((b) > 0.0 ? fabs(a) : - fabs(a))

static double maxarg1, maxarg2;
#define DMAX(a,b) (maxarg1 = (a),maxarg2 = (b),(maxarg1) > (maxarg2) ? (maxarg1) : (maxarg2))

static int iminarg1, iminarg2;
#define IMIN(a,b) (iminarg1 = (a),iminarg2 = (b),(iminarg1 < (iminarg2) ? (iminarg1) : iminarg2))

static double sqrarg;
#define SQR(a) ((sqrarg = (a)) == 0.0 ? 0.0 : sqrarg * sqrarg)

// calculates sqrt( a^2 + b^2 )
double pythag(double a, double b) {
    double absa, absb;

    absa = fabs(a);
    absb = fabs(b);

    if (absa > absb)
        return (absa * sqrt(1.0 + SQR(absb/absa)));
    else
        return (absb == 0.0 ? 0.0 : absb * sqrt(1.0 + SQR(absa / absb)));
}

/*
 Modified from Numerical Recipes in C
 Given a matrix a[dim1 * dim2], svdcmp() computes its singular value
 decomposition, mx = U * W * Vt. mx is replaced by U when svdcmp
 returns.  The diagonal matrix W is output as a vector w[dim2].
 V (not V transpose) is output as the matrix V[dim2 * dim2].
 */
int svdcmp(double *mx, size_t dim1, size_t dim2, double *w, double *v) {
    int flag, i, its, j, jj, k, l, nm, cm;
    double anorm, c, f, g, h, s, scale, x, y, z, max;
    double rv1[LINEAR_ALGEBRA_MAX_ARRAY_SIZE];
    vSetZero(rv1, dim2);

    g = scale = anorm = 0.0;
    for (i = 0; i < dim2; i++) {
        l = i + 1;
        rv1[i] = scale * g;
        g = s = scale = 0.0;
        if (i < dim1) {
            for (k = i; k < dim1; k++)
                scale += fabs(mx[MXINDEX(dim2, k, i)]);
            if (scale) {
                for (k = i; k < dim1; k++) {
                    mx[MXINDEX(dim2, k, i)] /= scale;
                    s += mx[MXINDEX(dim2, k, i)] * mx[MXINDEX(dim2, k, i)];
                }
                f = mx[MXINDEX(dim2, i, i)];
                g = -SIGN(sqrt(s),f);
                h = f * g - s;
                mx[MXINDEX(dim2, i, i)] = f - g;
                for (j = l; j < dim2; j++) {
                    for (s = 0.0, k = i; k < dim1; k++)
                        s += mx[MXINDEX(dim2, k, i)] * mx[MXINDEX(dim2, k, j)];
                    f = s / h;
                    for (k = i; k < dim1; k++)
                        mx[MXINDEX(dim2, k, j)] += f * mx[MXINDEX(dim2, k, i)];
                }
                for (k = i; k < dim1; k++)
                    mx[MXINDEX(dim2, k, i)] *= scale;
            }
        }
        w[i] = scale * g;
        g = s = scale = 0.0;
        if (i < dim1 && i != dim2 - 1) {
            for (k = l; k < dim2; k++)
                scale += fabs(mx[MXINDEX(dim2, i, k)]);
            if (scale) {
                for (k = l; k < dim2; k++) {
                    mx[MXINDEX(dim2, i, k)] /= scale;
                    s += mx[MXINDEX(dim2, i, k)] * mx[MXINDEX(dim2, i, k)];
                }
                f = mx[MXINDEX(dim2, i, l)];
                g = -SIGN(sqrt(s),f);
                h = f * g - s;
                mx[MXINDEX(dim2, i, l)] = f - g;
                for (k = l; k < dim2; k++)
                    rv1[k] = mx[MXINDEX(dim2, i, k)] / h;
                for (j = l; j < dim1; j++) {
                    for (s = 0.0, k = l; k < dim2; k++)
                        s += mx[MXINDEX(dim2, j, k)] * mx[MXINDEX(dim2, i, k)];
                    for (k = l; k < dim2; k++)
                        mx[MXINDEX(dim2, j, k)] += s * rv1[k];
                }
                for (k = l; k < dim2; k++)
                    mx[MXINDEX(dim2, i, k)] *= scale;
            }
        }
        anorm = DMAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
    }

    for (i = (int) dim2 - 1; i >= 0; i--) {
        if (i < dim2 - 1) {
            if (g) {
                for (j = l; j < dim2; j++)
                    v[MXINDEX(dim2, j, i)] = (mx[MXINDEX(dim2, i, j)] / mx[MXINDEX(dim2, i, l)]) / g;
                for (j = l; j < dim2; j++) {
                    for (s = 0.0, k = l; k < dim2; k++)
                        s += mx[MXINDEX(dim2, i, k)] * v[MXINDEX(dim2, k, j)];
                    for (k = l; k < dim2; k++)
                        v[MXINDEX(dim2, k, j)] += s * v[MXINDEX(dim2, k, i)];
                }
            }
            for (j = l; j < dim2; j++)
                v[MXINDEX(dim2, i, j)] = v[MXINDEX(dim2, j, i)] = 0.0;
        }
        v[MXINDEX(dim2, i, i)] = 1.0;
        g = rv1[i];
        l = i;
    }

    for (i = IMIN((int) dim1, (int) dim2) - 1; i >= 0; i--) {
        l = i + 1;
        g = w[i];
        for (j = l; j < dim2; j++)
            mx[MXINDEX(dim2, i, j)] = 0.0;
        if (g) {
            g = 1.0 / g;
            for (j = l; j < dim2; j++) {
                for (s = 0.0, k = l; k < dim1; k++)
                    s += mx[MXINDEX(dim2, k, i)] * mx[MXINDEX(dim2, k, j)];
                f = (s / mx[MXINDEX(dim2, i, i)]) * g;
                for (k = i; k < dim1; k++)
                    mx[MXINDEX(dim2, k, j)] += f * mx[MXINDEX(dim2, k, i)];
            }
            for (j = i; j < dim1; j++)
                mx[MXINDEX(dim2, j, i)] *= g;
        } else
            for (j = i; j < dim1; j++)
                mx[MXINDEX(dim2, j, i)] = 0.0;
        ++mx[MXINDEX(dim2, i, i)];
    }

    for (k = (int) dim2 - 1; k >= 0; k--) {
        for (its = 0; its < 30; its++) {
            flag = 1;
            for (l = k; l >= 0; l--) {
                nm = l - 1;
                if ((fabs(rv1[l]) + anorm) == anorm) {
                    flag = 0;
                    break;
                }
                if ((fabs(w[nm]) + anorm) == anorm)
                    break;
            }
            if (flag) {
                c = 0.0;
                s = 1.0;
                for (i = l; i <= k; i++) {
                    f = s * rv1[i];
                    rv1[i] = c * rv1[i];
                    if ((fabs(f) + anorm) == anorm)
                        break;
                    g = w[i];
                    h = pythag(f, g);
                    w[i] = h;
                    h = 1.0 / h;
                    c = g * h;
                    s = -f * h;
                    for (j = 0; j < dim1; j++) {
                        y = mx[MXINDEX(dim2, j, nm)];
                        z = mx[MXINDEX(dim2, j, i)];
                        mx[MXINDEX(dim2, j, nm)] = y * c + z * s;
                        mx[MXINDEX(dim2, j, i)] = z * c - y * s;
                    }
                }
            }
            z = w[k];
            if (l == k) {
                if (z < 0.0) {
                    w[k] = -z;
                    for (j = 0; j < dim2; j++)
                        v[MXINDEX(dim2, j, k)] = -v[MXINDEX(dim2, j, k)];
                }
                break;
            }
            if (its == 29)
                printf("no convergence in 30 svdcmp iterations\n");
            x = w[l];
            nm = k - 1;
            y = w[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = pythag(f, 1.0);
            f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g,f)))- h)) / x;
            c = s = 1.0;
            for (j = l; j <= nm; j++) {
                i = j + 1;
                g = rv1[i];
                y = w[i];
                h = s * g;
                g = c * g;
                z = pythag(f, h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y *= c;
                for (jj = 0; jj < dim2; jj++) {
                    x = v[MXINDEX(dim2, jj, j)];
                    z = v[MXINDEX(dim2, jj, i)];
                    v[MXINDEX(dim2, jj, j)] = x * c + z * s;
                    v[MXINDEX(dim2, jj, i)] = z * c - x * s;
                }
                z = pythag(f, h);
                w[j] = z;
                if (z) {
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                }
                f = c * g + s * y;
                x = c * y - s * g;
                for (jj = 0; jj < dim1; jj++) {
                    y = mx[MXINDEX(dim2, jj, j)];
                    z = mx[MXINDEX(dim2, jj, i)];
                    mx[MXINDEX(dim2, jj, j)] = y * c + z * s;
                    mx[MXINDEX(dim2, jj, i)] = z * c - y * s;
                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = x;
        }
    }
    
    /*    sort by largest singular value */
    for (i = 0; i < dim2 -  1; i++) {
        max = w[i];
        cm = i;
        for (j = i; j < dim2;j++) {
            if (w[j]>max) {
                max = w[j];
                cm = j;
            }
        }
        if (i != cm) {
            for (j = 0 ; j < dim1; j++) {
                f = mx[MXINDEX(dim2, j, i)];
                mx[MXINDEX(dim2, j, i)] = mx[MXINDEX(dim2, j, cm)];
                mx[MXINDEX(dim2, j, cm)] = f;
            }
            for (j = 0; j < dim2; j++) {
                f = v[MXINDEX(dim2, j, i)];
                v[MXINDEX(dim2, j, i)] = v[MXINDEX(dim2, j, cm)];
                v[MXINDEX(dim2, j, cm)] = f;
            }
            f = w[i];
            w[i] = w[cm];
            w[cm] = f;
        }
    }

    return (0);
}

void solveSVD(double *mx, size_t dim1, size_t dim2, double *x, double *b, double minSV)
{
    double mxCopy[LINEAR_ALGEBRA_MAX_ARRAY_SIZE];
    double w[LINEAR_ALGEBRA_MAX_ARRAY_SIZE];
    double v[LINEAR_ALGEBRA_MAX_ARRAY_SIZE];
    double A[LINEAR_ALGEBRA_MAX_ARRAY_SIZE];
    double uTranspose[LINEAR_ALGEBRA_MAX_ARRAY_SIZE];
    double wInvDiag[LINEAR_ALGEBRA_MAX_ARRAY_SIZE];
    double temp[LINEAR_ALGEBRA_MAX_ARRAY_SIZE];
    int j;
    
    vSetZero(w, dim2);
    mSetZero(v, dim2, dim2);
    mSetZero(A, dim1, dim2);
    mSetZero(uTranspose, dim1, dim2);
    mSetZero(wInvDiag, dim2, dim2);
    mCopy(mx, dim1, dim2, mxCopy);
    
    svdcmp(mxCopy, dim1, dim2, w, v);
    
    // condition wInvDiag
    for (j = 0; j < dim2;  j++)
    {
        if (w[j] >= minSV)
            wInvDiag[MXINDEX(dim2, j, j)] = 1.0 / w[j];
    }
    
    // compute A
    mTranspose(mxCopy, dim1, dim2, uTranspose);
    mMultM(v, dim2, dim2, wInvDiag, dim2, dim2, temp);
    mMultM(temp, dim2, dim2, uTranspose, dim2, dim1, A);

    // solve for x
    mMultV(A, dim2, dim1, b, x);
}
