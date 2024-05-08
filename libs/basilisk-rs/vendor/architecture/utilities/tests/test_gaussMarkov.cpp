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

#include "architecture/utilities/gauss_markov.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>


Eigen::Vector2d calculateSD(Eigen::MatrixXd dat, int64_t numPts)
{
    Eigen::Vector2d sum = dat.rowwise().sum();

    Eigen::Vector2d means = sum / numPts;
    Eigen::MatrixXd mean;
    mean.resize(2, numPts);
    mean.block(0, 0, 1, numPts).fill(means(0) / numPts);
    mean.block(1, 0, 1, numPts).fill(means(1) / numPts);

    Eigen::MatrixXd resid = dat - mean;
    resid = resid.cwiseProduct(resid);
    Eigen::MatrixXd stnd = resid.rowwise().sum();
    stnd = stnd / numPts;
    stnd = stnd.array().sqrt().matrix();

    return stnd;
}

TEST(GausssMarkov, stdDeviationIsExpected) {
    //Test if the std deviation is what we asked for
    uint64_t seedIn = 1000;
    Eigen::Matrix2d propIn;
    propIn << 1,1,0,1;
    Eigen::Matrix2d covar;
    covar << 1500,0,0,1.5;
    Eigen::Vector2d bounds;
    bounds << 1e-15, 1e-15; //small but non-zero required for "white" noise
    GaussMarkov errorModel = GaussMarkov(2);
    errorModel.setRNGSeed(seedIn);
    errorModel.setPropMatrix(propIn);
    errorModel.setNoiseMatrix(covar);
    errorModel.setUpperBounds(bounds);
    int64_t numPts = 100000;

    Eigen::MatrixXd noiseOut;
    noiseOut.resize(2, numPts);

    for(int64_t i = 0; i < numPts; i++){
        errorModel.computeNextState();
        noiseOut.block(0, i, 2, 1) = errorModel.getCurrentState();
    }

    //Test if the std deviation is what we asked for
    Eigen::Vector2d stds = calculateSD(noiseOut, numPts);
    Eigen::Vector2d stdsIn;
    stdsIn(0) = covar(0,0) / 1.5;
    stdsIn(1) = covar(1,1) / 1.5;
    EXPECT_LT((stdsIn(0) - stds(0))/(stdsIn(0)), 0.1);
    EXPECT_LT((stdsIn(1) - stds(1))/(stdsIn(1)), 0.1);
}

TEST(GaussMarkov, meanIsZero) {
    //Test if the mean is zero
    uint64_t seedIn = 1000;
    Eigen::Matrix2d propIn;
    propIn << 1,1,0,1;
    Eigen::Matrix2d covar;
    covar << 1500,0,0,1.5;
    Eigen::Vector2d bounds;
    bounds << 1e-15, 1e-15; //small but non-zero required for "white" noise
    GaussMarkov errorModel = GaussMarkov(2);
    errorModel.setRNGSeed(seedIn);
    errorModel.setPropMatrix(propIn);
    errorModel.setNoiseMatrix(covar);
    errorModel.setUpperBounds(bounds);
    int64_t numPts = 100000;

    Eigen::MatrixXd noiseOut;
    noiseOut.resize(2, numPts);

    for(int64_t i = 0; i < numPts; i++){
        errorModel.computeNextState();
        noiseOut.block(0, i, 2, 1) = errorModel.getCurrentState();
    }

    Eigen::Vector2d means = noiseOut.rowwise().mean();
    Eigen::Vector2d meansIn;
    meansIn << 0, 0;
    EXPECT_LT(fabs(meansIn(0) - means(0)), 5);
    EXPECT_LT(fabs(meansIn(1) - means(1)), 0.05);
}

TEST(GaussMarkov, boundsAreRespected) {
    //Test if the bounds are obeyed
    uint64_t seedIn = 1500;
    Eigen::Matrix2d propIn;
    propIn << 1,0,0,1;
    Eigen::Matrix2d covar;
    covar << 1.5,0,0,0.015;
    Eigen::Vector2d bounds;
    bounds << 10., 0.1;
    GaussMarkov errorModel = GaussMarkov(2);
    errorModel.setRNGSeed(seedIn);
    errorModel.setPropMatrix(propIn);
    errorModel.setNoiseMatrix(covar);
    errorModel.setUpperBounds(bounds);

    int64_t numPts = 100000;
    Eigen::MatrixXd noiseOut;
    noiseOut.resize(2, numPts);

    Eigen::Vector2d maxOut;
    maxOut.fill(0.0);
    Eigen::Vector2d minOut;
    minOut.fill(0.0);

    numPts = (int64_t) 1e6;
    noiseOut.resize(2,numPts);

    for(int64_t i = 0; i < numPts; i++){
        errorModel.computeNextState();
        noiseOut.block(0, i, 2, 1) = errorModel.getCurrentState();
        if (noiseOut(0,i) > maxOut(0)){
            maxOut(0) = noiseOut(0,i);
        }
        if (noiseOut(0,i) < minOut(0)){
            minOut(0) = noiseOut(0,i);
        }
        if (noiseOut(1,i) > maxOut(1)){
            maxOut(1) = noiseOut(1,i);
        }
        if (noiseOut(1,i) < minOut(1)){
            minOut(1) = noiseOut(1,i);
        }
    }

    EXPECT_LT(fabs(12.481655180914322 - maxOut(0)) / 12.481655180914322, 5e-1);
    EXPECT_LT(fabs(0.12052269089286843 - maxOut(1)) / 0.12052269089286843, 5e-1);
    EXPECT_LT(fabs(-12.230618182796439 - minOut(0)) / -12.230618182796439, 5e-1);
    EXPECT_LT(fabs(-0.12055787311661936 - minOut(1)) / -0.12055787311661936, 5e-1);
}
