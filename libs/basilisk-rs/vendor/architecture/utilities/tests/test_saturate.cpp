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

#include "architecture/utilities/saturate.h"
#include "architecture/utilities/linearAlgebra.h"
#include <gtest/gtest.h>


TEST(Saturate, testSaturate) {
    Eigen::Vector3d states;
    states << -555, 1.27, 5000000.;
    auto saturator = Saturate(3);
    Eigen::MatrixXd bounds;
    bounds.resize(3,2);
    bounds << -400., 0, 5, 10, -1, 5000001;
    saturator.setBounds(bounds);
    states = saturator.saturate(states);
    Eigen::Vector3d expected;
    expected << -400, 5, 5000000;
    EXPECT_TRUE(states == expected);
}
