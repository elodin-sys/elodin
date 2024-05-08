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

#include "architecture/utilities/orbitalMotion.h"
#include "architecture/utilities/astroConstants.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "unitTestComparators.h"
#include <gtest/gtest.h>


const double orbitalEnvironmentAccuracy = 1e-10;
const double orbitalElementsAccuracy = 1e-11;

TEST(OrbitalMotion, atmosphericDensity_200km) {
    double alt = 200.0;
    double result = atmosphericDensity(alt);
    EXPECT_NEAR(result, 1.64100656241e-10, orbitalEnvironmentAccuracy);
}

TEST(OrbitalMotion, atmosphericDensity_2000km) {
    double alt = 2000.0;
    double result = atmosphericDensity(alt);
    EXPECT_NEAR(result, 2.48885731828e-15, orbitalEnvironmentAccuracy);
}

class DebeyeLengthTests :public ::testing::TestWithParam<std::tuple<double, double>> {};

TEST_P(DebeyeLengthTests, checksDebeyeLengthAtAltitude) {
    auto [altitude, expected] = GetParam();
    EXPECT_NEAR(expected, debyeLength(altitude), orbitalEnvironmentAccuracy);
}

INSTANTIATE_TEST_SUITE_P(
        OrbitalMotion,
        DebeyeLengthTests,
        ::testing::Values(
                std::make_tuple(400.0, 0.00404),
                std::make_tuple(1000.0, 0.0159),
                std::make_tuple(10000.0, 0.0396),
                std::make_tuple(34000.0, 400.30000000000018))
                );

TEST(OrbitalMotion, atmosphericDrag) {
    double check[3];
    double ans[3];
    double r[3] = {6200.0, 100.0, 2000.0};
    double v[3] = {1.0, 9.0, 1.0};
    double A = 2.0;
    double Cd = 0.2;
    double m = 50.0;

    v3Set(-2.8245395411253663e-007, -2.5420855870128297e-006, -2.8245395411253663e-007, check);
    atmosphericDrag(Cd, A, m, r, v, ans);
    // @TODO refactor later to be a Google Test MATCHER
    EXPECT_TRUE(v3IsEqual(ans, check, 11));
}

TEST(OrbitalMotion, jPerturb_order_6) {
    double check[3];
    double ans[3];
    double r[3] = {6200.0, 100.0, 2000.0};
    int order = 6;

    v3Set(-7.3080959003487213e-006, -1.1787251452175358e-007, -1.1381118473672282e-005, check);
    jPerturb(r, order, ans);
    EXPECT_TRUE(v3IsEqual(ans, check, 11));
}

TEST(OrbitalMotion, solarRadiationPressure) {
    double check[3];
    double ans[3];
    double A = 2.0;
    double m = 50.0;
    double r[3] = {1.0, 0.3, -0.2};

    v3Set(-1.9825487816e-10, -5.94764634479e-11, 3.96509756319e-11, check);
    solarRad(A, m, r, ans);
    EXPECT_TRUE(v3IsEqual(ans, check, 11));
}

TEST(OrbitalMotion, elem2rv1DEccentric)
{
    double r[3];
    double v[3];
    double r2[3];
    double v3_2[3];
    classicElements elements;
    elements.a = 7500.0;
    elements.e = 1.0;
    elements.i = 40.0 * D2R;
    elements.Omega = 133.0 * D2R;
    elements.omega = 113.0 * D2R;
    elements.f = 23.0 * D2R;
    elements.alpha = 1.0 / elements.a;
    elements.rPeriap = elements.a*(1.-elements.e);
    elements.rApoap = elements.a*(1.+elements.e);
    elem2rv(MU_EARTH, &elements, r, v);
    v3Set(-148.596902253492, -457.100381534593, 352.773096481799, r2);
    v3Set(-8.93065944520745, -27.4716886950712, 21.2016289595043, v3_2);
    EXPECT_TRUE(v3IsEqualRel(r, r2, orbitalElementsAccuracy) || !v3IsEqualRel(v, v3_2, orbitalElementsAccuracy));
}

TEST(OrbitalMotion, elem2rv1DHyperbolic)
{
    double r[3];
    double v[3];
    double r2[3];
    double v3_2[3];
    classicElements elements;
    elements.a = -7500.0;
    elements.e = 1.0;
    elements.i = 40.0 * D2R;
    elements.Omega = 133.0 * D2R;
    elements.omega = 113.0 * D2R;
    elements.f = 23.0 * D2R;
    elements.alpha = 1.0 / elements.a;
    elements.rPeriap = elements.a*(1.-elements.e);
    elements.rApoap = elements.a*(1.+elements.e);
    elem2rv(MU_EARTH, &elements, r, v);
    v3Set(-152.641873349816, -469.543156608544, 362.375968124408, r2);
    v3Set(-9.17378720883851, -28.2195763820421, 21.7788208976681, v3_2);
    EXPECT_TRUE(v3IsEqualRel(r, r2, orbitalElementsAccuracy) && v3IsEqualRel(v, v3_2, orbitalElementsAccuracy));
}

class TwoDimensionHyperbolic : public ::testing::Test {
protected:
    double r[3];
    double v[3];
    double r2[3];
    double v3_2[3];
    classicElements elements;

    void SetUp() override {
        elements.a = -7500.0;
        elements.e = 1.4;
        elements.i = 40.0 * D2R;
        elements.Omega = 133.0 * D2R;
        elements.omega = 113.0 * D2R;
        elements.f = 23.0 * D2R;
        elements.alpha = 1.0 / elements.a;
        elements.rPeriap = elements.a*(1.-elements.e);
        elements.rApoap = elements.a*(1.+elements.e);

        v3Set(319.013136281857, -2796.71958333493, 1404.6919948109, r2);
        v3Set(15.3433051336115, -5.87012423567412, -6.05659420479213, v3_2);
    }
};

TEST_F(TwoDimensionHyperbolic, elem2rv)
{
    elem2rv(MU_EARTH, &elements, r, v);
    EXPECT_TRUE(v3IsEqualRel(r, r2, orbitalElementsAccuracy) && v3IsEqualRel(v, v3_2, orbitalElementsAccuracy));
}

TEST_F(TwoDimensionHyperbolic, rv2elem) {
    rv2elem(MU_EARTH, r2, v3_2, &elements);
    EXPECT_PRED3(isEqualRel, elements.a, -7500.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.e, 1.4, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.i, 40.0 * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.Omega, 133.0 * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.omega, 113.0 * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.f, 23.0 * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rmag, 3145.881340612725, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rPeriap,3000.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rApoap, -18000., orbitalElementsAccuracy);
}

class TwoDimensionParabolic : public ::testing::Test {
protected:
    double r[3];
    double v[3];
    double r2[3];
    double v3_2[3];
    classicElements elements;

    void SetUp() override {
        elements.alpha = 0.0; /* zero orbit energy, i.e. parabolic */
        elements.a = 0.0;
        elements.rPeriap = 7500.;
        elements.e = 1.0;
        elements.i = 40.0 * D2R;
        elements.Omega = 133.0 * D2R;
        elements.omega = 113.0 * D2R;
        elements.f = 123.0 * D2R;
        v3Set(27862.6148209797, 795.70270010667, -17554.0435142669, r2);
        v3Set(3.06499561197954, 2.21344887266898, -3.14760065404514, v3_2);
    }
};

TEST_F(TwoDimensionParabolic, elem2rv) {
    elem2rv(MU_EARTH, &elements, r, v);
    EXPECT_TRUE(v3IsEqualRel(r, r2, orbitalElementsAccuracy) && v3IsEqualRel(v, v3_2, orbitalElementsAccuracy));
}

TEST_F(TwoDimensionParabolic, rv2elem) {
    rv2elem(MU_EARTH, r2, v3_2, &elements);
    EXPECT_NEAR(elements.alpha, 0.0, orbitalElementsAccuracy);
    EXPECT_NEAR(elements.a, 0.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.e, 1.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.i, 40.0 * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.Omega, 133.0 * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.omega, 113.0 * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.f, 123.0 * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rmag, 32940.89997480352, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rPeriap,7500.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rApoap, 0.0, orbitalElementsAccuracy);
}

class TwoDimensionElliptical : public ::testing::Test {
protected:
    double r[3];
    double v[3];
    double r2[3];
    double v3_2[3];
    classicElements elements;

    void SetUp() override {
        elements.a     = 7500.0;
        elements.e     = 0.5;
        elements.i     = 40.0 * D2R;
        elements.Omega = 133.0 * D2R;
        elements.omega = 113.0 * D2R;
        elements.f     = 123.0 * D2R;    /* true anomaly */
        elements.alpha = 1.0 / elements.a;
        elements.rPeriap = elements.a*(1.-elements.e);
        elements.rApoap = elements.a*(1.+elements.e);

        v3Set(6538.3506963942027, 186.7227227879431, -4119.3008399778619, r2);
        v3Set(1.4414106130924005, 5.588901415902356, -4.0828931566657038, v3_2);
    }
};

TEST_F(TwoDimensionElliptical, elem2rv) {
    elem2rv(MU_EARTH, &elements, r, v);
    EXPECT_TRUE(v3IsEqualRel(r, r2, orbitalElementsAccuracy) && v3IsEqualRel(v, v3_2, orbitalElementsAccuracy));
}

TEST_F(TwoDimensionElliptical, rv2elem) {
    rv2elem(MU_EARTH, r2, v3_2, &elements);
    EXPECT_PRED3(isEqualRel, elements.a, 7500.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.e, 0.5, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.i, 40.0 * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.Omega, 133.0 * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.omega, 113.0 * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.f, 123.0 * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rmag, 7730.041048693483, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rPeriap, 3750.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rApoap, 11250.0, orbitalElementsAccuracy);
}

class NonCircularEquitorial : public ::testing::Test {
protected:
    double r[3];
    double v[3];
    double r2[3];
    double v3_2[3];
    classicElements elements;

    void SetUp() override {
        elements.a = 7500.0;
        elements.e = 0.5;
        elements.i = 0.0 * D2R;
        elements.Omega = 133.0 * D2R;
        elements.omega = 113.0 * D2R;
        elements.f = 123.0 * D2R;
        v3Set(7634.8714161163643, 1209.2448361913848, -0, r2);
        v3Set(2.5282399359829868, 6.6023861555546057, -0, v3_2);
    }
};

TEST_F(NonCircularEquitorial, elem2rv) {
    elem2rv(MU_EARTH, &elements, r, v);
    EXPECT_TRUE(v3IsEqualRel(r, r2, orbitalElementsAccuracy) && v3IsEqualRel(v, v3_2, orbitalElementsAccuracy));
}

TEST_F(NonCircularEquitorial, rv2elem) {
    rv2elem(MU_EARTH, r2, v3_2, &elements);
    EXPECT_PRED3(isEqualRel, elements.a, 7500.00, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.e, 0.5, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.i, 0.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.Omega, 0.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.omega, (133+113)*D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.f, 123.0 * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rmag,   7730.041048693483, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rPeriap,3750.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rApoap, 11250.0, orbitalElementsAccuracy);
}

class NonCircularNearEquitorial : public ::testing::Test {
protected:
    double r[3];
    double v[3];
    double r2[3];
    double v3_2[3];
    classicElements elements;
    double eps2 = 1e-12 * 0.5;

    void SetUp() override {
        elements.a = 7500.0;
        elements.e = 0.5;
        elements.i = eps2;
        elements.Omega = 133.0 * D2R;
        elements.omega = 113.0 * D2R;
        elements.f = 123.0 * D2R;
        v3Set(7634.87141611636, 1209.24483619139, -3.20424723337984e-09, r2);
        v3Set(2.52823993598298, 6.60238615555461, -3.17592708317508e-12, v3_2);
    }
};

TEST_F(NonCircularNearEquitorial, elem2rv) {
    elem2rv(MU_EARTH, &elements, r, v);
    EXPECT_TRUE(v3IsEqualRel(r, r2, orbitalElementsAccuracy) && v3IsEqualRel(v, v3_2, orbitalElementsAccuracy));
}

TEST_F(NonCircularNearEquitorial, rv2elem) {
    rv2elem(MU_EARTH, r2, v3_2, &elements);
    EXPECT_PRED3(isEqualRel, elements.a, 7500.00, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.e, 0.5, orbitalElementsAccuracy);
    EXPECT_NEAR(elements.i, eps2, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, wrapToPi(elements.Omega+elements.omega), (133+113-360.)*D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.f, 123.0 * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rmag,   7730.041048693483, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rPeriap,3750.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rApoap, 11250.0, orbitalElementsAccuracy);
}

class NonCircularNearEquitorial180Degree : public ::testing::Test {
protected:
    double r[3];
    double v[3];
    double r2[3];
    double v3_2[3];
    classicElements elements;
    double eps2 = 1e-12 * 0.5;

    void SetUp() override {
        elements.a = 7500.0;
        elements.e = 0.5;
        elements.i = eps2;
        elements.Omega = 133.0 * D2R;
        elements.omega = 113.0 * D2R;
        elements.f = 123.0 * D2R;
        v3Set(7634.87141611636, 1209.24483619139, -3.20424723337984e-09, r2);
        v3Set(2.52823993598298, 6.60238615555461, -3.17592708317508e-12, v3_2);
    }
};

TEST_F(NonCircularNearEquitorial180Degree, elem2rv) {
    elem2rv(MU_EARTH, &elements, r, v);
    EXPECT_TRUE(v3IsEqualRel(r, r2, orbitalElementsAccuracy) && v3IsEqualRel(v, v3_2, orbitalElementsAccuracy));
}

TEST_F(NonCircularNearEquitorial180Degree, rv2elem) {
    rv2elem(MU_EARTH, r2, v3_2, &elements);
    EXPECT_PRED3(isEqualRel, elements.a, 7500.00, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.e, 0.5, orbitalElementsAccuracy);
    EXPECT_NEAR(elements.i, eps2, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, wrapToPi(elements.Omega+elements.omega), (133+113-360.)*D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.f, 123.0 * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rmag,   7730.041048693483, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rPeriap,3750.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rApoap, 11250.0, orbitalElementsAccuracy);
}

class CircularInclined : public ::testing::Test {
protected:
    double r[3];
    double v[3];
    double r2[3];
    double v3_2[3];
    classicElements elements;

    void SetUp() override {
        elements.a = 7500.0;
        elements.e  = 0.0;
        elements.i = 40.0 * D2R;
        elements.Omega = 133.0 * D2R;
        elements.omega = 113.0 * D2R;
        elements.f = 123.0 * D2R;
        v3Set(6343.7735859429586, 181.16597468085499, -3996.7130970223939, r2);
        v3Set(-1.8379619466304487, 6.5499717954886121, -2.6203988553352131, v3_2);
    }
};

TEST_F(CircularInclined, elem2rv) {
    elem2rv(MU_EARTH, &elements, r, v);
    EXPECT_TRUE(v3IsEqualRel(r, r2, orbitalElementsAccuracy) && v3IsEqualRel(v, v3_2, orbitalElementsAccuracy));
}

TEST_F(CircularInclined, rv2elem) {
    rv2elem(MU_EARTH, r2, v3_2, &elements);
    EXPECT_PRED3(isEqualRel, elements.a, 7500.00, orbitalElementsAccuracy);
    EXPECT_NEAR(elements.e, 0.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.i, 40. * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.Omega, 133. * D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, wrapToPi(elements.omega+elements.f), (113+123-360.)*D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rmag, 7500.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rPeriap, 7500.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rApoap, 7500.0, orbitalElementsAccuracy);
}

class CircularEquitorial : public ::testing::Test {
protected:
    double r[3];
    double v[3];
    double r2[3];
    double v3_2[3];
    classicElements elements;

    void SetUp() override {
        elements.a = 7500.0;
        elements.e  = 0.0;
        elements.i = 0.0;
        elements.Omega = 133.0 * D2R;
        elements.omega = 113.0 * D2R;
        elements.f = 123.0 * D2R;
        v3Set(7407.6625544635335, 1173.2584878017262, 0, r2);
        v3Set(-1.1404354122910105, 7.2004258117414572, 0, v3_2);
    }
};

TEST_F(CircularEquitorial, elem2rv) {
    elem2rv(MU_EARTH, &elements, r, v);
    EXPECT_TRUE(v3IsEqualRel(r, r2, orbitalElementsAccuracy) && v3IsEqualRel(v, v3_2, orbitalElementsAccuracy));
}

TEST_F(CircularEquitorial, rv2elem) {
    rv2elem(MU_EARTH, r2, v3_2, &elements);
    EXPECT_PRED3(isEqualRel, elements.a, 7500.00, orbitalElementsAccuracy);
    EXPECT_NEAR(elements.e, 0.0, orbitalElementsAccuracy);
    EXPECT_NEAR(elements.i, 0.0, orbitalElementsAccuracy);
    EXPECT_NEAR(elements.Omega, 0.0, orbitalElementsAccuracy);
    EXPECT_NEAR(elements.omega, 0.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.f, (133+113+123-360)*D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rmag, 7500.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rPeriap, 7500.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rApoap, 7500.0, orbitalElementsAccuracy);
}

class CircularEquitorialRetrograde : public ::testing::Test {
protected:
    double r[3];
    double v[3];
    double r2[3];
    double v3_2[3];
    classicElements elements;

    void SetUp() override {
        elements.a = 7500.0;
        elements.e  = 0.0;
        elements.i = M_PI;
        elements.Omega = 133.0 * D2R;
        elements.omega = 113.0 * D2R;
        elements.f = 123.0 * D2R;
        v3Set(-1687.13290757899, -7307.77548588926, 0.0, r2);
        v3Set(-7.10333318346184, 1.63993368302803, 0.0, v3_2);
    }
};

TEST_F(CircularEquitorialRetrograde, elem2rv) {
    elem2rv(MU_EARTH, &elements, r, v);
    EXPECT_TRUE(v3IsEqualRel(r, r2, orbitalElementsAccuracy) && v3IsEqualRel(v, v3_2, orbitalElementsAccuracy));
}

TEST_F(CircularEquitorialRetrograde, rv2elem) {
    rv2elem(MU_EARTH, r2, v3_2, &elements);
    EXPECT_PRED3(isEqualRel, elements.a, 7500.00, orbitalElementsAccuracy);
    EXPECT_NEAR(elements.e, 0.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.i, M_PI, orbitalElementsAccuracy);
    EXPECT_NEAR(elements.Omega, 0.0, orbitalElementsAccuracy);
    EXPECT_NEAR(elements.omega, 0.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.f, (-133+113+123)*D2R, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rmag, 7500.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rPeriap, 7500.0, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements.rApoap, 7500.0, orbitalElementsAccuracy);
}

TEST(OrbitalMotion, classicElementsToMeanElements) {
    classicElements elements;
    elements.a     = 1000.0;
    elements.e     = 0.2;
    elements.i     = 0.2;
    elements.Omega = 0.15;
    elements.omega = 0.5;
    elements.f     = 0.2;
    double req = 300.0;
    double J2 = 1e-3;
    classicElements elements_p;
    clMeanOscMap(req, J2, &elements, &elements_p, 1);
    EXPECT_PRED3(isEqualRel, elements_p.a, 1000.07546442015950560744386166334152, orbitalElementsAccuracy);
    EXPECT_NEAR(elements_p.e, 0.20017786852908628358882481279579, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements_p.i, 0.20000333960738947425284095515963, orbitalElementsAccuracy);
    EXPECT_NEAR(elements_p.Omega, 0.15007256499303692209856819772540, orbitalElementsAccuracy);
    EXPECT_NEAR(elements_p.omega, 0.50011857315729335571319325026707, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements_p.f, 0.19982315726261962174348241205735, orbitalElementsAccuracy);
}

TEST(OrbitalMotion, classicElementsToEquinoctialElements) {
    classicElements elements;
    elements.a     = 1000.0;
    elements.e     = 0.2;
    elements.i     = 0.2;
    elements.Omega = 0.15;
    elements.omega = 0.5;
    elements.f     = 0.2;
    equinoctialElements elements_eq;
    clElem2eqElem(&elements, &elements_eq);

    EXPECT_PRED3(isEqualRel, elements_eq.a, 1000.00000000000000000000000000000000, orbitalElementsAccuracy);
    EXPECT_NEAR(elements_eq.P1, 0.12103728114720790909331071816268, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements_eq.P2, 0.15921675970981119530023306651856, orbitalElementsAccuracy);
    EXPECT_NEAR(elements_eq.Q1, 0.01499382601880069713906618034116, orbitalElementsAccuracy);
    EXPECT_NEAR(elements_eq.Q2, 0.09920802187229026125603326136115, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements_eq.l, 0.78093005232114087732497864635661, orbitalElementsAccuracy);
    EXPECT_PRED3(isEqualRel, elements_eq.L, 0.85000000000000008881784197001252, orbitalElementsAccuracy);
}
