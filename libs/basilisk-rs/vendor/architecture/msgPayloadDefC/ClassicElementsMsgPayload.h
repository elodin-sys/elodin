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

#ifndef _CLASSIC_ELEMENTS_H
#define _CLASSIC_ELEMENTS_H


/*! @brief Structure used to define classic orbit elements */
typedef struct {
    double a;         //!< object semi-major axis
    double e;         //!< Eccentricity of the orbit
    double i;         //!< inclination of the orbital plane
    double Omega;     //!< Right ascension of the ascending node
    double omega;     //!< Argument of periapsis of the orbit
    double f;         //!< True anomaly of the orbit
    double rmag;      //!< Magnitude of the position vector (extra)
    double alpha;     //!< Inverted semi-major axis (extra)
    double rPeriap;   //!< Radius of periapsis (extra)
    double rApoap;    //!< Radius if apoapsis (extra)
} ClassicElementsMsgPayload;

typedef ClassicElementsMsgPayload classicElements;


#endif
