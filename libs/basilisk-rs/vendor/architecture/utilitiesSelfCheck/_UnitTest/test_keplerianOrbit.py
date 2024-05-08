
# ISC License
#
# Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.




#
# Keplerian OrbitUnit Test
#
# Purpose:  Tests an object-oriented Keplerian Orbit Object
# Author:   Scott Carnahan
# Creation Date:  Sept 10 2019
#

import inspect
import os
from copy import copy

import numpy as np
from Basilisk.architecture import keplerianOrbit
from Basilisk.utilities import orbitalMotion

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

def test_unitKeplerianOrbit(show_plots=False):
    """Module Unit Test"""
    [testResults, testMessage] = unitKeplerianOrbit(show_plots)
    assert testResults < 1, testMessage


def unitKeplerianOrbit(show_plots=False):
    """
    Unit Test Keplerian Orbit object

    This test covers:
    1) elem to rv conversion
    2) orbit energy calculation
    3) orbital period calculation
    4) mean motion calculation
    5) changing individual orbital elements
    6) copy constructor
    7) constructor with arguments
    8) constructor without arguments

    Parameters
    ----------
    show_plots : bool
        unused. required for Basilisk test suite

    Returns
    -------
    testFailCount : int
        number of tests that failed
    testMessages : str
        a string of all of the failure messages
    """
    testFailCount = 0
    testMessages = []

    # constructor without arguments
    orb = keplerianOrbit.KeplerianOrbit()
    assert orb.a() == 100000.
    if not orb.a() == 100000.:
        testFailCount += 1
        testMessages.append('default constructor failure')

    # constructor with arguments
    oe = orb.oe()
    orb2 = keplerianOrbit.KeplerianOrbit(oe, orbitalMotion.MU_EARTH)
    assert orb2.r_BP_P() == orb.r_BP_P()
    if not orb2.r_BP_P() == orb.r_BP_P():
        testFailCount += 1
        testMessages.append('Argumented constructor failure')

    # copy constructor
    orb3 = keplerianOrbit.KeplerianOrbit(orb2)
    assert orb2.v_BP_P() == orb3.v_BP_P()
    if not orb2.v_BP_P() == orb3.v_BP_P():
        testFailCount += 1
        testMessages.append('Copy Constructor Failure')
    try:
        orb4 = copy(orb3)
    except:
        assert False
        testFailCount += 1
        testMessages.append('python copy not working')

    # changing orbital elements
    orb3.set_f(0.0)
    init_r = orb3.r()
    orb3.set_f(1.)
    assert init_r != orb3.r()
    if init_r == orb3.r():
        testFailCount += 1
        testMessages.append('Failure to change element')
    orb3.set_f(0.0)
    assert init_r == orb3.r()
    if init_r != orb3.r():
        testFailCount += 1
        testMessages.append('Failure to change element')

    # mean motion calc
    expected_n = np.sqrt(orbitalMotion.MU_EARTH / orb3.a()**3)
    assert orb3.n() == expected_n
    if not orb3.n() == expected_n:
        testFailCount += 1
        testMessages.append('Bad mean motion calc')

    # orbital period calc
    assert orb3.P() == 2 * np.pi / expected_n
    if not orb3.P() == 2 * np.pi / expected_n:
        testFailCount += 1
        testMessages.append('Bad period calc')

    # orbital energy calc
    expected_E = -orbitalMotion.MU_EARTH / 2.0 / orb3.a()
    assert orb3.Energy() == expected_E
    if not orb3.Energy() == expected_E:
        testFailCount += 1
        testMessages.append('Bad energy calc')

    # rv calc
    expected_r, expected_v = orbitalMotion.elem2rv(orbitalMotion.MU_EARTH, orb3.oe())
    dist = np.linalg.norm(np.array(orb3.r_BP_P()).flatten() - expected_r)
    assert dist == 0.0
    if not dist == 0.0:
        testFailCount += 1
        testMessages.append('RV conversion failure')

    return [testFailCount, ''.join(testMessages)]

if __name__ == "__main__":
    unitKeplerianOrbit()