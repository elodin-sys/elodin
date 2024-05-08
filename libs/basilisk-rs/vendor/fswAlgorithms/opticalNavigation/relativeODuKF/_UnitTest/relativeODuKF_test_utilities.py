#
#  ISC License
#
#  Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
import inspect
import os

import numpy as np
from Basilisk.utilities import unitTestSupport

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
splitPath = path.split('fswAlgorithms')

import matplotlib.pyplot as plt

color_x = 'dodgerblue'
color_y = 'salmon'
color_z = 'lightgreen'
m2km = 1.0 / 1000.0

def StatePlot(x, testName, show_plots):

    numStates = len(x[0,:])-1

    t= np.zeros(len(x[:,0]))
    for i in range(len(t)):
        t[i] = x[i, 0]*1E-9

    plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(321)
    plt.plot(t , x[:, 1], "b", label='Error Filter')
    plt.legend(loc='lower right')
    plt.title('First pos component (m)')
    plt.grid()

    plt.subplot(322)
    plt.plot(t , x[:, 4], "b")
    plt.title('Second rate component (m/s)')
    plt.grid()

    plt.subplot(323)
    plt.plot(t , x[:, 2], "b")
    plt.title('Second pos component (m)')
    plt.grid()

    plt.subplot(324)
    plt.plot(t , x[:, 5], "b")
    plt.xlabel('t(s)')
    plt.title('Third rate component (m/s)')
    plt.grid()

    plt.subplot(325)
    plt.plot(t , x[:, 3], "b")
    plt.xlabel('t(s)')
    plt.title('Third pos component (m)')
    plt.grid()

    plt.subplot(326)
    plt.plot(t , x[:, 6], "b")
    plt.xlabel('t(s)')
    plt.title('Third rate component (m/s)')
    plt.grid()

    unitTestSupport.writeFigureLaTeX('StatesPlot' + testName, 'State error', plt, 'height=0.9\\textwidth, keepaspectratio', path)
    if show_plots:
        plt.show()
    plt.close()

def EnergyPlot(t, energy, testName, show_plots):

    conserved= np.zeros(len(t))
    for i in range(len(t)):
        conserved[i] = (energy[i] - energy[0])/energy[0]

    plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(t , conserved, "b", label='Energy')
    plt.legend(loc='lower right')
    plt.title('Energy ' + testName)
    plt.grid()


    unitTestSupport.writeFigureLaTeX('Energy' + testName, 'Orbital Energy', plt, 'height=0.9\\textwidth, keepaspectratio', path)
    if show_plots:
        plt.show()
    plt.close()


def StateCovarPlot(x, Pflat, testName, show_plots):

    numStates = len(x[0,:])-1

    P = np.zeros([len(Pflat[:,0]),numStates,numStates])
    t= np.zeros(len(Pflat[:,0]))
    for i in range(len(Pflat[:,0])):
        t[i] = x[i, 0]*1E-9
        P[i,:,:] = Pflat[i,1:(numStates*numStates+1)].reshape([numStates,numStates])


    plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(321)
    plt.plot(t , x[:, 1], "b", label='Error Filter')
    plt.plot(t , x[:, 1]+3 * np.sqrt(P[:, 0, 0]), 'r--',  label='Covar Filter')
    plt.plot(t , x[:, 1]-3 * np.sqrt(P[:, 0, 0]), 'r--')
    plt.legend(loc='lower right')
    plt.title('First pos component (m)')
    plt.grid()

    plt.subplot(322)
    plt.plot(t , x[:, 4], "b")
    plt.plot(t , x[:, 4]+3 * np.sqrt(P[:, 3, 3]), 'r--')
    plt.plot(t , x[:, 4]-3 * np.sqrt(P[:, 3, 3]), 'r--')
    plt.title('Second rate component (m/s)')
    plt.grid()

    plt.subplot(323)
    plt.plot(t , x[:, 2], "b")
    plt.plot(t , x[:, 2]+3 * np.sqrt(P[:, 1, 1]), 'r--')
    plt.plot(t , x[:, 2]-3 * np.sqrt(P[:, 1, 1]), 'r--')
    plt.title('Second pos component (m)')
    plt.grid()

    plt.subplot(324)
    plt.plot(t , x[:, 5], "b")
    plt.plot(t , x[:, 5]+3 * np.sqrt(P[:, 4, 4]), 'r--')
    plt.plot(t , x[:, 5]-3 * np.sqrt(P[:, 4, 4]), 'r--')
    plt.xlabel('t(s)')
    plt.title('Third rate component (m/s)')
    plt.grid()

    plt.subplot(325)
    plt.plot(t , x[:, 3], "b")
    plt.plot(t , x[:, 3]+3 * np.sqrt(P[:, 2, 2]), 'r--')
    plt.plot(t , x[:, 3]-3 * np.sqrt(P[:, 2, 2]), 'r--')
    plt.xlabel('t(s)')
    plt.title('Third pos component (m)')
    plt.grid()

    plt.subplot(326)
    plt.plot(t , x[:, 6], "b")
    plt.plot(t , x[:, 6]+3 * np.sqrt(P[:, 5, 5]), 'r--')
    plt.plot(t , x[:, 6]-3 * np.sqrt(P[:, 5, 5]), 'r--')
    plt.xlabel('t(s)')
    plt.title('Third rate component (m/s)')
    plt.grid()

    unitTestSupport.writeFigureLaTeX('StatesPlot' + testName, 'State error and covariance', plt, 'height=0.9\\textwidth, keepaspectratio', path)
    if show_plots:
        plt.show()
    plt.close()



def PostFitResiduals(Res, noise, testName, show_plots):

    MeasNoise = np.zeros(len(Res[:,0]))
    t= np.zeros(len(Res[:,0]))
    for i in range(len(Res[:,0])):
        t[i] = Res[i, 0]*1E-9
        MeasNoise[i] = 3*noise
        # Don't plot zero values, since they mean that no measurement is taken
        for j in range(len(Res[0,:])-1):
            if -1E-10 < Res[i,j+1] < 1E-10:
                Res[i, j+1] = np.nan

    plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(311)
    plt.plot(t , Res[:, 1], "b.", label='Residual')
    plt.plot(t , MeasNoise, 'r--', label='Covar')
    plt.plot(t , -MeasNoise, 'r--')
    plt.legend(loc='lower right')
    plt.ylim([-10*noise, 10*noise])
    plt.title('First Meas Comp (m)')
    plt.grid()

    plt.subplot(312)
    plt.plot(t , Res[:, 2], "b.")
    plt.plot(t , MeasNoise, 'r--')
    plt.plot(t , -MeasNoise, 'r--')
    plt.ylim([-10*noise, 10*noise])
    plt.title('Second Meas Comp (m)')
    plt.grid()

    plt.subplot(313)
    plt.plot(t , Res[:, 3], "b.")
    plt.plot(t , MeasNoise, 'r--')
    plt.plot(t , -MeasNoise, 'r--')
    plt.ylim([-10*noise, 10*noise])
    plt.title('Third Meas Comp (m)')
    plt.grid()


    unitTestSupport.writeFigureLaTeX('PostFit' + testName, 'Post Fit Residuals', plt, 'height=0.9\\textwidth, keepaspectratio', path)

    if show_plots:
        plt.show()
    plt.close()

def plot_TwoOrbits(r_BN, r_BN2):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('$R_x$, km')
    ax.set_ylabel('$R_y$, km')
    ax.set_zlabel('$R_z$, km')
    ax.plot(r_BN[:, 1] * m2km, r_BN[:, 2] * m2km, r_BN[:, 3] * m2km, color_x, label="True orbit")
    for i in range(len(r_BN2[:,0])):
        if np.abs(r_BN2[i, 1])>0 or np.abs(r_BN2[i, 2])>0:
            ax.scatter(r_BN2[i, 1] * m2km, r_BN2[i, 2] * m2km, r_BN2[i, 3] * m2km, color=color_y, label="Meas orbit")
    ax.scatter(0, 0, color='r')
    ax.set_title('Spacecraft Orbits')
    return