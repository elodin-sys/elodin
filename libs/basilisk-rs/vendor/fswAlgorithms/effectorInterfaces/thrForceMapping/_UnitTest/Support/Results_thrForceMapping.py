''' '''
import numpy as np


class Results_thrForceMapping():
    def __init__(self, Lr, COrig, COM, rData, gData, thrForceSign, thrForceMag, angErrThresh, numThrusters, epsilon, use2ndLoop):
        self.rData = np.array(rData)
        self.gData = np.array(gData)
        self.Lr_B = np.array(Lr) #Original Requested Torque in B Frame
        self.COM = COM

        self.thrForceSign = thrForceSign # -1 is DV thrusters; +1 is RCS thrusters
        self.thrForceMag = thrForceMag # specifies the max thrust for each thruster
        self.angErrThresh = angErrThresh # Determines when the thrusters are considered saturated

        self.numThrusters = numThrusters # number of explicitly configured thrusters

        self.C = np.array(COrig) # Control "Frame" (could be 1, 2, or 3 axii controllable)
        self.C = np.reshape(self.C, ((len(self.C)//3),3),'C')

        self.epsilon = epsilon
        self.use2ndLoop = use2ndLoop
        return

    def results_thrForceMapping(self):
        # Produce the forces with all thrusters included
        Lr_offset = [0.0, 0.0, 0.0]
        CT = np.transpose(self.C)

        # Compute D Matrix and Determine Force
        D = np.zeros((3,len(self.rData)))
        for i in range(len(self.rData)):
            D[:,i] = np.cross((self.rData[i,:] - self.COM), self.gData[i,:])
            if(self.thrForceSign < 0):
                Lr_offset -= self.thrForceMag[i]*D[:,i]

        self.Lr_B = self.Lr_B + Lr_offset
        Lr_Bar = np.dot(self.C, self.Lr_B)
        F = self.mapToForce(D, Lr_Bar)

        # Subtract off minimum force (remove null space contribution)
        if self.thrForceSign > 0:
            #F = self.subtractPairwiseNullSpace(F, D)
            F = self.subtractMin(F, self.numThrusters)

        # Identify any negative forces
        t = (F[:]*self.thrForceSign > self.epsilon)

        # Recompute the D Matrix with negative forces removed and compute Force
        if self.thrForceSign < 0 or self.use2ndLoop:
            DNew = np.array([])
            for i in range(0,len(F)):
                if t[i]:
                    DNew = np.append(DNew, np.cross((self.rData[i,:] - self.COM), self.gData[i]))
            DNew = np.reshape(DNew, (3, (len(DNew) // 3)), 'F')
            FNew = self.mapToForce(DNew, Lr_Bar)
            if (self.thrForceSign > 0):
                FNew = self.subtractMin(FNew,len(DNew[0])) # Produced negative forces when doing 2nd loop, dropped thruster, and COM offset
            # Remove minumum force
            count = 0
            for i in range(0,len(F)):
                if t[i]:
                    F[i] = FNew[count]
                    count += 1
                else:
                    F[i] = 0.0
        else:
            DNew = D
        angle = self.results_computeAngErr(D, Lr_Bar, F)

        if angle > self.angErrThresh:

            maxFractUse = 0.0
            for i in range(0, self.numThrusters):
                if self.thrForceMag[i] > 0 and abs(F[i])/self.thrForceMag[i] > maxFractUse:
                    maxFractUse = abs(F[i])/self.thrForceMag[i]
            if maxFractUse > 1.0:
                F = F/maxFractUse
                angleErr = self.results_computeAngErr(D, Lr_Bar, F)

        return F, DNew

    def results_computeAngErr(self, D, BLr_B, F):
        returnAngle = 0.0
        DT = np.transpose(D)

        if np.linalg.norm(BLr_B) > 10 ** -9:
            tauActual_B = [0.0, 0.0, 0.0]
            BLr_B_hat = BLr_B / np.linalg.norm(BLr_B)
            for i in range(0, self.numThrusters):
                if abs(F[i]) < self.thrForceMag[i]:
                    thrForce = F[i]
                else:
                    thrForce = self.thrForceMag[i] * abs(F[i]) / F[i]

                LrEffector_B = thrForce * DT[i, :]
                tauActual_B += LrEffector_B

            tauActual_B = tauActual_B / np.linalg.norm(tauActual_B)

            if np.dot(BLr_B_hat, tauActual_B) < 1.0:
                returnAngle = np.arccos(np.dot(BLr_B_hat, tauActual_B))

        return returnAngle

    def numRelEqualElements(self, array1, array2, accuracy):
        count = 0
        for i in range(3):
            if abs(array1[i] - array2[i]) < accuracy:
                count += 1
        return count

    def mapToForce(self, D, Lr_Bar):
        numControlAxes = 0
        for i in range(0, len(self.C[0])):
            if not np.array_equal(self.C[:, i], [0.0, 0.0, 0.0]):
                numControlAxes = numControlAxes + 1
        numThr = 0
        for i in range(0, len(D[0])):
            if not np.array_equal(D[:, i], [0.0, 0.0, 0.0]):
                numThr = numThr + 1
        D = np.matmul(self.C, D)
        DT = np.transpose(D)
        DDT = np.eye(3)
        for i in range(0, numControlAxes):
            for j in range(0, numControlAxes):
                DDT[i][j] = 0.0
                for k in range(0, numThr):
                    DDT[i][j] += D[i][k] * D[j][k]
        try:
            DDTInv = np.linalg.inv(DDT)
            if np.linalg.det(DDT) < self.epsilon:
                raise np.linalg.LinAlgError()
        except:
            DDTInv = np.zeros((3, 3))
            print("Singular Matrix! Outputting Zeros.")

        DDTInvLr_Bar = np.dot(DDTInv, Lr_Bar)
        F = np.dot(DT, DDTInvLr_Bar)
        return F

    def subtractPairwiseNullSpace(self, F, D):

        for i in range(self.numThrusters):
            if F[i] < 0.0:
                for j in range(self.numThrusters):
                    if (np.allclose(D[:, i], D[:, j], atol=self.epsilon) and i != j):
                        F[j] -= F[i]
                        break
                F[i] = 0.0

        return F

    def subtractMin(self, F, size):
        minValue = 0.0
        for i in range(size):
            if F[i] < minValue:
                minValue = F[i]

        for i in range(size):
            F[i] -= minValue

        return F
