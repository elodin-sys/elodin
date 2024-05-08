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
#include "fswAlgorithms/attGuidance/waypointReference/waypointReference.h"
#include <sstream>
#include <string>
#include <string.h>
#include "architecture/utilities/avsEigenSupport.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/macroDefinitions.h"

/*! This is the constructor for the module class.  It sets default variable
    values and initializes the various parts of the model */
WaypointReference::WaypointReference()
{
    this->dataFileName = "";
    this->delimiter = ",";
    this->headerLines = 0;
    this->attitudeType = 0;
	this->useReferenceFrame = false;
	this->endOfFile = false;

    return;
}

/*! Module Destructor.  */
WaypointReference::~WaypointReference()
{
    /* close the data file if it is open */
    if(this->fileHandle->is_open()) {
        this->fileHandle->close();
        bskLogger.bskLog(BSK_INFORMATION, "WaypointReference:\nclosed the file: %s.", this->dataFileName.c_str());
    }

    return;
}


/*! A Reset method to put the module back into a clean state
 @param CurrentSimNanos The current sim time in nanoseconds
 */
void WaypointReference::Reset(uint64_t CurrentSimNanos)
{	
    if (this->dataFileName.length() == 0) {
        bskLogger.bskLog(BSK_ERROR, "WaypointReference: dataFileName must be an non-empty string.");
    }

    /* open the data file*/
    this->fileHandle = new std::ifstream(this->dataFileName);
    if (this->fileHandle->fail()) {
        bskLogger.bskLog(BSK_ERROR, "WaypointReference: was not able to load the file %s.", this->dataFileName.c_str());
    }

	/* read and bypass header line(s) */
	int count = 0;
	std::string line;
    while (count < this->headerLines) {
		count += 1;
        getline(*this->fileHandle, line);
    }

    /* Pull first line of dataFileName and stores relative time and attitude in this->t_a and this->attRefMsg_a */
	pullDataLine(&this->t_a, &this->attRefMsg_a);
	/* Pull first line of dataFileName and stores relative time and attitude in this->t_b and this->attRefMsg_b */
	pullDataLine(&this->t_b, &this->attRefMsg_b);
 
    bskLogger.bskLog(BSK_INFORMATION, "WaypointReference:\nloaded the file: %s.", this->dataFileName.c_str());

    return;
}


/*! Update this module at the task rate
 @param CurrentSimNanos The current sim time
 */
void WaypointReference::UpdateState(uint64_t CurrentSimNanos)
{
    /* ensure that a file was opened */
    if (this->fileHandle->is_open()) {
        /* read in next line*/
        std::string line;
		
	    /* create the attitude output message buffer */
		AttRefMsgPayload attMsgBuffer;
		
		/* zero output message */
        attMsgBuffer = this->attRefOutMsg.zeroMsgPayload;
		
		/* current time */
		uint64_t t =  CurrentSimNanos;
		
		/* for CurrentTime < t_0 hold initial attitude with zero angular rates and accelerations */
		if (t < this->t_a) {
			v3Copy(this->attRefMsg_a.sigma_RN, attMsgBuffer.sigma_RN);
		}
	    else {
			/* while CurrentTime > t_b read next line and update t_a, t_b, attRefMsg_a and attRefMsg_b */
			while (t > this->t_b && this->endOfFile == false) {
				this->t_a = this->t_b;
			    this->attRefMsg_a = this->attRefMsg_b;
				pullDataLine(&this->t_b, &this->attRefMsg_b);
			}
			/* if t_a <= CurrentTime <= t_b interpolate between attRefMsg_a and attRefMsg_b */
			if (t >= this->t_a && t <= this->t_b) {
				/* check the norm of the MRP difference between two consecutive waypoints */
				double deltaSigma[3];
			    v3Subtract(this->attRefMsg_b.sigma_RN, this->attRefMsg_a.sigma_RN, deltaSigma);
                double normDeltaSigma = v3Norm(deltaSigma);
				/* if norm <= 1 interpolate between waypoints */
				if (normDeltaSigma <= 1) {
					linearInterpolation(this->t_a, this->attRefMsg_a.sigma_RN, this->t_b, this->attRefMsg_b.sigma_RN, t, &attMsgBuffer.sigma_RN[0]);
				}
				/* if norm > 1 interpolate between waypoint a and shadow set of waypoint b */
				else {
					double sigma_RN_b_S[3];
                    MRPshadow(this->attRefMsg_b.sigma_RN, sigma_RN_b_S);
					linearInterpolation(this->t_a, this->attRefMsg_a.sigma_RN, this->t_b, sigma_RN_b_S, t, &attMsgBuffer.sigma_RN[0]);
				}
                linearInterpolation(this->t_a, this->attRefMsg_a.omega_RN_N, this->t_b, this->attRefMsg_b.omega_RN_N, t, &attMsgBuffer.omega_RN_N[0]);
                linearInterpolation(this->t_a, this->attRefMsg_a.domega_RN_N, this->t_b, this->attRefMsg_b.domega_RN_N, t, &attMsgBuffer.domega_RN_N[0]);
			}
		}
		/* for CurrentTime > t_b and t_b is the last time step in file, hold final attitude with zero angular rates and accelerations */
		if (t > this->t_b && this->endOfFile == true) {
			v3Copy(this->attRefMsg_b.sigma_RN, attMsgBuffer.sigma_RN);
		}
	
		/* write output attitude reference message */
        this->attRefOutMsg.write(&attMsgBuffer, this->moduleID, CurrentSimNanos);
		
    }
    return;
}


/*! Pull one line of dataFileName and stores time t and relative attitude in attRefMsg_t */
void WaypointReference::pullDataLine(uint64_t *t, AttRefMsgPayload *attRefMsg_t)
{   
    std::string line;
	
	/* read in next line, if line is not empty, stores the information ;
	   if line is empty, switches this_endOfFile to true */
	if (getline(*this->fileHandle, line)) {
		std::istringstream iss(line);
		
		*attRefMsg_t = this->attRefOutMsg.zeroMsgPayload;

        /* pull time, this is not used in the BSK msg */
        *t = (uint64_t) (pullScalar(&iss) * SEC2NANO);
		
		/* get inertial attitude of reference frame R with respect to N and store in msg */
		double attNorm;
		double att3[3];
		double att4[4];
		double att4Norm[4];
		switch (this->attitudeType) {
			case 0:
			    /* 3D attitude coordinate set */
				/* if MRP norm <= 1 save the MRP set immediately,
				   if not map to the shadow set and saves the shadow set */
                pullVector(&iss, att3);
				attNorm = v3Norm(att3);
				if (attNorm <= 1) {
					v3Copy(att3, attRefMsg_t->sigma_RN);
				}
				else {
					MRPshadow(att3, attRefMsg_t->sigma_RN);
				}
				break;
			case 1:
			    /* 4D attitude coordinate set (q0, q1, q2, q3) */
                pullVector4(&iss, att4);
				vNormalize(att4, 4, att4Norm);
				EP2MRP(att4Norm, attRefMsg_t->sigma_RN);
				break;
			case 2:
			    /* 4D attitude coordinate set (q1, q2, q3, qs) */
			    double attBuffer[4];
                pullVector4(&iss, attBuffer);
				att4[0] = attBuffer[3];
				att4[1] = attBuffer[0];
				att4[2] = attBuffer[1];
				att4[3] = attBuffer[2];
				vNormalize(att4, 4, att4Norm);
				EP2MRP(att4Norm, attRefMsg_t->sigma_RN);
				break;
			default:
			    bskLogger.bskLog(BSK_ERROR, "WaypointReference: the attitude type provided is invalid.");
		}

		if (this->useReferenceFrame == false) {
			/* get inertial angular rates in inertial frame components and store them in msg */
		    pullVector(&iss, attRefMsg_t->omega_RN_N);
			
			/* get inertial angular accelerations in inertial frame components and store them in msg */
		    pullVector(&iss, attRefMsg_t->domega_RN_N);
			
		}
		else {
			/* get inertial angular rates in reference frame components */
			double omega_RN_R[3];
		    pullVector(&iss, omega_RN_R);
			
			/* get inertial angular accelerations in reference frame components */
			double omegaDot_RN_R[3];
		    pullVector(&iss, omegaDot_RN_R);
			
			/* compute direction cosine matrix [RN] */
			double RN[3][3];
		    MRP2C(attRefMsg_t->sigma_RN, RN);
			
			/* change angular rates and accelerations to inertial frame and stores them in msg */
			v3tMultM33(omega_RN_R, RN, attRefMsg_t->omega_RN_N);
		    v3tMultM33(omegaDot_RN_R, RN, attRefMsg_t->domega_RN_N);
		}
			
    } else {
        bskLogger.bskLog(BSK_INFORMATION, "WaypointReference: reached end of file.");
		this->endOfFile = true;
    }
}


/*! pull a 3-d set of double values from the input stream
 */
void WaypointReference::pullVector(std::istringstream *iss, double vec[3]) {
    double x,y,z;
    x = pullScalar(iss);
    y = pullScalar(iss);
    z = pullScalar(iss);
    v3Set(x, y, z, vec);
}

/*! pull a 4-d set of double values from the input stream
 */
void WaypointReference::pullVector4(std::istringstream *iss, double *vec) {
    double q0, q1, q2, q3;
    q0 = pullScalar(iss);
    q1 = pullScalar(iss);
    q2 = pullScalar(iss);
    q3 = pullScalar(iss);
    v4Set(q0, q1, q2, q3, vec);
}

/*! pull a double from the input stream
*/
double WaypointReference::pullScalar(std::istringstream *iss) {
    const char delimiterString = *this->delimiter.c_str();
    std::string item;

    getline(*iss, item, delimiterString);

    return stod(item);
}

/*! linearly interpolate between two vectors v_a and v_b
*/
void WaypointReference::linearInterpolation(uint64_t t_a, double v_a[3], uint64_t t_b, double v_b[3], uint64_t t, double *v) {
	for (int i = 0; i < 3; i++) {
        *(v+i) = v_a[i] + (v_b[i] - v_a[i]) / (t_b - t_a) * (t - t_a);
    }
}

