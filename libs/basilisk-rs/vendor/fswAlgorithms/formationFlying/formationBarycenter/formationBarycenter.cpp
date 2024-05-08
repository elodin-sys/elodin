/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado Boulder

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


#include "fswAlgorithms/formationFlying/formationBarycenter/formationBarycenter.h"
#include "architecture/utilities/orbitalMotion.h"
#include "architecture/msgPayloadDefC/ClassicElementsMsgPayload.h"
#include <math.h>


/*! This is the constructor for the module class.  It sets default variable
    values and initializes the various parts of the model */
FormationBarycenter::FormationBarycenter() {
    this->useOrbitalElements = false;
    this->mu = 0;
}

/*! Module Destructor */
FormationBarycenter::~FormationBarycenter() {
}

/*! This method self initializes the C-wrapped output message.
*/
void FormationBarycenter::SelfInit()
{
    NavTransMsg_C_init(&this->transOutMsgC);
}



/*! This method is used to reset the module and checks that required input messages are connected.
*/
void FormationBarycenter::Reset(uint64_t CurrentSimNanos) {
    // check that required input messages are connected
    if (this->scNavInMsgs.size() == 0 || this->scPayloadInMsgs.size() == 0) {
        bskLogger.bskLog(BSK_ERROR, "FormationBarycenter module must have at least one spacecraft added through `addSpacecraftToModel`");
    }

    // check if the gravitational parameter is set if using orbital elements averaging
    if (this->mu == 0 && this->useOrbitalElements) {
        bskLogger.bskLog(BSK_ERROR, "FormationBarycenter module requires defining a gravitational parameter if using orbital elements.");
    }

}

/*! Adds a scNav and scPayload messages name to the vector of names to be subscribed to.
*/
void FormationBarycenter::addSpacecraftToModel(Message<NavTransMsgPayload>* tmpScNavMsg, Message<VehicleConfigMsgPayload>* tmpScPayloadMsg) {
    this->scNavInMsgs.push_back(tmpScNavMsg->addSubscriber());
    this->scPayloadInMsgs.push_back(tmpScPayloadMsg->addSubscriber());
}

/*! Reads the input messages
*/
void FormationBarycenter::ReadInputMessages() {

    NavTransMsgPayload scNavMsg;
    VehicleConfigMsgPayload scPayloadMsg;

    // clear out the vector of spacecraft navigation and mass messages.  This is created freshly below.
    this->scNavBuffer.clear();
    this->scPayloadBuffer.clear();

    // read in the spacecraft state messages
    for (long unsigned int c = 0; c < this->scNavInMsgs.size(); c++) {
        scNavMsg = this->scNavInMsgs.at(c)();
        scPayloadMsg = this->scPayloadInMsgs.at(c)();
        this->scNavBuffer.push_back(scNavMsg);
        this->scPayloadBuffer.push_back(scPayloadMsg);
    }

}

/*! Does the barycenter calculations
*/
void FormationBarycenter::computeBaricenter() {
    //create temporarary variables
    double barycenter[] {0, 0, 0};
    double barycenterVelocity[] {0, 0, 0};
    double totalMass {0};

    // check which averaging to use
    if (!this->useOrbitalElements) {
        // compute the cartesian barycenter
        for (long unsigned int c = 0; c < this->scNavInMsgs.size(); c++) {
            for (int n = 0; n < 3; n++) {
                barycenter[n] += this->scPayloadBuffer.at(c).massSC * this->scNavBuffer.at(c).r_BN_N[n];
                barycenterVelocity[n] += this->scPayloadBuffer.at(c).massSC * this->scNavBuffer.at(c).v_BN_N[n];
            }
            totalMass += this->scPayloadBuffer.at(c).massSC;
        }

        for (int n = 0; n < 3; n++) {
            barycenter[n] /= totalMass;
            barycenterVelocity[n] /= totalMass;
        }
    } else {
        classicElements orbitElements = {}; // zero the orbit elements first
        classicElements tempElements;
        double OmegaSineSum = 0;
        double OmegaCosineSum = 0;
        double omegaSineSum = 0;
        double omegaCosineSum = 0;
        double fSineSum = 0;
        double fCosineSum = 0;

        // compute the orbital elements barycenter
        for (long unsigned int c = 0; c < this->scNavInMsgs.size(); c++) {
            // convert the position and velocity vectors into orbital elements
            rv2elem(this->mu, this->scNavBuffer.at(c).r_BN_N, this->scNavBuffer.at(c).v_BN_N, &tempElements);

            orbitElements.a += this->scPayloadBuffer.at(c).massSC * tempElements.a;
            orbitElements.e += this->scPayloadBuffer.at(c).massSC * tempElements.e;
            orbitElements.i += this->scPayloadBuffer.at(c).massSC * tempElements.i;
            
            OmegaSineSum += this->scPayloadBuffer.at(c).massSC * sin(tempElements.Omega);
            OmegaCosineSum += this->scPayloadBuffer.at(c).massSC * cos(tempElements.Omega);
            omegaSineSum += this->scPayloadBuffer.at(c).massSC * sin(tempElements.omega);
            omegaCosineSum += this->scPayloadBuffer.at(c).massSC * cos(tempElements.omega);
            fSineSum += this->scPayloadBuffer.at(c).massSC * sin(tempElements.f);
            fCosineSum += this->scPayloadBuffer.at(c).massSC * cos(tempElements.f);

            totalMass += this->scPayloadBuffer.at(c).massSC;

        }

        orbitElements.a /= totalMass;
        orbitElements.e /= totalMass;
        orbitElements.i /= totalMass;
        orbitElements.Omega = atan2(OmegaSineSum, OmegaCosineSum);
        orbitElements.omega = atan2(omegaSineSum, omegaCosineSum);
        orbitElements.f = atan2(fSineSum, fCosineSum);

        // convert orbital elements into position and velocity vectors
        elem2rv(this->mu, &orbitElements, barycenter, barycenterVelocity);
    }

    // save the information to the output buffer
    for (int n = 0; n < 3; n++) {
        this->transOutBuffer.r_BN_N[n] = barycenter[n];
        this->transOutBuffer.v_BN_N[n] = barycenterVelocity[n];
    }
}

/*! writes the output messages
*/
void FormationBarycenter::WriteOutputMessage(uint64_t CurrentClock) {
    // write C++ output message
    this->transOutMsg.write(&this->transOutBuffer, this->moduleID, CurrentClock);

    // write C output message
    NavTransMsg_C_write(&this->transOutBuffer, &this->transOutMsgC, this->moduleID, CurrentClock);
}

/*! This is the main method that gets called every time the module is updated.
*/
void FormationBarycenter::UpdateState(uint64_t CurrentSimNanos)
{
    this->ReadInputMessages();
    this->transOutBuffer = this->transOutMsg.zeroMsgPayload; // zero the output message buffer
    this->computeBaricenter();
    this->WriteOutputMessage(CurrentSimNanos);
}

