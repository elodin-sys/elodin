/*
 ISC License

 Copyright (c) 2022, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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
#include "fswAlgorithms/attGuidance/constrainedAttitudeManeuver/constrainedAttitudeManeuver.h"
#include <map>
#include <sstream>
#include <string>
#include <string.h>
#include <math.h>
#include "architecture/utilities/avsEigenSupport.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/macroDefinitions.h"


/*! This is the constructor for the Node class.  It sets default variable
    values and initializes the various parts of the model */
Node::Node()
{
    return;
}

/*! The constructor requires the MRP set, the constraint structure and the sc boresight structure */
Node::Node(double sigma_BN[3], constraintStruct constraints, scBoresightStruct boresights)
{
    // MRPswitch(sigma_BN, 1, this->sigma_BN);
	v3Copy(sigma_BN, this->sigma_BN);
	this->isBoundary = false;
	if (abs(v3Norm(this->sigma_BN) - 1) < 1e-5) {
		this->isBoundary = true;
	}
	this->heuristic = 0;
	this->priority = 0;
	this->neighborCount = 0;

	// check constraint compliance
	this->isFree = true;
	double BN[3][3];
	MRP2C(this->sigma_BN, BN);
	double boresight_B[3], boresight_N[3];
	int N;
	// keep-out
	if (constraints.keepOut) {
		N = boresights.keepOutBoresightCount;
		for (int n = 0; n < N; n++) {
			for (int j = 0; j < 3; j++) {
				boresight_B[j] = boresights.keepOutBoresight_B[n][j];
			}
			v3tMultM33(boresight_B, BN, boresight_N);
			if ( v3Dot(boresight_N, constraints.keepOutDir_N) >= cos(boresights.keepOutFov[n]) ) {
				this->isFree = false;
				return;
			}
		}
	}
	// keep-in
	if (constraints.keepIn) {
		N = boresights.keepInBoresightCount;
		bool isIn = false;
		for (int n = 0; n < N; n++) {
			for (int j = 0; j < 3; j++) {
				boresight_B[j] = boresights.keepInBoresight_B[n][j];
			}
			v3tMultM33(boresight_B, BN, boresight_N);
			if ( v3Dot(boresight_N, constraints.keepInDir_N) >= cos(boresights.keepInFov[n]) ) {
				isIn = true;
			}
		}
		if (!isIn) {
			this->isFree = false;
		}
	}

    return;
}

/*! Module Destructor.  */
Node::~Node()
{
    return;
}

/*! This method appends a pointer to neighboring node to the neighbors class variable */
void Node::appendNeighbor(Node *node)
{
	this->neighbors[this->neighborCount] = node;
	this->neighborCount += 1;
    return;
}


/*! This is the constructor for the NodeList class. */
NodeList::NodeList()
{
	this-> N = 0;
    return;
}

/*! Class Destructor. */
NodeList::~NodeList()
{
    return;
}

/*! This method appends a pointer to the node list. */
void NodeList::append(Node* node)
{
	this->list[this->N] = node;
	this->N += 1;
}

/*! This method removes the pointer at index M from the node list. */
void NodeList::pop(int M)
{
	for (int m = M; m < this->N-1; m++) {
		this->list[m] = this->list[m+1];
	}
	this->N -= 1;
}

/*! This method resets the list counter to 0, effectively clearing the list. */
void NodeList::clear()
{
	this->N = 0;
}

/*! This method swaps the two pointers at indices m and n. */
void NodeList::swap(int m, int n)
{
	Node *p1 = this->list[m];
	Node *p2 = this->list[n];
	this->list[m] = p2;
	this->list[n] = p1;
}

/*! This method sorts the nodes in the list according to their priority. */
void NodeList::sort()
{
	int M;
	double p;
	for (int n = 0; n < this->N; n++) {
		p = 1e5;
		M = this->N;
		for (int m = n; m < this->N; m++) {
			if (this->list[m]->priority < p) {
				p = this->list[m]->priority;
				M = m;
			}
		}
		swap(n, M);
	}
}

/*! This method returns true if a node is contained in the list, false otherwise. */
bool NodeList::contains(Node *node)
{
	bool flag = false;
	for (int n = 0; n < this->N; n++) {
		if (node == this->list[n]) {
			flag = true;
		}
	}
	return flag;
}


/*! This is the constructor for the module class.  It sets default variable
    values and initializes the various parts of the model */
ConstrainedAttitudeManeuver::ConstrainedAttitudeManeuver()
{
    return;
}

/*! This is the constructor for the module class. It sets default variable
    values and initializes the various parts of the model */
ConstrainedAttitudeManeuver::ConstrainedAttitudeManeuver(int N)
{
    this->N = N;
    this->scStateMsgBuffer        = this->scStateInMsg.zeroMsgPayload;
	this->keepOutCelBodyMsgBuffer = this->keepOutCelBodyInMsg.zeroMsgPayload;
	this->keepInCelBodyMsgBuffer  = this->keepInCelBodyInMsg.zeroMsgPayload;

    return;
}

/*! Module Destructor.  */
ConstrainedAttitudeManeuver::~ConstrainedAttitudeManeuver()
{
    return;
}

/*! Initialize C-wrapped output messages */
void ConstrainedAttitudeManeuver::SelfInit(){
    AttRefMsg_C_init(&this->attRefOutMsgC);
}

/*! Add keep-out body-frame direction  */
void ConstrainedAttitudeManeuver::appendKeepOutDirection(double direction[3], double Fov)
{
	int N = this->boresights.keepOutBoresightCount;
	v3Normalize(direction, this->boresights.keepOutBoresight_B[N]);
	this->boresights.keepOutFov[N] = Fov;
	this->boresights.keepOutBoresightCount += 1;
}

/*! Add keep-in body-frame direction  */
void ConstrainedAttitudeManeuver::appendKeepInDirection(double direction[3], double Fov)
{
	int N = this->boresights.keepInBoresightCount;
	v3Normalize(direction, this->boresights.keepInBoresight_B[N]);
	this->boresights.keepInFov[N] = Fov;
	this->boresights.keepInBoresightCount += 1;
}

/*! This method is used to reset the module. The input messages are read here,
the grid is generated and the graph search is performed. */
void ConstrainedAttitudeManeuver::Reset(uint64_t CurrentSimNanos)
{
	ReadInputs();

	Node startNode = Node(this->scStateMsgBuffer.sigma_BN, this->constraints, this->boresights);
	Node goalNode = Node(this->sigma_BN_goal, this->constraints, this->boresights);
	if (!startNode.isFree) {
		bskLogger.bskLog(BSK_WARNING, "ConstraintAttitudeManeuver: the initial attitude of the S/C is not constraint-compliant.");
	}
	if (!goalNode.isFree) {
		bskLogger.bskLog(BSK_WARNING, "ConstraintAttitudeManeuver: the target attitude of the S/C is not constraint-compliant.");
	}
	GenerateGrid(startNode, goalNode);
	if (this->costFcnType == 0) {
		AStar();
	}
	else if (this->costFcnType == 1) {
		effortBasedAStar();
	}
	else {
		bskLogger.bskLog(BSK_ERROR, "ConstraintAttitudeManeuver: costFcnType has not been specified.");
	}
	pathHandle();
	spline();
	this->pathCost = effortEvaluation();
    return;

}

/*! This method is the state update. It reads the information from the interpolated
trajectory and writes the output message.
 @return void
 @param CurrentSimNanos The current simulation time for system
 */
void ConstrainedAttitudeManeuver::UpdateState(uint64_t CurrentSimNanos)
{
	double t = CurrentSimNanos * 1e-9;
	double sigma_RN[3], sigmaDot_RN[3], sigmaDDot_RN[3], omega_RN_R[3], omegaDot_RN_R[3];
	this->Output.getData(t, sigma_RN, sigmaDot_RN, sigmaDDot_RN);

	dMRP2Omega(sigma_RN, sigmaDot_RN, omega_RN_R);
	ddMRP2dOmega(sigma_RN, sigmaDot_RN, sigmaDDot_RN, omegaDot_RN_R);
	
	// create the attitude output message buffer
	AttRefMsgPayload attMsgBuffer;
	// zero output message
    attMsgBuffer = this->attRefOutMsg.zeroMsgPayload;

	// compute direction cosine matrix [RN]
	double RN[3][3];
	MRP2C(sigma_RN, RN);

	v3Copy(sigma_RN, attMsgBuffer.sigma_RN);
	v3tMultM33(omega_RN_R, RN, attMsgBuffer.omega_RN_N);
	v3tMultM33(omegaDot_RN_R, RN, attMsgBuffer.domega_RN_N);

	// write output attitude reference message
    this->attRefOutMsg.write(&attMsgBuffer, this->moduleID, CurrentSimNanos);

    /* Write to the C-wrapped output messages */
    AttRefMsg_C_write(&attMsgBuffer, &this->attRefOutMsgC, this->moduleID, CurrentSimNanos);

	return;
}

/*! This method reads the input messages in from the system and sets the
 appropriate parameters
 @return void
 */
void ConstrainedAttitudeManeuver::ReadInputs()
{
	double relPosVector[3];

    //! - Read the input messages into the correct pointer
	if (this->scStateInMsg.isWritten()) {
		this->scStateMsgBuffer = this->scStateInMsg();
	}
	if (this->vehicleConfigInMsg.isWritten()) {
		this->vehicleConfigMsgBuffer = this->vehicleConfigInMsg();
	}
	if (this->keepOutCelBodyInMsg.isWritten()) {
		this->keepOutCelBodyMsgBuffer = this->keepOutCelBodyInMsg();

		//! - Compute the inertial direction of the object w.r.t. the S/C
		v3Subtract(this->keepOutCelBodyMsgBuffer.PositionVector, this->scStateMsgBuffer.r_BN_N, relPosVector);
	    v3Normalize(relPosVector, this->constraints.keepOutDir_N);
		this->constraints.keepOut = true;
	}
	if (this->keepInCelBodyInMsg.isWritten()) {
		this->keepInCelBodyMsgBuffer = this->keepInCelBodyInMsg();

		//! - Compute the inertial direction of the object w.r.t. the S/C
		v3Subtract(this->keepInCelBodyMsgBuffer.PositionVector, this->scStateMsgBuffer.r_BN_N, relPosVector);
	    v3Normalize(relPosVector, this->constraints.keepInDir_N);
		this->constraints.keepIn = true;
	}
	if (!this->keepOutCelBodyInMsg.isWritten() && !this->keepInCelBodyInMsg.isWritten()) {
		bskLogger.bskLog(BSK_WARNING, "ConstraintAttitudeManeuver: no celBodyMsgs are connected. There are no rotational constraints. \n");
	}
}

/*! This method generates the MRP grid and connects the free neighboring nodes
 @return void
 */
void ConstrainedAttitudeManeuver::GenerateGrid(Node startNode, Node goalNode)
{
	int N = this->N;
    double u[20];
	for (int n = 0; n < N; n++) {
		u[n] = n / ((double)N - 1);
	}

	// add internal nodes (|sigma_BN| < 1)
	int indices[3], mirrorIndices[8][3];
	double sigma_BN[3];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				if (pow(u[i],2) + pow(u[j],2) + pow(u[k],2) <= 1) {
					indices[0] = i; indices[1] = j; indices[2] = k;
					mirrorFunction(indices, mirrorIndices);
					for (int m = 0; m < 8; m++) {
						if (this->NodesMap[mirrorIndices[m][0]][mirrorIndices[m][1]].count(mirrorIndices[m][2]) == 0) {
							for (int p = 0; p < 3; p++) {
								if (indices[p] != 0) { sigma_BN[p] = mirrorIndices[m][p]/indices[p]*u[indices[p]]; } else { sigma_BN[p] = 0; }
							}
							this->NodesMap[mirrorIndices[m][0]][mirrorIndices[m][1]][mirrorIndices[m][2]] = Node(sigma_BN, this->constraints, this->boresights);
						}
					}
				}
			}
		}
	}
	// add missing boundary nodes (|sigma_BN| = 1)
	for (int i = 0; i < N-1; i++) {
		for (int j = 0; j < N-1; j++) {
			for (int k = 0; k < N-1; k++) {
				if (this->NodesMap[i][j].count(k) == 1) {
					if (this->NodesMap[i][j][k].isBoundary == false) {
						// along i direction
						if (this->NodesMap[i+1][j].count(k) == 0) {
							indices[0] = i+1; indices[1] = j; indices[2] = k;
							mirrorFunction(indices, mirrorIndices);
							for (int m = 0; m < 8; m++) {
								if (this->NodesMap[mirrorIndices[m][0]][mirrorIndices[m][1]].count(mirrorIndices[m][2]) == 0) {
									sigma_BN[0] = mirrorIndices[m][0]/indices[0] * pow(1-pow(u[j],2)-pow(u[k],2),0.5);
									if (indices[1] != 0) { sigma_BN[1] = mirrorIndices[m][1]/indices[1]*u[indices[1]]; } else { sigma_BN[1] = 0; }
									if (indices[2] != 0) { sigma_BN[2] = mirrorIndices[m][2]/indices[2]*u[indices[2]]; } else { sigma_BN[2] = 0; }
									this->NodesMap[mirrorIndices[m][0]][mirrorIndices[m][1]][mirrorIndices[m][2]] = Node(sigma_BN, this->constraints, this->boresights);
								}
							}
						}
						// along j direction
						if (this->NodesMap[i][j+1].count(k) == 0) {
							indices[0] = i; indices[1] = j+1; indices[2] = k;
							mirrorFunction(indices, mirrorIndices);
							for (int m = 0; m < 8; m++) {
								if (this->NodesMap[mirrorIndices[m][0]][mirrorIndices[m][1]].count(mirrorIndices[m][2]) == 0) {
									if (indices[0] != 0) { sigma_BN[0] = mirrorIndices[m][0]/indices[0]*u[indices[0]]; } else { sigma_BN[0] = 0; }
									sigma_BN[1] = mirrorIndices[m][1]/indices[1] * pow(1-pow(u[i],2)-pow(u[k],2),0.5);
									if (indices[2] != 0) { sigma_BN[2] = mirrorIndices[m][2]/indices[2]*u[indices[2]]; } else { sigma_BN[2] = 0; }
									this->NodesMap[mirrorIndices[m][0]][mirrorIndices[m][1]][mirrorIndices[m][2]] = Node(sigma_BN, this->constraints, this->boresights);
								}
							}
						}
						// along k direction
						if (this->NodesMap[i][j].count(k+1) == 0) {
							indices[0] = i; indices[1] = j; indices[2] = k+1;
							mirrorFunction(indices, mirrorIndices);
							for (int m = 0; m < 8; m++) {
								if (this->NodesMap[mirrorIndices[m][0]][mirrorIndices[m][1]].count(mirrorIndices[m][2]) == 0) {
									if (indices[0] != 0) { sigma_BN[0] = mirrorIndices[m][0]/indices[0]*u[indices[0]]; } else { sigma_BN[0] = 0; }
									if (indices[1] != 0) { sigma_BN[1] = mirrorIndices[m][1]/indices[1]*u[indices[1]]; } else { sigma_BN[1] = 0; }
									sigma_BN[2] = mirrorIndices[m][2]/indices[2] * pow(1-pow(u[i],2)-pow(u[j],2),0.5);
									this->NodesMap[mirrorIndices[m][0]][mirrorIndices[m][1]][mirrorIndices[m][2]] = Node(sigma_BN, this->constraints, this->boresights);
								}
							}
						}
					}
				}
			}
		}
	}
	// link nodes to adjacent neighbors
	int neighbors[26][3];
	for (std::map<int,std::map<int,std::map<int,Node>>>::iterator it1=this->NodesMap.begin(); it1!=this->NodesMap.end(); it1++) {
		for (std::map<int,std::map<int,Node>>::iterator it2=it1->second.begin(); it2!=it1->second.end(); it2++) {
			for (std::map<int,Node>::iterator it3=it2->second.begin(); it3!=it2->second.end(); it3++) {
				indices[0] = it1->first; indices[1] = it2->first; indices[2] = it3->first;
				neighboringNodes(indices, neighbors);
				for (int n = 0; n < 26; n++) {
					if (this->NodesMap[neighbors[n][0]][neighbors[n][1]].count(neighbors[n][2]) == 1) {
						if (this->NodesMap[indices[0]][indices[1]][indices[2]].isFree && this->NodesMap[neighbors[n][0]][neighbors[n][1]][neighbors[n][2]].isFree) {
							this->NodesMap[indices[0]][indices[1]][indices[2]].appendNeighbor(&this->NodesMap[neighbors[n][0]][neighbors[n][1]][neighbors[n][2]]);
						}
					}
				}
			}
		}
	}
	// link boundary nodes to neighbors of shadow set
	bool flag;
	for (std::map<int,std::map<int,std::map<int,Node>>>::iterator it1=this->NodesMap.begin(); it1!=this->NodesMap.end(); it1++) {
		for (std::map<int,std::map<int,Node>>::iterator it2=it1->second.begin(); it2!=it1->second.end(); it2++) {
			for (std::map<int,Node>::iterator it3=it2->second.begin(); it3!=it2->second.end(); it3++) {
				if (it3->second.isBoundary && it3->second.isFree) {
					int i, j, k;
					i = it1->first;  j = it2->first;  k = it3->first;
					for (int n = 0; n < this->NodesMap[-i][-j][-k].neighborCount; n++) {
						flag = true;
						for (int m = 0; m < this->NodesMap[i][j][k].neighborCount; m++) {
							if (this->NodesMap[-i][-j][-k].neighbors[n] == this->NodesMap[i][j][k].neighbors[m]) {
								flag = false;
							}
						}
						if (flag == true) {
							it3->second.appendNeighbor(this->NodesMap[-i][-j][-k].neighbors[n]);
						}
					}
				}
			}
		}
	}
    
	// add start and goal node to grid and connecting them to the neighboring nodes:
	double ds = 10;
	double dg = 10;
	double d1, d2;
	int keyS[3];
	int keyG[3];
	for (std::map<int,std::map<int,std::map<int,Node>>>::iterator it1=this->NodesMap.begin(); it1!=this->NodesMap.end(); it1++) {
		for (std::map<int,std::map<int,Node>>::iterator it2=it1->second.begin(); it2!=it1->second.end(); it2++) {
			for (std::map<int,Node>::iterator it3=it2->second.begin(); it3!=it2->second.end(); it3++) {
				if (it3->second.isFree) {
					d1 = distance(startNode, it3->second);
					if (abs(d1-ds) < 1e-6) {
						if (v3Norm(it3->second.sigma_BN) < v3Norm(NodesMap[keyS[0]][keyS[1]][keyS[2]].sigma_BN)) {
							ds = d1;
							keyS[0] = it1->first;   keyS[1] = it2->first;   keyS[2] = it3->first;
						}
					}
					else {
						if (d1 < ds) {
							ds = d1;
							keyS[0] = it1->first;   keyS[1] = it2->first;   keyS[2] = it3->first;
						}
					}
					d2 = distance(goalNode, it3->second);
					if (abs(d2-dg) < 1e-6) {
						if (v3Norm(it3->second.sigma_BN) < v3Norm(NodesMap[keyG[0]][keyG[1]][keyG[2]].sigma_BN)) {
							dg = d2;
							keyG[0] = it1->first;   keyG[1] = it2->first;   keyG[2] = it3->first;
						}
					}
					else {
						if (d2 < dg) {
							dg = d2;
							keyG[0] = it1->first;   keyG[1] = it2->first;   keyG[2] = it3->first;
						}
					}
				}
			}
		}
	}
	if (startNode.isBoundary) {
		for (int n = 0; n < 3; n++) {
			if (keyS[n]*startNode.sigma_BN[n] < 0) {keyS[n] = -keyS[n];}
		}
	}
		if (goalNode.isBoundary) {
		for (int n = 0; n < 3; n++) {
			if (keyG[n]*goalNode.sigma_BN[n] < 0) {keyG[n] = -keyG[n];}
		}
	}
	for (int n = 0; n < this->NodesMap[keyS[0]][keyS[1]][keyS[2]].neighborCount; n++) {
		startNode.appendNeighbor(this->NodesMap[keyS[0]][keyS[1]][keyS[2]].neighbors[n]);
	}
	for (int n = 0; n < this->NodesMap[keyG[0]][keyG[1]][keyG[2]].neighborCount; n++) {
		this->NodesMap[keyG[0]][keyG[1]][keyG[2]].neighbors[n]->appendNeighbor(&goalNode);
	}
	this->NodesMap[keyS[0]][keyS[1]][keyS[2]] = startNode;
	this->NodesMap[keyG[0]][keyG[1]][keyG[2]] = goalNode;
	this->keyS[0] = keyS[0]; this->keyS[1] = keyS[1]; this->keyS[2] = keyS[2];
	this->keyG[0] = keyG[0]; this->keyG[1] = keyG[1]; this->keyG[2] = keyG[2];
	
}

/*! This method is used inside A* to track the path from goal to start, order it from start to goal and store in class variable path
 @return void
 */
void ConstrainedAttitudeManeuver::backtrack(Node *p)
{
	if (p == &NodesMap[this->keyS[0]][this->keyS[1]][this->keyS[2]]) {
		this->path.append(p);
		return;
	}
	else {
		backtrack(p->backPointer);
		this->path.append(p);
		return;
	}
}

/*! This method applies standard distance-based A* to find a valid path
 @return void
 */
void ConstrainedAttitudeManeuver::AStar()
{
	for (std::map<int,std::map<int,std::map<int,Node>>>::iterator it1=NodesMap.begin(); it1!=NodesMap.end(); it1++) {
		for (std::map<int,std::map<int,Node>>::iterator it2=it1->second.begin(); it2!=it1->second.end(); it2++) {
			for (std::map<int,Node>::iterator it3=it2->second.begin(); it3!=it2->second.end(); it3++) {
				it3->second.heuristic = distance(it3->second, NodesMap[this->keyG[0]][this->keyG[1]][this->keyG[2]]);
			}
		}
	}
	int Nmax = 4*this->N*this->N;
	int n = 0;
	double p;
	Node *key;
    NodeList O, C;
	O.append(&NodesMap[this->keyS[0]][this->keyS[1]][this->keyS[2]]);

	while (O.list[0] != &NodesMap[this->keyG[0]][this->keyG[1]][this->keyG[2]] && n < Nmax) {
		n += 1;
		C.append(O.list[0]);

		for (int k = 0; k < O.list[0]->neighborCount; k++) {
			key = O.list[0]->neighbors[k];

			if (C.contains(key) == false) {
				p = key->heuristic + distance(*O.list[0], *key) + O.list[0]->priority - O.list[0]->heuristic;
				if (O.contains(key)) {
					if (p < key->priority) {
						key->priority = p;
						key->backPointer = O.list[0];
					}
				}
				else {
					key->priority = p;
					key->backPointer = O.list[0];
					O.append(key);
				}
			}
		}
		
		O.pop(0);
		O.sort();
	}

	backtrack(O.list[0]);
    
	// Uncomment to print path node coordinates
	/*
	for (int n = 0; n < this->path.N; n++) {
		std::cout << this->path.list[n]->sigma_BN[0] << " " << this->path.list[n]->sigma_BN[1] << " " << this->path.list[n]->sigma_BN[2] << "\n";
	}*/
}

/*! This method applies the effort-based A* to find a valid path
 @return void
 */
void ConstrainedAttitudeManeuver::effortBasedAStar()
{
	int Nmax = 100;
	int n = 0;
	double p;
	Node *key;
    NodeList O, C;
	O.append(&NodesMap[this->keyS[0]][this->keyS[1]][this->keyS[2]]);

	while (O.list[0] != &NodesMap[this->keyG[0]][this->keyG[1]][this->keyG[2]] && n < Nmax) {
		n += 1;
		// std::cout << "N = " << n << "\n"; // uncomment to show the number of nodes explored
		C.append(O.list[0]);

		for (int k = 0; k < O.list[0]->neighborCount; k++) {

			key = O.list[0]->neighbors[k];

			if (C.contains(key) == false) {
				backtrack(O.list[0]);
				this->path.append(key);
				if (key != &NodesMap[this->keyG[0]][this->keyG[1]][this->keyG[2]]) {
					this->path.append(&NodesMap[this->keyG[0]][this->keyG[1]][this->keyG[2]]);
				}
				pathHandle();
				spline();
				p = effortEvaluation();
				this->path.clear();
				if (O.contains(key)) {
					if (p < key->priority) {
						key->priority = p;
						key->backPointer = O.list[0];
					}
				}
				else {
					key->priority = p;
					key->backPointer = O.list[0];
					O.append(key);
				}
			}
		}
		
		O.pop(0);
		O.sort();
	}

	backtrack(O.list[0]);
    
	// Uncomment to print path node coordinates
	/*
	std::cout << "Waypoints: \n";
	for (int n = 0; n < this->path.N; n++) {
		std::cout << this->path.list[n]->sigma_BN[0] << " " << this->path.list[n]->sigma_BN[1] << " " << this->path.list[n]->sigma_BN[2] << "\n";
	} */
}

/*! This method takes a path of waypoints and returns an Input structure suitable for BSpline interpolation/approximation
 @return void
 */
void ConstrainedAttitudeManeuver::pathHandle()
{
	double S = 0;
	Eigen::VectorXd T(this->path.N);
	T[0] = 0;
	for (int n = 0; n < this->path.N-1; n++) {
		T[n+1] = T[n] + distance(*path.list[n], *path.list[n+1]);
		S += T[n+1] - T[n];
	}
	Eigen::VectorXd X1(this->path.N);
	Eigen::VectorXd X2(this->path.N);
	Eigen::VectorXd X3(this->path.N);

	bool shadowSet = false;
	double sigma[3], delSigma[3];

	for (int n = 0; n < this->path.N-1; n++) {
		if (!shadowSet) {
			v3Copy(path.list[n]->sigma_BN, sigma);
		}
		else {
			if (path.list[n] == &NodesMap[0][0][0]) {
				for (int m = n-1; m > -1; m--) {
					sigma[0] = X1[m]; sigma[1] = X2[m]; sigma[2] = X3[m];
					X1[m] = -X1[m] / v3Dot(sigma, sigma);
					X2[m] = -X2[m] / v3Dot(sigma, sigma);
					X3[m] = -X3[m] / v3Dot(sigma, sigma);
				}
				shadowSet = !shadowSet;
			}
			MRPshadow(path.list[n]->sigma_BN, sigma);
		}
		v3Subtract(path.list[n]->sigma_BN, path.list[n+1]->sigma_BN, delSigma);
		if (v3Norm(delSigma) > 1) {
			shadowSet = !shadowSet;
		}
		X1[n] = sigma[0]; X2[n] = sigma[1]; X3[n] = sigma[2];
	}
	if (!shadowSet) {
		v3Copy(path.list[path.N-1]->sigma_BN, sigma);
	}
	else {
		MRPshadow(path.list[path.N-1]->sigma_BN, sigma);
	}
	X1[path.N-1] = sigma[0]; X2[path.N-1] = sigma[1]; X3[path.N-1] = sigma[2];

	this->Input = InputDataSet(X1, X2, X3);
	this->Input.setT(T * 4 * S / (T[this->path.N-1] * this->avgOmega));
}

/*! This method performs the BSpline interpolation/approximation and outputs an Output structure
 @return void
 */
void ConstrainedAttitudeManeuver::spline()
{
	double sigmaDot_start[3], sigmaDot_goal[3];
	dMRP(this->scStateMsgBuffer.sigma_BN, this->scStateMsgBuffer.omega_BN_B, sigmaDot_start);
	dMRP(this->sigma_BN_goal, this->omega_BN_B_goal, sigmaDot_goal);
	Eigen::Vector3d sDot_s, sDot_g;
	for (int i = 0; i < 3; i++) {
		sDot_s[i] = sigmaDot_start[i];
		sDot_g[i] = sigmaDot_goal[i];
	}
	
	this->Input.setXDot_0(sDot_s);
	this->Input.setXDot_N(sDot_g);
	if (this->BSplineType == 0) {
		interpolate(this->Input, 100, 4, &this->Output);
	}
	else if (this->BSplineType == 1) {
		approximate(this->Input, 100, (int) this->Input.X1.size(), 4, &this->Output);  // review
	}
	else {
		bskLogger.bskLog(BSK_ERROR, "ConstraintAttitudeManeuver: BSplineType has not been specified.");
	}
}

/*! This method computes the torque vector required at time step with index n
 @return void
 */
void ConstrainedAttitudeManeuver::computeTorque(int n, double I[9], double L[3])
{
	double sigma[3], sigmaDot[3], sigmaDDot[3], omega[3], omegaDot[3], L1[3], L2[3], H[3];
	
	sigma[0]     = this->Output.X1[n];    sigma[1]     = this->Output.X2[n];    sigma[2]     = this->Output.X3[n];
	sigmaDot[0]  = this->Output.XD1[n];   sigmaDot[1]  = this->Output.XD2[n];   sigmaDot[2]  = this->Output.XD3[n];
	sigmaDDot[0] = this->Output.XDD1[n];  sigmaDDot[1] = this->Output.XDD2[n];  sigmaDDot[2] = this->Output.XDD3[n];
	dMRP2Omega(sigma, sigmaDot, omega);
	ddMRP2dOmega(sigma, sigmaDot, sigmaDDot, omegaDot);
    m33MultV3(RECAST3X3 I, omegaDot, L1);
	m33MultV3(RECAST3X3 I, omega, H);
	v3Cross(omega, H, L2);
	v3Add(L1, L2, L);
}

/*! This method computes the torque vector required at time step with index n
 @return void
 */
double ConstrainedAttitudeManeuver::computeTorqueNorm(int n, double I[9])
{
	double L[3];
	computeTorque(n, this->vehicleConfigMsgBuffer.ISCPntB_B, L);
	
	return v3Norm(L);
}

/*! This method integrates the control torque norm over maneuver time and returns the cost for the effort-based A*.
 @return void
 */
double ConstrainedAttitudeManeuver::effortEvaluation()
{
    double effort = 0;
	double l_a, l_b;
	double L_a[3], L_b[3];
	computeTorque(0, this->vehicleConfigMsgBuffer.ISCPntB_B, L_a);

	N = (int) this->Output.T.size();
	for (int n = 0; n < N-1; n++) {
		computeTorque(n+1, this->vehicleConfigMsgBuffer.ISCPntB_B, L_b);
		l_a = v3Norm(L_a);
		l_b = v3Norm(L_b);
		effort += (l_a + l_b) * (Output.T[n+1] - Output.T[n]) / 2;

		v3Copy(L_b, L_a);
	}

	return effort;
}

/*! This method  allows to access the coordinates of a Node in NodesMap without swigging the C++ Map.
    It is designed to be used in the UnitTest primarily.
 @return void
 */
double ConstrainedAttitudeManeuver::returnNodeCoord(int key[3], int nodeCoord)
{
	if (nodeCoord < 0 || nodeCoord > 2 || NodesMap[key[0]][key[1]].count(key[2]) == 0) {
		return 1000; // random large number that will cause the UnitTest comparison to fail
	}
	else {
		return this->NodesMap[key[0]][key[1]][key[2]].sigma_BN[nodeCoord];
	}
}

/*! This method  allows to access the state of a Node (free or not free) in NodesMap without swigging the C++ Map.
    It is designed to be used in the UnitTest primarily.
 @return void
 */
bool ConstrainedAttitudeManeuver::returnNodeState(int key[3])
{
	if (NodesMap[key[0]][key[1]].count(key[2]) != 0) {
		return this->NodesMap[key[0]][key[1]][key[2]].isFree;
	}
	else {
		return false;
	}
}

/*! This method allows to access the coordinates of path Nodes in without swigging NodesList C++.
    It is designed to be used in the UnitTest primarily.
 @return void
 */
double ConstrainedAttitudeManeuver::returnPathCoord(int index, int nodeCoord)
{
	if (index < 0 || index > this->path.N-1) {
		return 1000; // random large number that will cause the UnitTest comparison to fail
	}
	else {
		return this->path.list[index]->sigma_BN[nodeCoord];
	}
}

/*! This helper function returns the coordinates of the 8 symmetrical points to a point in 3D cartesian space. */
void mirrorFunction(int indices[3], int mirrorIndices[8][3]) 
{
	mirrorIndices[0][0] =  indices[0];   mirrorIndices[0][1] =  indices[1];   mirrorIndices[0][2] =  indices[2];
	mirrorIndices[1][0] = -indices[0];   mirrorIndices[1][1] =  indices[1];   mirrorIndices[1][2] =  indices[2];
	mirrorIndices[2][0] =  indices[0];   mirrorIndices[2][1] = -indices[1];   mirrorIndices[2][2] =  indices[2];
	mirrorIndices[3][0] =  indices[0];   mirrorIndices[3][1] =  indices[1];   mirrorIndices[3][2] = -indices[2];
	mirrorIndices[4][0] = -indices[0];   mirrorIndices[4][1] = -indices[1];   mirrorIndices[4][2] =  indices[2];
	mirrorIndices[5][0] = -indices[0];   mirrorIndices[5][1] =  indices[1];   mirrorIndices[5][2] = -indices[2];
	mirrorIndices[6][0] =  indices[0];   mirrorIndices[6][1] = -indices[1];   mirrorIndices[6][2] = -indices[2];
	mirrorIndices[7][0] = -indices[0];   mirrorIndices[7][1] = -indices[1];   mirrorIndices[7][2] = -indices[2];
}

/*! For a set of indices, this helper function returns the indices of the 26 immediate neighbors. */
void neighboringNodes(int indices[3], int neighbors[26][3])
{
	neighbors[0][0]  = indices[0]-1;     neighbors[0][1]  = indices[1];       neighbors[0][2]  = indices[2];
	neighbors[1][0]  = indices[0]+1;     neighbors[1][1]  = indices[1];       neighbors[1][2]  = indices[2];
	neighbors[2][0]  = indices[0];       neighbors[2][1]  = indices[1]-1;     neighbors[2][2]  = indices[2];
	neighbors[3][0]  = indices[0];       neighbors[3][1]  = indices[1]+1;     neighbors[3][2]  = indices[2];
	neighbors[4][0]  = indices[0];       neighbors[4][1]  = indices[1];       neighbors[4][2]  = indices[2]-1;
	neighbors[5][0]  = indices[0];       neighbors[5][1]  = indices[1];       neighbors[5][2]  = indices[2]+1;
	neighbors[6][0]  = indices[0]-1;     neighbors[6][1]  = indices[1]-1;     neighbors[6][2]  = indices[2];
	neighbors[7][0]  = indices[0]+1;     neighbors[7][1]  = indices[1]-1;     neighbors[7][2]  = indices[2];
	neighbors[8][0]  = indices[0]-1;     neighbors[8][1]  = indices[1]+1;     neighbors[8][2]  = indices[2];
	neighbors[9][0]  = indices[0]+1;     neighbors[9][1]  = indices[1]+1;     neighbors[9][2]  = indices[2];
	neighbors[10][0] = indices[0]-1;     neighbors[10][1] = indices[1];       neighbors[10][2] = indices[2]-1;
	neighbors[11][0] = indices[0]+1;     neighbors[11][1] = indices[1];       neighbors[11][2] = indices[2]-1;
    neighbors[12][0] = indices[0]-1;     neighbors[12][1] = indices[1];       neighbors[12][2] = indices[2]+1;
	neighbors[13][0] = indices[0]+1;     neighbors[13][1] = indices[1];       neighbors[13][2] = indices[2]+1;
	neighbors[14][0] = indices[0];       neighbors[14][1] = indices[1]-1;     neighbors[14][2] = indices[2]-1;
	neighbors[15][0] = indices[0];       neighbors[15][1] = indices[1]+1;     neighbors[15][2] = indices[2]-1;
	neighbors[16][0] = indices[0];       neighbors[16][1] = indices[1]-1;     neighbors[16][2] = indices[2]+1;
	neighbors[17][0] = indices[0];       neighbors[17][1] = indices[1]+1;     neighbors[17][2] = indices[2]+1;
	neighbors[18][0] = indices[0]-1;     neighbors[18][1] = indices[1]-1;     neighbors[18][2] = indices[2]-1;
	neighbors[19][0] = indices[0]+1;     neighbors[19][1] = indices[1]-1;     neighbors[19][2] = indices[2]-1;
    neighbors[20][0] = indices[0]-1;     neighbors[20][1] = indices[1]+1;     neighbors[20][2] = indices[2]-1;
	neighbors[21][0] = indices[0]-1;     neighbors[21][1] = indices[1]-1;     neighbors[21][2] = indices[2]+1;
	neighbors[22][0] = indices[0]+1;     neighbors[22][1] = indices[1]+1;     neighbors[22][2] = indices[2]-1;
	neighbors[23][0] = indices[0]+1;     neighbors[23][1] = indices[1]-1;     neighbors[23][2] = indices[2]+1;
	neighbors[24][0] = indices[0]-1;     neighbors[24][1] = indices[1]+1;     neighbors[24][2] = indices[2]+1;
	neighbors[25][0] = indices[0]+1;     neighbors[25][1] = indices[1]+1;     neighbors[25][2] = indices[2]+1;
}

/*! This function implements the MRP cartesian distance between 2 nodes.  */
double distance(Node n1, Node n2)
{
	double n1n, n2n;
	double D, d[4];
	double dn[3], n1s[3], n2s[3];
	n1n = v3Norm(n1.sigma_BN);
	n2n = v3Norm(n2.sigma_BN);
	v3Subtract(n1.sigma_BN, n2.sigma_BN, dn);
	d[0] = v3Norm(dn);
	if (n2n > 1e-8) {
		MRPshadow(n2.sigma_BN, n2s);
		v3Subtract(n1.sigma_BN, n2s, dn);
		d[1] = v3Norm(dn);
	}
	else {
		d[1] = d[0];
	}
	if (n1n > 1e-8) {
		MRPshadow(n1.sigma_BN, n1s);
		v3Subtract(n2.sigma_BN, n1s, dn);
		d[2] = v3Norm(dn);
	}
	else {
		d[2] = d[0];
	}
	if (n1n > 1e-8 && n2n > 1e-8) {
		v3Subtract(n1s, n2s, dn);
		d[3] = v3Norm(dn);
	}
	else {
		d[3] = d[0];
	}
	D = d[0];
    for (int i = 1; i < 4; i++) {
        if (d[i] < D) { D = d[i]; }
    }
	return D;
}
