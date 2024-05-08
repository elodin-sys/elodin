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

#ifndef WAYPOINTREFERENCE_H
#define WAYPOINTREFERENCE_H

#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/utilities/bskLogging.h"
#include "architecture/utilities/BSpline.h"
#include "architecture/messaging/messaging.h"
#include <map>
#include <iostream>
#include <fstream>
#include "architecture/msgPayloadDefC/SCStatesMsgPayload.h"
#include "architecture/msgPayloadDefC/VehicleConfigMsgPayload.h"
#include "architecture/msgPayloadDefC/SpicePlanetStateMsgPayload.h"
#include "cMsgCInterface/AttRefMsg_C.h"

//! @brief The constraintStruc structure is used to store the inertial direction of the keep-in and keep-out zones
struct constraintStruct {

    double keepOutDir_N[3];                                         //!< Inertial direction of keepOut celestial bodies
    double keepInDir_N[3];                                          //!< Inertial direction of keepIn celestial bodies
    bool keepOut;                                                   //!< Flag to assess whether keepOut constraints are being considered
    bool keepIn;                                                    //!< Flag to assess whether keepIn constraints are being considered
};

//! @brief The scBoresightStruc structure is used to store the body frame directions and fields of view of the instruments
struct scBoresightStruct {

    double keepOutBoresight_B[10][3];                               //!< Unit vectors containing body frame directions of keepOut instruments
    double keepInBoresight_B[10][3];                                //!< Unit vectors containing body frame directions of keepIn instruments
    double keepOutFov[10];                                          //!< Fields of view of the keepOut instruments
    double keepInFov[10];                                           //!< Fields of view of the keepIn instruments
    int keepOutBoresightCount = 0;                                  //!< Number of keepOut sensitive instruments
    int keepInBoresightCount = 0;                                   //!< Number of keepIn instruments
};

//! @brief The Node class is used to create nodes in the 3D MRP graph
class Node {
public:
    Node();
    Node(double sigma_BN[3], constraintStruct constraints, scBoresightStruct boresights);
    ~Node();

    double sigma_BN[3];                                             //!< MRP set corresponding to the node
    bool isBoundary;                                                //!< If true, the node lies on the |sigma| = 1 boundary surface
    bool isFree;                                                    //!< If true, the node is constraint-compliant
    double heuristic;                                               //!< Heuristic value used by cartesian distance A*
    double priority;                                                //!< Priority of the node in A*
    Node *neighbors[52];                                            //!< Container of pointers to neighboring nodes
    int neighborCount;                                              //!< Number of neighboring nodes
    Node *backPointer;                                              //!< Pointer to the previous node in the path computer by A*
    void appendNeighbor(Node *node);
};

//! @brief The NodeList class is used in the A* algorithm to handle Open and Closed lists O and C
class NodeList {
public:
    NodeList();
    ~NodeList();

    Node* list[10000];                                              //!< Container of pointers to the nodes in the list
    int N;                                                          //!< Number of nodes in the list
    void append(Node* node);
    void pop(int M);
    void clear();
    void swap(int m, int n);
    void sort();
    bool contains(Node *node);
};

/*! @brief waypoint reference module class */
class ConstrainedAttitudeManeuver: public SysModel {
public:
    ConstrainedAttitudeManeuver();
    ConstrainedAttitudeManeuver(int N);
    ~ConstrainedAttitudeManeuver(); 
    void SelfInit();  //!< Self initialization for C-wrapped messages
    void Reset(uint64_t CurrentSimNanos);
    void UpdateState(uint64_t CurrentSimNanos);
    void ReadInputs();
    void GenerateGrid(Node startNode, Node goalNode);
    void appendKeepOutDirection(double direction[3], double Fov);
    void appendKeepInDirection(double direction[3], double Fov);
    void AStar();
    void effortBasedAStar();
    void backtrack(Node *p);
    void pathHandle();
    void spline();
    void computeTorque(int n, double I[9], double L[3]);
    double computeTorqueNorm(int n, double I[9]);
    double effortEvaluation();
    double returnNodeCoord(int key[3], int nodeCoord);
    bool returnNodeState(int key[3]);
    double returnPathCoord(int index, int nodeCoord);

public:
    int N;                                                                          //!< Fineness level of discretization
    int BSplineType;                                                                //!< 0 for interpolation; 1 for LS approximation
    int costFcnType;                                                                //!< 0 for minimum distance path; 1 for minimum control effort path
    double sigma_BN_goal[3];                                                        //!< Initial S/C attitude
    double omega_BN_B_goal[3];                                                      //!< Initial S/C angular rate
    double avgOmega;                                                                //!< Average angular rate norm during the maneuver
    double keepOutFov;                                                              //!< Field of view of the sensitive instrument
    double keepOutBore_B[3];                                                        //!< Body-frame direction of the boresight of the sensitive instrument
    constraintStruct constraints;                                                   //!< Structure containing the constraint directions in inertial coordinates
    scBoresightStruct boresights;                                                   //!< Structure containing the instrument boresight directions in body frame coordinates
    std::map<int,std::map<int,std::map<int,Node>>> NodesMap;                        //!< C++ map from node indices to Node class
    int keyS[3];                                                                    //!< Key to Start node in NodesMap
    int keyG[3];                                                                    //!< Key to Goal node in NodesMap
    NodeList path;                                                                  //!< Path of nodes from start to goal
    double pathCost;                                                                //!< Cost of the path above, according to the cost function used
    InputDataSet Input;                                                             //!< Input structure for the BSpline interpolation/approximation
    OutputDataSet Output;                                                           //!< Output structure of the BSpline interpolation/approximation

    ReadFunctor<SCStatesMsgPayload> scStateInMsg;                                   //!< Spacecraft state input message
    ReadFunctor<VehicleConfigMsgPayload> vehicleConfigInMsg;                        //!< FSW vehicle configuration input message
    ReadFunctor<SpicePlanetStateMsgPayload> keepOutCelBodyInMsg;                    //!< Celestial body state msg - keep out direction
    ReadFunctor<SpicePlanetStateMsgPayload> keepInCelBodyInMsg;                     //!< Celestial body state msg - keep in direction
    Message<AttRefMsgPayload> attRefOutMsg;                                         //!< Attitude reference output message
    AttRefMsg_C attRefOutMsgC = {};                                                 //!< C-wrapped attitude reference output message

    BSKLogger bskLogger;                                                            //!< BSK Logging

private:
    SCStatesMsgPayload scStateMsgBuffer;
    VehicleConfigMsgPayload vehicleConfigMsgBuffer;
    SpicePlanetStateMsgPayload keepOutCelBodyMsgBuffer;
    SpicePlanetStateMsgPayload keepInCelBodyMsgBuffer;
};

void mirrorFunction(int indices[3], int mirrorIndices[8][3]);

void neighboringNodes(int indices[3], int neighbors[26][3]);

double distance(Node n1, Node n2);

#endif
