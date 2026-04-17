/*
 * Soufiane El mouahid
 * 
 * Martin Hart
*/

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <cstring>
#include <sys/mman.h>    
#include <unistd.h>      
#include <sys/wait.h>    
#include <cstdlib>       

#include "SimulationNBodyPosix.hpp"

SimulationNBodyPosix::SimulationNBodyPosix(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 18.f * (float)this->getBodies().getN() * (float)this->getBodies().getN() + (float)this->getBodies().getN() + 1.0;
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyPosix::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyPosix::computeBodiesAcceleration()
{
    const unsigned long Nbodies = this->getBodies().getN();

    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    
    // compute G.mn in advance
    std::vector<float> g(Nbodies); 
    // flops = n
    for (unsigned long iBody = 0; iBody < Nbodies; iBody++) {
        g[iBody] = this->G * d[iBody].m; // 1 flop
    }

    // compute e²
    const float softSquared = this->soft * this->soft; // 1 flop

    // number on processes based on Nvidia Jetson TX2 number of cpu cores
    unsigned int numProcesses = 6;
    if (Nbodies < numProcesses) {
        numProcesses = Nbodies;
    }
    
    // shared memory for child processes to accumulate accelerations and for parent process to collect them
    auto* sharedAccelerations = (accAoS_t<float>*)mmap(nullptr, Nbodies * sizeof(accAoS_t<float>),
                                            PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    
    std::vector<pid_t> pids(numProcesses);

    // launch computation on child processus
    for (unsigned int pid = 0; pid < numProcesses; pid++) {
        if ((pids[pid] = fork()) == 0) {
            
            // last child process deals with the remaining bodies
            const unsigned long end = (pid == numProcesses - 1) ? Nbodies : (pid + 1) * (Nbodies / numProcesses);

            // flops = n² * 18 / numProcesses environ
            for (unsigned long iBody = pid * (Nbodies / numProcesses); iBody < end; iBody++) {
                // flops = n * 18
                for (unsigned long jBody = 0; jBody < Nbodies; jBody++) {
                    const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
                    const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
                    const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

                    // compute the || rij ||² distance between body i and body j
                    const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
                    
                    // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
                    const float dist = std::pow(rijSquared + softSquared, 3.f / 2.f); // 3 flops
                    const float ai = g[jBody] / dist; // 1 flop

                    // add the acceleration value into the acceleration vector: ai += || ai ||.rij
                    sharedAccelerations[iBody].ax += ai * rijx; // 2 flops
                    sharedAccelerations[iBody].ay += ai * rijy; // 2 flops
                    sharedAccelerations[iBody].az += ai * rijz; // 2 flops
                }
            }
            exit(0);
        }
    }

    // wait for processes to end
    for (unsigned int pid = 0; pid < numProcesses; pid++) {
        waitpid(pids[pid], nullptr, 0);
    }

    // collect results
    std::memcpy(&this->accelerations[0], sharedAccelerations, Nbodies * sizeof(accAoS_t<float>));

    // free shared memory
    munmap(sharedAccelerations, Nbodies * sizeof(accAoS_t<float>));
}

void SimulationNBodyPosix::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
