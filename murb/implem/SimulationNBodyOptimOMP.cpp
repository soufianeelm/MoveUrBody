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

#include "SimulationNBodyOptimOMP.hpp"

SimulationNBodyOptimOMP::SimulationNBodyOptimOMP(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = ( 25.f * (float)this->getBodies().getN()-1 * (float)this->getBodies().getN() * 1 / 2 ) + (float)this->getBodies().getN() + 1.0;
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyOptimOMP::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyOptimOMP::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    // compute G.mn in advance
    std::vector<float> g(this->getBodies().getN()); 
    // flops = n
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        g[iBody] = this->G * d[iBody].m; // 1 flop
    }

    // compute e²
    const float softSquared = this->soft * this->soft; // 1 flop

    // flops = n(n-1)/2 * 25
    // OMP_NUM_THREADS=6 OMP_SCHEDULE="dynamic"
    #pragma omp parallel 
    {
        std::vector<accAoS_t<float>> local_acc(this->getBodies().getN(), {0.0f, 0.0f, 0.0f});
        
        #pragma omp for schedule(runtime)
        for (unsigned long iBody = 0; iBody < this->getBodies().getN() - 1; iBody++) {
            // flops = (n - iBody + 1) * 25
            for (unsigned long jBody = iBody + 1; jBody < this->getBodies().getN(); jBody++) {
                const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
                const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
                const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

                // compute the || rij ||² distance between body i and body j
                const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
                
                // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
                const float dist = std::pow(rijSquared + softSquared, 3.f / 2.f); // 3 flops
                const float ai_0 = g[jBody] / dist; // 1 flop

                // add the acceleration value into the acceleration vector: ai += || ai ||.rij
                local_acc[iBody].ax += ai_0 * rijx; // 2 flops
                local_acc[iBody].ay += ai_0 * rijy; // 2 flops
                local_acc[iBody].az += ai_0 * rijz; // 2 flops

                // reflexive computation
                
                // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
                const float ai_1 = g[iBody] / dist; // 1 flop

                // add the acceleration value into the acceleration vector: ai += || ai ||.rij
                local_acc[jBody].ax -= ai_1 * rijx; // 2 flops
                local_acc[jBody].ay -= ai_1 * rijy; // 2 flops
                local_acc[jBody].az -= ai_1 * rijz; // 2 flops
            }
        }

        // combine threads results
        #pragma omp critical
        for (unsigned long Body = 0; Body < this->getBodies().getN(); Body++) {
            this->accelerations[Body].ax += local_acc[Body].ax;
            this->accelerations[Body].ay += local_acc[Body].ay;
            this->accelerations[Body].az += local_acc[Body].az;
        }
    }
}

void SimulationNBodyOptimOMP::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
