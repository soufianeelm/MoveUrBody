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

#include "SimulationNBodyOptimSIMDOMP.hpp"

SimulationNBodyOptimSIMDOMP::SimulationNBodyOptimSIMDOMP(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN(); // to do
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());
}

void SimulationNBodyOptimSIMDOMP::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations.ax[iBody] = 0.f;
        this->accelerations.ay[iBody] = 0.f;
        this->accelerations.az[iBody] = 0.f;
    }
}

void SimulationNBodyOptimSIMDOMP::computeBodiesAcceleration()
{
    unsigned long Nbodies = this->getBodies().getN(), index;
    
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();

    // compute G.mn in advance
    std::vector<float> g(Nbodies); 
    // flops = n
    for (unsigned long iBody = 0; iBody < Nbodies; iBody++) {
        g[iBody] = this->G * d.m[iBody]; // 1 flop
    }

    // compute e²
    const float softSquared = this->soft * this->soft; // 1 flop

    // compute e² register
    mipp::Reg<float> softSquared_r = softSquared;

    // flops = to do
    // OMP_NUM_THREADS=5 OMP_SCHEDULE="dynamic"
    #pragma omp parallel private(index)
    {   
        accSoA_t<float> local_acc;
        local_acc.ax = std::vector<float>(Nbodies, 0.0f);
        local_acc.ay = std::vector<float>(Nbodies, 0.0f);
        local_acc.az = std::vector<float>(Nbodies, 0.0f);

        #pragma omp for schedule(runtime)
        for (unsigned long iBody = 0; iBody < Nbodies - 1; iBody++) {
            // set 'index' to start of bodies that don't fill a full SIMD register; if not, use scalar loop only
            index = (iBody + 1) + ((Nbodies - (iBody + 1)) / mipp::N<float>()) * mipp::N<float>();
            
            mipp::Reg<float> ax = 0.0f, ay = 0.0f, az = 0.0f;

            // flops = to do
            for (unsigned long jBody = iBody + 1; jBody < index; jBody += mipp::N<float>()) {
                mipp::Reg<float> rijx_r = mipp::load(&d.qx[jBody]) - d.qx[iBody]; // 4 flops
                mipp::Reg<float> rijy_r = mipp::load(&d.qy[jBody]) - d.qy[iBody]; // 4 flops
                mipp::Reg<float> rijz_r = mipp::load(&d.qz[jBody]) - d.qz[iBody]; // 4 flops

                // compute the || rij ||² distance between body i and body j
                mipp::Reg<float> rijSquared_r = mipp::fmadd(rijx_r, rijx_r, mipp::fmadd(rijy_r, rijy_r, rijz_r * rijz_r)); // 20 flops
                // compute sum
                mipp::Reg<float> sum_r = rijSquared_r + softSquared_r; // 4 flops
                // compute dist
                mipp::Reg<float> dist_r = (sum_r) * mipp::sqrt(sum_r); // 8 flops
                // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
                mipp::Reg<float> mass_0_r = &g[jBody];
                mipp::Reg<float> ai_r = mass_0_r / dist_r; // 4 flops

                // add the acceleration value into the acceleration coordinates: ai += || ai ||.rij
                ax += ai_r * rijx_r; // 8 flops
                ay += ai_r * rijy_r; // 8 flops
                az += ai_r * rijz_r; // 8 flops

                // reflexive computation

                // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
                mipp::Reg<float> mass_1_r = g[iBody];
                mipp::Reg<float> aj_r = mass_1_r / dist_r; // 4 flops

                // add the acceleration value into the acceleration vector: ai += || ai ||.rij
                mipp::Reg<float> aj_x_r = &local_acc.ax[jBody];
                mipp::Reg<float> aj_y_r = &local_acc.ay[jBody];
                mipp::Reg<float> aj_z_r = &local_acc.az[jBody];

                mipp::store(&local_acc.ax[jBody], aj_x_r - (aj_r * rijx_r)); // 8 flops
                mipp::store(&local_acc.ay[jBody], aj_y_r - (aj_r * rijy_r)); // 8 flops
                mipp::store(&local_acc.az[jBody], aj_z_r - (aj_r * rijz_r)); // 8 flops
            }

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            local_acc.ax[iBody] += ax.sum();
            local_acc.ay[iBody] += ay.sum();
            local_acc.az[iBody] += az.sum();

            for (unsigned long jBody = index; jBody < Nbodies; jBody++) {
                const float rijx = d.qx[jBody] - d.qx[iBody]; // 1 flop
                const float rijy = d.qy[jBody] - d.qy[iBody]; // 1 flop
                const float rijz = d.qz[jBody] - d.qz[iBody]; // 1 flop

                // compute the || rij ||² distance between body i and body j
                const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
                
                // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
                const float dist = std::pow(rijSquared + softSquared, 3.f / 2.f); // 3 flops
                const float ai = g[jBody] / dist; // 1 flop

                // add the acceleration value into the acceleration vector: ai += || ai ||.rij
                local_acc.ax[iBody] += ai * rijx; // 2 flops
                local_acc.ay[iBody] += ai * rijy; // 2 flops
                local_acc.az[iBody] += ai * rijz; // 2 flops

                // reflexive computation
                
                // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
                const float aj = g[iBody] / dist; // 1 flop

                // add the acceleration value into the acceleration vector: ai += || ai ||.rij
                local_acc.ax[jBody] -= aj * rijx; // 2 flops
                local_acc.ay[jBody] -= aj * rijy; // 2 flops
                local_acc.az[jBody] -= aj * rijz; // 2 flops
            }
        }

        // combine threads results
        #pragma omp critical
        for (unsigned long Body = 0; Body < this->getBodies().getN(); Body++) {
            this->accelerations.ax[Body] += local_acc.ax[Body];
            this->accelerations.ay[Body] += local_acc.ay[Body];
            this->accelerations.az[Body] += local_acc.az[Body];
        }
    }
}

void SimulationNBodyOptimSIMDOMP::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
