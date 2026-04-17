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

#include "SimulationNBodySIMD.hpp"

SimulationNBodySIMD::SimulationNBodySIMD(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = (float)this->getBodies().getN() * (((float)this->getBodies().getN() / (float)mipp::N<float>()) * 72.f) + ((float)(this->getBodies().getN() % mipp::N<float>()) * 18.f) + (float)this->getBodies().getN() + 1.0;
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodySIMD::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodySIMD::computeBodiesAcceleration()
{
    // set 'index' to start of bodies that don't fill a full SIMD register; if Nbodies < SIMD size, use scalar loop only
    unsigned long Nbodies = this->getBodies().getN(), index = ((Nbodies + mipp::N<float>() - 1) / mipp::N<float>()) * mipp::N<float>();

    const dataSoA_t<float> &d = this->getBodies().getDataSoA();

    const dataSoA_t<float>* tmp_ptr;
    dataSoA_t<float> tmp;
    if (index > Nbodies) {
        tmp.qx = std::vector<float>(index);
        tmp.qy = std::vector<float>(index);
        tmp.qz = std::vector<float>(index);
        std::copy(d.qx.begin(), d.qx.end(), tmp.qx.begin());
        std::copy(d.qy.begin(), d.qy.end(), tmp.qy.begin());
        std::copy(d.qz.begin(), d.qz.end(), tmp.qz.begin());

        tmp_ptr = &tmp;
    } else {
        tmp_ptr = &d;
    }

    // compute G.mn in advance
    std::vector<float> g(index, 0.0f); 
    // flops = n
    for (unsigned long iBody = 0; iBody < Nbodies; iBody++) {
        g[iBody] = this->G * d.m[iBody]; // 1 flop
    }

    // compute e²
    const float softSquared = this->soft * this->soft; // 1 flop

    // compute e² register
    mipp::Reg<float> softSquared_r = softSquared;

    // flops = n((n / mipp::N<float>()) * 72) + ((n % mipp::N<float>()) * 18)
    for (unsigned long iBody = 0; iBody < Nbodies; iBody++) {
        mipp::Reg<float> ax = 0.0f, ay = 0.0f, az = 0.0f;

        // flops = ((n / mipp::N<float>()) * 72) + ((n % mipp::N<float>()) * 18)
        for (unsigned long jBody = 0; jBody < Nbodies; jBody += mipp::N<float>()) {
            mipp::Reg<float> rijx_r = mipp::load(&(*tmp_ptr).qx[jBody]) - (*tmp_ptr).qx[iBody]; // 4 flops
            mipp::Reg<float> rijy_r = mipp::load(&(*tmp_ptr).qy[jBody]) - (*tmp_ptr).qy[iBody]; // 4 flops
            mipp::Reg<float> rijz_r = mipp::load(&(*tmp_ptr).qz[jBody]) - (*tmp_ptr).qz[iBody]; // 4 flops

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
        }

        // add the acceleration value into the acceleration vector: ai += || ai ||.rij
        this->accelerations[iBody].ax += ax.sum();
        this->accelerations[iBody].ay += ay.sum();
        this->accelerations[iBody].az += az.sum();
    }
}

void SimulationNBodySIMD::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}