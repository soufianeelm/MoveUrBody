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

#include "SimulationNBodySIMDOMPCUDA.hpp"

__global__ void kernel_hetero_op(const dataAoS_t<float>* d, accAoS_t<float>* accelerations, const unsigned long Nbodies, const float softSquared, const float* g)
{
    int thread_x = threadIdx.x; // Thread's position in the block on X
    int thread_y = threadIdx.y; // Thread's position in the block on Y

    int iBody = blockIdx.x * blockDim.x + thread_x; // Thread's position in the grid on X
    int jBody = blockIdx.y * blockDim.y + thread_y; // Thread's position in the grid on Y

    // in the case where the blocks division doesn't perfectly fit the Nbodies matrix
    if (iBody < Nbodies && jBody < Nbodies) {
        const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
        const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
        const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

        // compute the || rij ||² distance between body i and body j
        const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops

        // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
        const float dist = std::pow(rijSquared + softSquared, 3.f / 2.f); // 3 flops
        const float ai_0 = g[jBody] / dist; // 1 flop

        // add the acceleration value into the acceleration vector: ai += || ai ||.rij
        atomicAdd(&accelerations[iBody].ax, ai_0 * rijx); // 2 flops
        atomicAdd(&accelerations[iBody].ay, ai_0 * rijy); // 2 flops
        atomicAdd(&accelerations[iBody].az, ai_0 * rijz); // 2 flops
    }
}

SimulationNBodySIMDOMPCUDA::SimulationNBodySIMDOMPCUDA(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = ((float)nBodies - ((float)nBodies * 24.f / 32.f + 24.f - 1.f) * 24.f) * (((float)nBodies + (float)mipp::N<float>() - 1 / (float)mipp::N<float>()) * 72.f) + ((float)nBodies * 24.f / 32.f + 24.f - 1.f) * 24.f * 18.f  + (float)this->getBodies().getN() + 1.0;;
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodySIMDOMPCUDA::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodySIMDOMPCUDA::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &h_d = this->getBodies().getDataAoS();
    const unsigned long Nbodies = this->getBodies().getN(), index = ((Nbodies + mipp::N<float>() - 1) / mipp::N<float>()) * mipp::N<float>();

    // compute G.mn in advance
    std::vector<float> h_g(index, 0.0f); 
    // flops = n
    for (unsigned long iBody = 0; iBody < Nbodies; iBody++) {
        h_g[iBody] = this->G * h_d[iBody].m; // 1 flop
    }

    // compute e²
    const float softSquared = this->soft * this->soft; // 1 flop    

    // gpu part
    dataAoS_t<float>* d_d;
    accAoS_t<float>* d_accelerations;
    float* d_g;

    // allocate memory on gpu side
    cudaMalloc((void**)&d_d, Nbodies * sizeof(dataAoS_t<float>));
    cudaMalloc((void**)&d_accelerations, Nbodies * sizeof(accAoS_t<float>));
    cudaMalloc((void**)&d_g, Nbodies * sizeof(float));
    cudaMemset(d_accelerations, 0, Nbodies * sizeof(accAoS_t<float>));

    // send data to gpu
    cudaMemcpy(d_d, &h_d[0], Nbodies * sizeof(dataAoS_t<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, &h_g[0], Nbodies * sizeof(float), cudaMemcpyHostToDevice);

    // kernel launch configuration
    int tile_width = 24;
    int tile_height = 32;
    dim3 blockDim(tile_width, tile_height);
    dim3 gridDim((Nbodies * 24 / 32 + tile_width - 1) / tile_width, (Nbodies + tile_height - 1) / tile_height);

    // flops = n² * 18
    kernel_hetero_op<<<gridDim, blockDim>>>(d_d, d_accelerations, Nbodies, softSquared, d_g);

    // cpu part
    
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

    // compute e² register
    mipp::Reg<float> softSquared_r = softSquared;

    // flops = (n - gridDim.x * tile_width) * ((n + mipp::N<float>() - 1 / mipp::N<float>()) * 72)
    #pragma omp parallel
    {   
        std::vector<accAoS_t<float>> local_acc(Nbodies, {0.0f, 0.0f, 0.0f});

        #pragma omp for schedule(runtime)
        for (unsigned long iBody = gridDim.x * tile_width; iBody < Nbodies; iBody++) {
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
                mipp::Reg<float> mass_0_r = &h_g[jBody];
                mipp::Reg<float> ai_r = mass_0_r / dist_r; // 4 flops
                // add the acceleration value into the acceleration coordinates: ai += || ai ||.rij
                ax += ai_r * rijx_r; // 8 flops
                ay += ai_r * rijy_r; // 8 flops
                az += ai_r * rijz_r; // 8 flops
            }

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            local_acc[iBody].ax += ax.sum();
            local_acc[iBody].ay += ay.sum();
            local_acc[iBody].az += az.sum();
        }

        // combine threads results
        #pragma omp critical
        for (unsigned long Body = gridDim.x * tile_width; Body < Nbodies; Body++) {
            this->accelerations[Body].ax += local_acc[Body].ax;
            this->accelerations[Body].ay += local_acc[Body].ay;
            this->accelerations[Body].az += local_acc[Body].az;
        }
    }

    // Wait for the kernel to be done
    cudaDeviceSynchronize();

    // retrieve computed accelerations from gpu
    std::vector<accAoS_t<float>> h_tmp(Nbodies, {0.0f, 0.0f, 0.0f});
    cudaMemcpy(&h_tmp[0], d_accelerations, Nbodies * sizeof(accAoS_t<float>), cudaMemcpyDeviceToHost);

    for (unsigned long Body = 0; Body < Nbodies; Body++) {
        this->accelerations[Body].ax += h_tmp[Body].ax;
        this->accelerations[Body].ay += h_tmp[Body].ay;
        this->accelerations[Body].az += h_tmp[Body].az;
    }

    // free allocated memory on gpu
    cudaFree(d_d);
    cudaFree(d_accelerations);
    cudaFree(d_d);
}

void SimulationNBodySIMDOMPCUDA::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}