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

#include "SimulationNBodyOptimCUDA.hpp"

__global__ void kernel_optimized(const dataAoS_t<float>* d, accAoS_t<float>* accelerations, const unsigned long Nbodies, const float softSquared, const float* g)
{
    int thread_x = threadIdx.x; // Thread's position in the block on X
    int thread_y = threadIdx.y; // Thread's position in the block on Y

    unsigned long iBody = blockIdx.x * blockDim.x + thread_x; // Thread's position in the grid on X
    unsigned long jBody = blockIdx.y * blockDim.y + thread_y; // Thread's position in the grid on Y

    // in the case where the blocks division doesn't perfectly fit the Nbodies matrix
    if (iBody < Nbodies - 1 && jBody < Nbodies && jBody > iBody) {
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

        // reflexive computation
        
        // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
        const float ai_1 = g[iBody] / dist; // 1 flop

        // add the acceleration value into the acceleration vector: ai += || ai ||.rij
        atomicAdd(&accelerations[jBody].ax, -ai_1 * rijx); // 2 flops
        atomicAdd(&accelerations[jBody].ay, -ai_1 * rijy); // 2 flops
        atomicAdd(&accelerations[jBody].az, -ai_1 * rijz); // 2 flops
    }
}

SimulationNBodyOptimCUDA::SimulationNBodyOptimCUDA(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = ( 25.f * (float)this->getBodies().getN()-1 * (float)this->getBodies().getN() * 1 / 2 ) + (float)this->getBodies().getN() + 1.0;
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyOptimCUDA::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyOptimCUDA::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &h_d = this->getBodies().getDataAoS();
    const unsigned long Nbodies = this->getBodies().getN();

    // compute G.mn in advance
    std::vector<float> h_g(Nbodies); 
    // flops = n
    for (unsigned long iBody = 0; iBody < Nbodies; iBody++) {
        h_g[iBody] = this->G * h_d[iBody].m; // 1 flop
    }

    // compute e²
    const float softSquared = this->soft * this->soft; // 1 flop    

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
    int tile_width = 16;
    int tile_height = 16;    
    dim3 blockDim(tile_width, tile_height);
    dim3 gridDim((Nbodies + tile_width - 1) / tile_width, (Nbodies + tile_height - 1) / tile_height);

    // flops = n(n-1)/2 * 25
    kernel_optimized<<<gridDim, blockDim>>>(d_d, d_accelerations, Nbodies, softSquared, d_g);

    // Wait for the kernel to be done
    cudaDeviceSynchronize();

    // retrieve computed accelerations from gpu
    cudaMemcpy(&this->accelerations[0], d_accelerations, Nbodies * sizeof(accAoS_t<float>), cudaMemcpyDeviceToHost);

    // free allocated memory on gpu
    cudaFree(d_d);
    cudaFree(d_accelerations);
    cudaFree(d_g);
}

void SimulationNBodyOptimCUDA::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}