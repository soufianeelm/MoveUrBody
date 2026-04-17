/*
 * Soufiane El mouahid
 * 
 * Martin Hart
*/

#ifndef SIMULATION_N_BODY_SIMD_OMP_CUDA_HPP_
#define SIMULATION_N_BODY_SIMD_OMP_CUDA_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"
#include "mipp.h"

class SimulationNBodySIMDOMPCUDA : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodySIMDOMPCUDA(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodySIMDOMPCUDA() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_SIMD_OMP_CUDA_HPP_ */
