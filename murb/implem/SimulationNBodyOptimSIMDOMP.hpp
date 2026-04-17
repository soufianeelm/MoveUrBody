/*
 * Soufiane El mouahid
 * 
 * Martin Hart
*/

#ifndef SIMULATION_N_BODY_OPTIM_SIMD_OMP_HPP_
#define SIMULATION_N_BODY_OPTIM_SIMD_OMP_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"
#include "mipp.h"

class SimulationNBodyOptimSIMDOMP : public SimulationNBodyInterface {
  protected:
    accSoA_t<float> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyOptimSIMDOMP(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyOptimSIMDOMP() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_OPTIM_SIMD_OMP_HPP_ */
