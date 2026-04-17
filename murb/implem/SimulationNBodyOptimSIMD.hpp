/*
 * Soufiane El mouahid
 * 
 * Martin Hart
*/

#ifndef SIMULATION_N_BODY_OPTIM_SIMD_HPP_
#define SIMULATION_N_BODY_OPTIM_SIMD_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"
#include "mipp.h"

class SimulationNBodyOptimSIMD : public SimulationNBodyInterface {
  protected:
    accSoA_t<float> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyOptimSIMD(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyOptimSIMD() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_OPTIM_SIMD_HPP_ */
