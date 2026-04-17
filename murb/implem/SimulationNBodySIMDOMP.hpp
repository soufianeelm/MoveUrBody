/*
 * Soufiane El mouahid
 * 
 * Martin Hart
*/

#ifndef SIMULATION_N_BODY_SIMD_OMP_HPP_
#define SIMULATION_N_BODY_SIMD_OMP_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"
#include "mipp.h"

class SimulationNBodySIMDOMP : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodySIMDOMP(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodySIMDOMP() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_SIMD_OMP_HPP_ */
