/*
 * Soufiane El mouahid
 * 
 * Martin Hart
*/

#ifndef SIMULATION_N_BODY_CUDA_HPP_
#define SIMULATION_N_BODY_CUDA_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyCUDA : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyCUDA(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyCUDA() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_CUDA_HPP_ */
