/*
 * Soufiane El mouahid
 * 
 * Martin Hart
*/

#ifndef SIMULATION_N_BODY_POSIX_HPP_
#define SIMULATION_N_BODY_POSIX_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyPosix : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyPosix(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyPosix() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_POSIX_HPP_ */
