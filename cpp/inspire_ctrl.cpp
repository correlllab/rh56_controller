#include "inspire.h"
#include "param.h"

#include "dds/Publisher.h"
#include "dds/Subscription.h"
#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/common/thread/recurrent_thread.hpp>

class InspireRunner
{
public:
  InspireRunner()
  {
    serial = std::make_shared<SerialPort>(param::serial_port, B115200);

    // inspire
    righthand = std::make_shared<inspire::InspireHand>(serial, 1); // ID 1
    lefthand = std::make_shared<inspire::InspireHand>(serial, 2); // ID 2

    // dds
    handcmd = std::make_shared<unitree::robot::SubscriptionBase<unitree_go::msg::dds_::MotorCmds_>>(
        "rt/" + param::ns + "/cmd");
    handcmd->msg_.cmds().resize(12);
    handstate = std::make_unique<unitree::robot::RealTimePublisher<unitree_go::msg::dds_::MotorStates_>>(
        "rt/" + param::ns + "/state");
    handstate->msg_.states().resize(12);

    // Start running
    thread = std::make_shared<unitree::common::RecurrentThread>(
      10000, std::bind(&InspireRunner::run, this)
    );
  }

  void updateHandState(inspire::InspireHand& hand, int state_offset, const std::string& hand_name)
  {
    Eigen::Matrix<double, 6, 1> qtemp;
    if(hand.GetPosition(qtemp) == 0)
    {
      qstate.block<6, 1>(state_offset, 0) = qtemp;
    }
    else
    {
      for(int i(0); i<6; i++)
      {
        handstate->msg_.states()[i + state_offset].lost()++;
      }
      // spdlog::debug("Failed to get {} hand state", hand_name);
    }
  }

  void run()
  {
    // Set command
    if(!handcmd->isTimeout())
    {
      for(int i(0); i<12; i++)
      {
        qcmd(i) = handcmd->msg_.cmds()[i].q();
      }
      righthand->SetPosition(qcmd.block<6, 1>(0, 0));
      lefthand->SetPosition(qcmd.block<6, 1>(6, 0));
    }

    // Recv state
    updateHandState(*righthand, 0, "right");
    updateHandState(*lefthand, 6, "left");

    if(handstate->trylock())
    {
        for(int i(0); i<12; i++)
        {
            handstate->msg_.states()[i].q() = qstate(i);
        }
        handstate->unlockAndPublish();
    }
  }

  unitree::common::ThreadPtr thread;

  // inspire
  SerialPort::SharedPtr serial;
  std::shared_ptr<inspire::InspireHand> lefthand;
  std::shared_ptr<inspire::InspireHand> righthand;
  Eigen::Matrix<double, 12, 1> qcmd, qstate;

  // dds
  std::unique_ptr<unitree::robot::RealTimePublisher<unitree_go::msg::dds_::MotorStates_>> handstate;
  std::shared_ptr<unitree::robot::SubscriptionBase<unitree_go::msg::dds_::MotorCmds_>> handcmd;
};

int main(int argc, char ** argv)
{
  auto vm = param::helper(argc, argv);
  unitree::robot::ChannelFactory::Instance()->Init(0, param::network);

  std::cout << " --- Unitree Robotics --- " << std::endl;
  std::cout << "  Inspire Hand Controller  " << std::endl;

  InspireRunner runner;
 
  while (true)
  {
    sleep(1);
  }
  return 0;
}