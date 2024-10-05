import pybullet as p
import random
import time
import numpy as np
from environment.utilities import setup_sisbot

# Connect to PyBullet (in GUI mode for visualization)
p.connect(p.GUI)
# Set the simulation time step
p.setTimeStep(1./240.)

# Load the URDF model of the robot
robot_id = p.loadURDF("environment/urdf/ur5_robotiq_140.urdf", flags=p.URDF_USE_INERTIA_FROM_FILE)

joints, controlGripper, controlJoints, mimicParentName = setup_sisbot(p, robot_id, "140")

def reset_robot():
    user_parameters = (-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                        -1.5707970583733368, 0.0009377758247187636, 0.085)
    for _ in range(60):
        for i, name in enumerate(controlJoints):
            if i == 6:
                controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=user_parameters[i])
                break
            joint = joints[name]
            # control robot joints
            p.setJointMotorControl2(robot_id, joint.id, p.POSITION_CONTROL,
                                    targetPosition=user_parameters[i], force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)
            p.stepSimulation()

reset_robot()
for _ in range(300):
    p.stepSimulation()


def generate_linear_path(start_pos, end_pos, steps=100):
    return np.linspace(start_pos, end_pos, steps)

# Identify the end effector link index
end_effector_link_index = 7  # Replace with your end-effector link index

# Get the initial position and orientation of the end effector
start_position = p.getLinkState(robot_id, end_effector_link_index)[0]

joint_poses = p.calculateInverseKinematics(robot_id, end_effector_link_index, start_position)
# Set the robot's joint motors to the calculated joint positions
for i in range(len(joint_poses)):
    p.setJointMotorControl2(bodyIndex=robot_id, 
                            jointIndex=i, 
                            controlMode=p.POSITION_CONTROL, 
                            targetPosition=joint_poses[i])
    for j in range(300):
        p.stepSimulation()


for i in range(1000):
    # Generate a linear path from the start to the end position
    end_position = np.array(start_position) + (np.random.random(size=3)-0.5)*0.1
    end_position = end_position.tolist()
    path = generate_linear_path(start_position, end_position, steps=100)

    debug_id = p.addUserDebugLine(start_position, end_position, [1,0,0])

    # Move the robot's end effector along the linear path
    for target_pos in path:
        # Compute the joint angles using inverse kinematics
        joint_poses = p.calculateInverseKinematics(robot_id, end_effector_link_index, target_pos)

        # Set the robot's joint motors to the calculated joint positions
        for i in range(len(joint_poses)):
            p.setJointMotorControl2(bodyIndex=robot_id, 
                                    jointIndex=i, 
                                    controlMode=p.POSITION_CONTROL, 
                                    targetPosition=joint_poses[i])

        # Step the simulation
        p.stepSimulation()
        time.sleep(1./60.)

    p.removeUserDebugItem(debug_id)

    time.sleep(2)
    start_position = p.getLinkState(robot_id, end_effector_link_index)[0]

# Disconnect from PyBullet
p.disconnect()
