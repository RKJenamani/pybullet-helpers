"""Kinova Gen3 robots."""

from functools import cached_property
from pathlib import Path
from typing import Optional

import pybullet as p

from pybullet_helpers.ikfast import IKFastInfo
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import (
    SingleArmPyBulletRobot,
    SingleArmTwoFingerGripperPyBulletRobot,
)
from pybullet_helpers.utils import get_assets_path


class KinovaGen3NoGripperPyBulletRobot(SingleArmPyBulletRobot):
    """A Kinova Gen3 robot arm with no gripper."""

    @classmethod
    def get_name(cls) -> str:
        return "kinova-gen3-no-gripper"

    @classmethod
    def urdf_path(cls) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "kinova_no_gripper" / "GEN3_URDF_V12.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [-4.3, -1.6, -4.8, -1.8, -1.4, -1.1, 1.6]

    @property
    def end_effector_name(self) -> str:
        return "EndEffector"

    @property
    def tool_link_name(self) -> str:
        return "EndEffector_Link"


class KinovaGen3RobotiqGripperPyBulletRobot(SingleArmPyBulletRobot):
    """A Kinova Gen3 robot arm with a robotiq gripper.

    We do not inherit from SingleArmTwoFingerGripperPyBulletRobot
    because the gripper has mimic joints.
    """

    @classmethod
    def get_name(cls) -> str:
        return "kinova-gen3"

    @classmethod
    def urdf_path(cls) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "kortex_description" / "robots" / "gen3_7dof.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [-4.3, -1.6, -4.8, -1.8, -1.4, -1.1, 1.6, 0.0]

    @property
    def end_effector_name(self) -> str:
        return "tool_frame_joint"

    @property
    def tool_link_name(self) -> str:
        return "tool_frame"

    @cached_property
    def arm_joints(self) -> list[int]:
        """Add the only revolute finger to the arm joints."""
        joint_ids = super().arm_joints
        joint_ids.extend([self.finger_joint_id])
        return joint_ids

    @property
    def finger_joint_name(self) -> str:
        """The name of the revolute finger joint."""
        return "finger_joint"

    @cached_property
    def finger_joint_id(self) -> int:
        """The PyBullet joint ID for the revolute finger joint."""
        return self.joint_from_name(self.finger_joint_name)

    @cached_property
    def finger_joint_idx(self) -> int:
        """The index into the joints corresponding to the revolute finger
        joint.

        Note this is not the joint ID, but the index of the joint within
        the list of arm joints.
        """
        return self.arm_joints.index(self.finger_joint_id)

    @property
    def open_fingers_joint_value(self) -> float:
        return 0.0

    @property
    def closed_fingers_joint_value(self) -> float:
        return 0.8

    @property
    def tool_grasp_final_joint_value(self) -> float:
        return 0.5

    def open_fingers(self) -> None:
        """Execute opening the fingers."""
        self._change_fingers(self.open_fingers_joint_value)

    def close_fingers(self) -> None:
        """Execute closing the fingers."""
        self._change_fingers(self.closed_fingers_joint_value)

    def tool_grasp(self) -> None:
        """Execute tool grasp."""
        self._change_fingers(self.tool_grasp_final_joint_value)

    def _change_fingers(self, new_value: float) -> None:
        current_joints = self.get_joint_positions()
        current_joints[self.finger_joint_idx] = new_value
        self.set_motors(current_joints)

    def get_finger_state(self) -> float:
        """Get the state of the gripper fingers."""
        return p.getJointState(
            self.robot_id,
            self.finger_joint_id,
            physicsClientId=self.physics_client_id,
        )[0]

    @cached_property
    def positive_mimic_joints(self) -> list[int]:
        """Mimic joints of gripper that are positive multiplier of finger
        joint."""
        positive_mimic_joints = [
            "right_outer_knuckle_joint",
            "left_inner_knuckle_joint",
            "right_inner_knuckle_joint",
        ]
        return [
            self.joint_from_name(joint_name) for joint_name in positive_mimic_joints
        ]

    @cached_property
    def negative_mimic_joints(self) -> list[int]:
        """Mimic joints of gripper that are negative multiplier of finger
        joint."""
        negative_mimic_joints = ["left_inner_finger_joint", "right_inner_finger_joint"]
        return [
            self.joint_from_name(joint_name) for joint_name in negative_mimic_joints
        ]

    def set_joints(self, joint_positions: JointPositions) -> None:
        """Directly set the joint positions.

        Outside of resetting to an initial state, this should not be
        used with the robot that uses stepSimulation(); it should only
        be used for motion planning, collision checks, etc., in a robot
        that does not maintain state.
        """
        assert len(joint_positions) == len(self.arm_joints), (
            f"Expected {len(self.arm_joints)} joint positions, "
            f"got {len(joint_positions)}"
        )

        arm_joints_with_mimic_joints = (
            self.arm_joints + self.positive_mimic_joints + self.negative_mimic_joints
        )
        joint_positions_with_mimic_joints = (
            joint_positions
            + [joint_positions[-1]] * len(self.positive_mimic_joints)
            + [-joint_positions[-1]] * len(self.negative_mimic_joints)
        )

        for joint_id, joint_val in zip(
            arm_joints_with_mimic_joints, joint_positions_with_mimic_joints
        ):
            p.resetJointState(
                self.robot_id,
                joint_id,
                targetValue=joint_val,
                targetVelocity=0,
                physicsClientId=self.physics_client_id,
            )

    def set_motors(self, joint_positions: JointPositions) -> None:
        """Update the motors to move toward the given joint positions."""
        assert len(joint_positions) == len(self.arm_joints)

        arm_joints_with_mimic_joints = (
            self.arm_joints + self.positive_mimic_joints + self.negative_mimic_joints
        )
        joint_positions_with_mimic_joints = (
            joint_positions
            + [1.0] * len(self.positive_mimic_joints)
            + [-1.0] * len(self.negative_mimic_joints)
        )

        # Set arm joint motors.
        if self._control_mode == "position":
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=arm_joints_with_mimic_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=joint_positions_with_mimic_joints,
                physicsClientId=self.physics_client_id,
            )
        elif self._control_mode == "reset":
            self.set_joints(joint_positions)
        else:
            raise NotImplementedError(
                "Unrecognized pybullet_control_mode: " f"{self._control_mode }"
            )

    @classmethod
    def ikfast_info(cls) -> Optional[IKFastInfo]:
        return IKFastInfo(
            module_dir="kortex",
            module_name="ikfast_kortex",
            base_link="base_link",
            ee_link="tool_frame",
            free_joints=["joint_7"],
        )
