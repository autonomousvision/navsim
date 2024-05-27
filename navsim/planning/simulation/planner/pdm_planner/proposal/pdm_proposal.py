from typing import List
from dataclasses import dataclass

from shapely.geometry import LineString

from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


@dataclass
class PDMProposal:
    """Dataclass for storing proposal information."""

    proposal_idx: int
    lateral_idx: int
    longitudinal_idx: int
    path: PDMPath

    @property
    def linestring(self) -> LineString:
        """Getter for linestring of proposal's path."""
        return self.path.linestring

    @property
    def length(self):
        """Getter for length [m] of proposal's path."""
        return self.path.length


class PDMProposalManager:
    """Class to store and manage lateral and longitudinal combination of proposals."""

    def __init__(
        self,
        lateral_proposals: List[PDMPath],
        longitudinal_policies: BatchIDMPolicy,
    ):
        """
        Constructor for PDMProposalManager
        :param lateral_proposals: list of path's to follow
        :param longitudinal_policies: IDM policy class (batch-wise)
        """

        self._num_lateral_proposals: int = len(lateral_proposals)
        self._num_longitudinal_proposals: int = longitudinal_policies.num_policies
        self._longitudinal_policies: BatchIDMPolicy = longitudinal_policies

        self._proposals: List[PDMProposal] = []
        proposal_idx = 0

        for lateral_idx in range(self._num_lateral_proposals):
            for longitudinal_idx in range(self._num_longitudinal_proposals):
                self._proposals.append(
                    PDMProposal(
                        proposal_idx=proposal_idx,
                        lateral_idx=lateral_idx,
                        longitudinal_idx=longitudinal_idx,
                        path=lateral_proposals[lateral_idx],
                    )
                )
                proposal_idx += 1

    def __len__(self) -> int:
        """Returns number of proposals (paths x policies)."""
        return len(self._proposals)

    def __getitem__(self, proposal_idx) -> PDMProposal:
        """
        Returns the requested proposal.
        :param proposal_idx: index for each proposal
        :return: PDMProposal dataclass
        """
        return self._proposals[proposal_idx]

    def update(self, speed_limit_mps: float) -> None:
        """
        Updates target velocities of IDM policies with current speed-limit.
        :param speed_limit_mps: current speed-limit [m/s]
        """
        self._longitudinal_policies.update(speed_limit_mps)

    @property
    def num_lateral_proposals(self) -> int:
        return self._num_lateral_proposals

    @property
    def num_longitudinal_proposals(self) -> int:
        return self._longitudinal_policies._num_longitudinal_proposals

    @property
    def max_target_velocity(self) -> float:
        return self._longitudinal_policies.max_target_velocity

    @property
    def longitudinal_policies(self) -> BatchIDMPolicy:
        return self._longitudinal_policies
