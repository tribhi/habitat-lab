#!/usr/bin/env python3

import random
from typing import Optional, cast

import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.tasks.rearrange.multi_task.pddl_task import PddlTask
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_task import NavToInfo

from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


@registry.register_task(name="PointNavPddlSocialNavTask-v1")
class TestSocialNavTask(PddlTask):
    """based on PDDL: Planning Domain Description Language"""
    
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)
    
    
