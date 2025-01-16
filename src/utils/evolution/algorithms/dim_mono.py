# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import numpy as np
from copy import deepcopy
from evolution.algorithms import GeneticAlgorithm, Individual


logger = logging.getLogger(__file__)


class DimMonoGeneticAlgorithm(GeneticAlgorithm):

    def mutate(self, indv: Individual) -> Individual:
        list_step = self.list_step
        # evo_list = np.arange(1.0, 1.0 + self.scale + list_step, list_step)
        new_points = indv.points
        new_points = deepcopy(new_points)
        y_max=2.5
        y_min=1
        area_size = 0.2
        while (Individual(new_points) in self.history):
            for dim in range(len(new_points)):
                cur_x = new_points[dim][0]
                cur_y = new_points[dim][1]
                if np.random.rand() < 0.3:
                    if dim == 0:
                        x_evo_list_curr = np.arange(0.0, new_points[dim + 1][0], list_step)
                        y_evo_list_curr = np.arange(max(y_min, cur_y-area_size),min(y_max,cur_y+area_size),list_step)
                    elif dim == len(new_points) - 1:
                        x_evo_list_curr = np.arange(new_points[dim - 1][0], cur_x+area_size, list_step)
                        y_evo_list_curr = np.arange(max(y_min, cur_y-area_size),min(y_max,cur_y+area_size),list_step)
                    else:
                        x_evo_list_curr = np.arange(new_points[dim - 1][0], new_points[dim + 1][0], list_step)
                        y_evo_list_curr = np.arange(max(y_min, cur_y-area_size),min(y_max,cur_y+area_size),list_step)

                    if len(x_evo_list_curr) > 0:
                        x_layer_index = np.random.randint(0, x_evo_list_curr.shape[0])
                        y_layer_index = np.random.randint(0, y_evo_list_curr.shape[0])
                        new_points[dim][0] = x_evo_list_curr[x_layer_index]
                        new_points[dim][1] = y_evo_list_curr[y_layer_index]

        indv = self.make_indv(new_points)
        self.history.append(indv)
        return indv

    def crossover(self, indv_1: Individual, indv_2: Individual) -> Individual:
        par_points_1 = deepcopy(indv_1.points)
        par_points_2 = deepcopy(indv_2.points)
        if np.allclose(par_points_1, par_points_2):
            return None
        new_points = par_points_1.copy()
        for _ in range(self.max_crossover_try):
            for i in range(len(new_points)):
                if np.random.rand() < 0.3:
                    new_points = new_points.copy()
                    if np.random.rand() < 0.5:
                        new_points[i] = par_points_2[i]
                    #保证x的坐标是递增的
                    x_array = []
                    for i in range(len(new_points)):
                        x_array.append(new_points[i][0])
                    x_array = np.array(x_array)
                    
                    if (Individual(new_points) in self.history) or (not np.all(np.diff(x_array) >= 0)):
                        continue
                    indv = self.make_indv(new_points)
                    self.history.append(indv)
                    return indv
        return None
    