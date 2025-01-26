# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import abc
import json
import socket
import logging
import subprocess

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__file__)


class Individual(object):
    #如果后面以及前面的性能都有提升则有提升
    def __init__(self, points: list, scores: list = None):
        self.points = points
        self.scores = scores

    def __eq__(self, other):
        return np.allclose(self.points, other.points)

    def __str__(self):
        return f'{self.points} => {self.scores}'


class Evaluator(object):
    """
    Evaluator for the genetic algorithm.
    Launches a subprocess to evaluate the individuals.

    Args:
        sock (socket.socket): The socket object for communication.
        args (dict): A dictionary of arguments for the evaluator.
        device_list (list): A list of device indices.
        buf_size (int, optional): The buffer size for communication. Defaults to 4096.
    """

    def __init__(self, sock: socket.socket, args: dict, device_list: list, buf_size: int = 4096):
        self.buf_size = buf_size
        script_path = __file__.replace(os.path.join('algorithms', 'genetic_base.py'), 'evaluate_layerwise_chat.py')
        self.device_str = ','.join([str(device_idx) for device_idx in device_list])
        env_str = f'CUDA_VISIBLE_DEVICES={self.device_str}'
        script_args = ''
        for key, value in args.items():
            if type(value) is bool:
                if value:
                    script_args += f'--{key} '
            else:
                script_args += f'--{key} {value} '
        #执行脚本，在search.py中创建，每创建一个evaluator，首先启动一个进程，该进程连接相应的地址（connect函数），然后evalutor接受（accept）
        self.process = subprocess.Popen(f'{env_str} python {script_path} {script_args}', shell=True)
        self.conn, self.addr = sock.accept()
        logger.info(f'Evaluator [addr={self.addr}, device={self.device_str}] connected ')

    """
    初始化所有的脚本进程
    """
    def model_ready(self):
        assert json.loads(self.conn.recv(self.buf_size).decode())['model_ready']
        logger.info(f'Evaluator [addr={self.addr}, device={self.device_str}] model loaded')

    """
    给评价脚本设置进程
    """
    def set_points(self, points: list):
        print(json.dumps({'points': points}))
        self.conn.send(json.dumps({'points': points}).encode())
        

    """
    等待评价脚本返回相关的结果
    """
    def get_result(self) -> float:
        result = json.loads(self.conn.recv(self.buf_size).decode())['result']
        logger.debug(f'Evaluator [addr={self.addr}, device={self.device_str}] result={result}')
        return result

    """
    通知所有的评价脚本，搜索算法已经结束
    """
    def finalize(self):
        self.conn.send(json.dumps({'finalize': True}).encode())
        self.conn.close()
        # self.process.kill()


class EvaluatorQueue(object):
    """
    Queue of evaluators.

    Args:
        evaluators (list[Evaluator]): A list of evaluators.
        indvs (list[Individual]): A list of individuals.
    """

    def __init__(self, evaluators: list[Evaluator]):
        self.evaluators = evaluators
        self.indvs: list[Individual] = []

    def push(self, indv: Individual):
        """
        Pushes an individual to the queue and sets the rope arguments for the corresponding evaluator.

        Args:
            indv (Individual): The individual to be pushed to the queue.
            rope_args (dict): The rope arguments to be set for the corresponding evaluator.
        """
        idx = len(self.indvs)
        self.indvs.append(indv)
        self.evaluators[idx].set_points(indv.points)
        if len(self.indvs) >= len(self.evaluators):
            self.join()

    def join(self):
        """
        Get evaluation results and updates their PPL values.
        """
        for evaluator, indv in zip(self.evaluators, self.indvs):
            indv.scores = evaluator.get_result()
        self.indvs = []


class GeneticAlgorithm:
    """
    Genetic Algorithm for LongRoPE evolution search.

    Args:
        evaluators (list[Evaluator]): List of evaluators used to evaluate individuals.
        scale (float): Length scale.
        target_length (int): Target sequence length.
        hyper_params (dict[str, float]): Hyperparameters for the genetic algorithm.
        init_factors (np.ndarray): Initial LongRoPE rescale factors.
        rope_args (dict): Additional LongRoPE parameters.
        log_json_path (str): Path to the log file.
        output_dir (str): Directory to save the output files.
        recovery (str, optional): Path to the log file to recovery the search process. Defaults to None.
    """

    def __init__(
        self,
        evaluators: list[Evaluator],
        hyper_params: dict[str, float],
        init_points: list,
        log_json_path: str,
        output_dir: str,
        recovery: str = None,
    ):
        self.queue = EvaluatorQueue(evaluators)
        self.history: list[Individual] = []

        evo_scale = hyper_params["evo_scale"]
        self.population_size = int(evo_scale * hyper_params["population_size"])      # 种群大小
        self.max_time_budget = int(evo_scale * hyper_params["max_time_budget"])      # 迭代伦次
        self.mutation_numbers = int(evo_scale * hyper_params["mutation_numbers"])    # 变异操作数量
        self.crossover_size = int(evo_scale * hyper_params["crossover_size"])        # 交叉操作数量
        self.max_crossover_try = int(evo_scale * hyper_params["max_crossover_try"])  # 交叉重试次数
        self.parents_size = int(evo_scale * hyper_params["parents_size"])            # 亲代数量
        self.list_step = hyper_params["list_step"]                                   # 搜索空间粒度
        assert self.parents_size <= self.population_size, \
            f'Number of parents ({self.parents_size}) should not be larger than population size ({self.population_size})'
        self.init_points = init_points

        self.recovery = recovery
        self.log_json_path = log_json_path
        self.output_dir = output_dir

    def make_indv(self, points: np.ndarray) -> Individual:
        """
        Creates a new individual with the given factors.

        Args:
            factors (np.ndarray): The factors for creating the individual.

        Returns:
            Individual: The newly created individual.
        """
        indv = Individual(points)
        self.queue.push(indv)
        return indv

    @abc.abstractmethod
    def mutate(self, indv: Individual) -> Individual:
        "Generate new individual with constraints by mutatation."

    @abc.abstractmethod        
    def crossover(self, indv_1: Individual, indv_2: Individual) -> Individual:
        "Generate new individual with constraints by crossover."

    def log(self, iteration: int, population: list[Individual]):
        """
        Logs the iteration number, population, and history to a JSON file and
        saves the factors of the best individual in the population to a CSV file.

        Args:
        - iteration (int): The current iteration number.
        - population (list[Individual]): The list of individuals in the population.
        """
        with open(self.log_json_path, 'w') as file:
            file.write(json.dumps(
                {
                    'iteration': iteration,
                    'population': [[indv.points, indv.scores] for indv in population],
                    'history': [[indv.points, indv.scores] for indv in self.history],
                },
                indent = 4,
            ))
        np.savetxt(
            os.path.join(self.output_dir, f"result_it{iteration:0>3d}.csv"), population[0].points,
            delimiter='\n',
        )

    def run_genetic_algorithm(self):
        "Main loop of Genetic Algorithm."
        print("starting。。。。。。。。。。。。。。。")
        if self.recovery is None:
            population = []
            latest_iteration = 0
            pbar = tqdm(range(self.population_size), desc=f'Generate Initial Population')
            indv = self.make_indv(self.init_points)
            for i in pbar:
                if i == 0:
                    new_indv = indv
                else:
                    new_indv = self.mutate(indv)

                population.append(new_indv)
                self.history.append(new_indv)
                if new_indv.scores is not None:
                    pbar.set_postfix(last_score=new_indv.scores[1])
                logger.debug(f'[Population #{i:0>3d}] {new_indv}')
        else:
            logger.info(f"Recover from {self.recovery}")
            with open(self.recovery) as f:
                data = json.loads(f.read())
            latest_iteration = data['iteration']
            population = [Individual(points, scores) for points, scores in data['population']]
            if "history" in data:
                self.history = [Individual(points, scores) for points, scores in data['history']]

        self.queue.join()

        population_str = '\n'.join(map(str, population))
        logger.debug(f"Iteration #{latest_iteration}\nPopulation:\n{population_str}")
        logger.info("Start Evolution Search")

        best_indv = Individual(None, [-1, -1])
        best_score_records = []

        for i in range(latest_iteration, latest_iteration + self.max_time_budget):
            #根据末尾分数 选择一部分个体作为父代,分数高的是最好的
            # parents = sorted(population, key=lambda x: -x.scores[1])[:self.parents_size]
            parents = sorted(population, key=lambda x: (-x.scores[1]*1.2-x.scores[0],-x.scores[1],-x.scores[0]))[:self.parents_size]

            self.log(i, parents)
            current_best_indv = parents[0]
            best_score_records.append(current_best_indv.scores)

            logger.info(f"[Iter #{i + 1:0>3d} Best] {current_best_indv}")

            if current_best_indv.scores[1] < best_indv.scores[1]:
                best_indv = current_best_indv

            population = parents

            # 变异
            pbar = tqdm(range(self.mutation_numbers), desc=f'Iter #{i + 1:0>3d} Mutation')
            for j in pbar:
                idx = np.random.randint(self.parents_size)
                mutated_indv = self.mutate(parents[idx])
                population.append(mutated_indv)
                if mutated_indv.scores is not None:
                    pbar.set_postfix(end_score=mutated_indv.scores[1])
                logger.debug(f'[Mutate #{i:0>3d} / #{j:0>3d}] {parents[idx]} / {mutated_indv}')

            self.queue.join()

            # 交叉
            pbar = tqdm(range(self.crossover_size), desc=f'Iter #{i + 1:0>3d} Crossover')
            for j in pbar:
                idx1, idx2 = np.random.choice(self.parents_size, 2, replace=False)
                idx1, idx2 = sorted([idx1, idx2])
                crossover_indv = self.crossover(parents[idx1], parents[idx2])
                if crossover_indv is None:
                    logger.debug(f'Crossover reach max {self.max_crossover_try} trys. Mutate from parent #1 instead.')
                    crossover_indv = self.mutate(parents[idx1])
                population.append(crossover_indv)
                if crossover_indv.scores is not None:
                    pbar.set_postfix(end_score=crossover_indv.scores[1])
                logger.debug(f'[Crossover #{i:0>3d} / #{j:0>3d}] From #{idx1:0>3d} + {idx2:0>3d}')

            self.queue.join()

        final_population = sorted(population, key=lambda x: -x.scores[1])[:self.parents_size]
        self.log(i, final_population)
        logger.info(f"score curve: {best_score_records}")
        return final_population[0].points
