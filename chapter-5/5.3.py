import random
import numpy as np


def log(i, a, b):
    print("epoch --> ",
          str(i + 1).rjust(5, " "), " max:",
          str(round(a, 4)).rjust(8, " "), "mean:",
          str(round(b, 4)).rjust(8, " "), "alpha:",
          str(round(a / b, 4)).rjust(8, " ")
          )


class GeneSolve:
    def __init__(self, pop_size, epoch, cross_prob, mutate_prob, alpha, print_batch=10):
        self.pop_size = pop_size
        self.epoch = epoch
        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob
        self.print_batch = print_batch
        self.alpha = alpha
        self.width = 11
        self.best = None

        # 产生初始种群
        self.genes = np.array(
            [''.join([random.choice(['0', '1']) for i in range(self.width)]) for j in range(self.pop_size)]
        )

    def inter_cross(self):
        """对染色体进行交叉操作"""
        ready_index = list(range(self.pop_size))

        while len(ready_index) >= 2:
            d1 = random.choice(ready_index)
            ready_index.remove(d1)
            d2 = random.choice(ready_index)
            ready_index.remove(d2)

            if np.random.uniform(0, 1) <= self.cross_prob:
                loc = random.choice(range(1, self.width - 1))
                d1_a, d1_b = self.genes[d1][0:loc], self.genes[d1][loc:]
                d2_a, d2_b = self.genes[d2][0:loc], self.genes[d2][loc:]
                self.genes[d1] = d1_a + d2_b
                self.genes[d2] = d2_a + d1_b

    def mutate(self):
        """基因突变"""
        ready_index = list(range(self.pop_size))
        for i in ready_index:
            if np.random.uniform(0, 1) <= self.mutate_prob:
                loc = random.choice(range(0, self.width))
                t0 = list(self.genes[i])
                t0[loc] = str(1 - int(self.genes[i][loc]))
                self.genes[i] = ''.join(t0)

    def get_adjust(self):
        """计算适应度"""
        x = self.get_decode()
        return x * np.sin(x) + 12

    def get_decode(self):
        return np.array([int(x, 2) * 12.55 / (2 ** 11 - 1) for x in self.genes])

    def cycle_select(self):
        """通过轮盘赌来进行选择"""
        adjusts = self.get_adjust()
        if self.best is None or np.max(adjusts) > self.best[1]:
            self.best = self.genes[np.argmax(adjusts)], np.max(adjusts)
        p = adjusts / np.sum(adjusts)
        cu_p = []

        for i in range(self.pop_size):
            cu_p.append(np.sum(p[0:i]))
        cu_p = np.array(cu_p)
        r0 = np.random.uniform(0, 1, self.pop_size)
        sel = [max(list(np.where(r > cu_p)[0]) + [0]) for r in r0]

        # 保留最优的个体
        if np.max(adjusts[sel]) < self.best[1]:
            self.genes[sel[np.argmin(adjusts[sel])]] = self.best[0]
        self.genes = self.genes[sel]

    def evolve(self):
        for i in range(self.epoch):
            self.cycle_select()
            self.inter_cross()
            self.mutate()
            a, b = np.max(gs.get_adjust()), np.mean(gs.get_adjust())
            if i % self.print_batch == self.print_batch - 1 or i == 0:
                log(i, a, b)
            if a / b < self.alpha:
                log(i, a, b)
                print("进化终止，算法已收敛！共进化 ", i + 1, " 代！")
                break


gs = GeneSolve(100, 500, 0.85, 0.1, 1.02, 100)
gs.evolve()
# epoch -->      1  max:   19.915 mean:  13.2785 alpha:   1.4998
# epoch -->    100  max:  19.9167 mean:  19.2524 alpha:   1.0345
# epoch -->    136  max:  19.9167 mean:  19.6173 alpha:   1.0153
# 进化终止，算法已收敛！共进化  136  代！

print(gs.best)
# ('10100010101', 19.916705125479506)
