class Config:
    def __init__(self):
        self.g=-2# 重力
        self.eta=0.05# 粘滞
        self.mass=0.02# 质量
        self.h=0.11# 间隔
        self.dt=0.01# 仿真步长
        self.grid_size=50# 网格数量，覆盖100*100的范围
        self.cell_size=0.31# 网格大小，范围0到10
        self.num_particles=3860# 总粒子数
        self.max_neighbour=100# 最多几个邻居
        self.up=9.5
        self.down=0.5
        self.left=0.5
        self.right=9.5