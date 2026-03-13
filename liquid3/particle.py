import taichi as ti
import config
con=config.Config()
@ti.func
def dist(a,b):
    return ti.sqrt(ti.pow(a[0]-b[0],2)+ti.pow(a[1]-b[1],2))
@ti.data_oriented
class particle:
    def __init__(self):
        self.grid_size=con.grid_size
        self.cell_size=con.cell_size
        self.h=con.h
        self.max_neighbour=con.max_neighbour# 允许的最多邻居
        self.num_particles=con.num_particles# 总粒子数
        self.pos=ti.Vector.field(2,dtype=ti.float32,shape=self.num_particles)# 位置
        self.mass=ti.field(dtype=ti.float32,shape=self.num_particles)# 这一块的质量
        self.vel=ti.Vector.field(2,dtype=ti.float32,shape=self.num_particles)# 速度
        self.fraction=ti.Vector.field(2,dtype=ti.float32,shape=self.num_particles)# 粘滞力
        self.acc=ti.Vector.field(2,dtype=ti.float32,shape=self.num_particles)# 加速
        self.pressure=ti.field(dtype=ti.float32,shape=self.num_particles)# 压力
        self.rho=ti.field(dtype=ti.float32,shape=self.num_particles)# 密度
        self.grid_keys=ti.field(dtype=int,shape=self.num_particles)# 每一个点在对应的网格坐标
        self.cell_start=ti.field(dtype=int,shape=self.grid_size*self.grid_size)# 网格起始点
        self.cell_end=ti.field(dtype=int,shape=self.grid_size*self.grid_size)# 网格结束点，左闭右开
        self.sorted_indices=ti.field(dtype=int,shape=self.num_particles)# 按网格顺序排序
        self.neighbour=ti.field(dtype=int,shape=self.num_particles*self.max_neighbour)# 每个点的邻居
        self.neighbour_count=ti.field(dtype=int,shape=self.num_particles)# 每个点有几个邻居
        self.edge=ti.field(dtype=int,shape=self.num_particles)# 是否是容器壁虚粒子
        self.avg_neighbour_count=ti.field(dtype=ti.float32,shape=())# 总平均邻居数
        self.temp_keys=ti.field(dtype=int,shape=self.num_particles)
    @ti.kernel
    def compute_grid_keys(self):
        for i in range(self.num_particles):
            x=self.pos[i][0]
            y=self.pos[i][1]
            grid_x=int(ti.floor(x/self.cell_size))
            grid_y=int(ti.floor(y/self.cell_size))
            key=grid_x+grid_y*self.grid_size
            self.grid_keys[i]=key
            self.sorted_indices[i]=i
    @ti.kernel
    def get_cell(self):
        for i,j in ti.ndrange(self.grid_size,self.grid_size):
            self.cell_start[i*self.grid_size+j]=-1
            self.cell_end[i*self.grid_size+j]=-1
        prev_key=-1
        ti.loop_config(serialize=True)
        for idx in range(self.num_particles):
            i=self.sorted_indices[idx]
            key=self.grid_keys[i]
            if key!=prev_key:
                if prev_key!=-1:
                    self.cell_end[prev_key]=idx
                prev_key=key
                self.cell_start[key]=idx
        if prev_key!=-1:
            self.cell_end[prev_key]=self.num_particles
    def debug_grid_keys(self):
        for i in ti.static([0,1,2]):
            x = self.pos[i][0]
            y = self.pos[i][1]
            gx = int(ti.floor(x / self.cell_size))
            gy = int(ti.floor(y / self.cell_size))
            key = gx + gy * self.grid_size
            print(f"i={i}, pos=({x},{y}), grid=({gx},{gy}), key={key}")
    @ti.kernel
    def find_neighbour(self):
        total_neighbour=0
        total_eff=0
        for idx in range(self.num_particles):
            gx=int(ti.floor(self.pos[idx][0]/self.cell_size))
            gy=int(ti.floor(self.pos[idx][1]/self.cell_size))
            count=0
            for dx in ti.static(range(-1,2)):
                for dy in ti.static(range(-1,2)):
                    nx=gx+dx
                    ny=gy+dy
                    if nx<self.grid_size and ny<self.grid_size and nx>=0 and ny>=0:
                        next=nx+ny*self.grid_size
                        start=self.cell_start[next]
                        end=self.cell_end[next]
                        if start!=-1:
                            for i in range(start,end):
                                new_idx=self.sorted_indices[i]
                                if new_idx!=idx and dist(self.pos[idx],self.pos[new_idx])<=2.1*self.h:
                                    if count<self.max_neighbour:
                                        self.neighbour[idx*self.max_neighbour+count]=new_idx
                                        count+=1
            self.neighbour_count[idx]=count
            total_neighbour+=count
            total_eff+=1
        self.avg_neighbour_count[None]=total_neighbour/total_eff
    def grid_update(self):
        self.compute_grid_keys()
        self.temp_keys.copy_from(self.grid_keys)
        ti.algorithms.parallel_sort(self.grid_keys,self.sorted_indices)
        self.grid_keys.copy_from(self.temp_keys)
        self.get_cell()
        self.find_neighbour()