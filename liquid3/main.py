import taichi as ti
ti.init(arch=ti.cuda)
import sphsolver as sph
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import particle
import config
import numpy as np
con=config.Config()
N=30
p=particle.particle()
@ti.kernel
def init():
    for i,j in ti.ndrange(N,N):
        p.pos[i*N+j]=[i*0.05+4,j*0.05+0.5]
        p.edge[i*N+j]=0
        p.rho[i*N+j]=8
        p.vel[i*N+j]=[0,0]
        p.mass[i*N+j]=con.mass
        p.pressure[i*N+j]=0
@ti.kernel
def add_edge():
    for i in range(200):
        for j in range(5):
            idx=N*N+i+j*200
            p.pos[idx]=[i*0.05,0.05*j]
            p.rho[idx]=8
            p.vel[idx]=[0,0]
            p.mass[idx]=con.mass
            p.edge[idx]=1
    for i in range(196):
        for j in range(5):
            idx=N*N+i+5*200+196*j
            p.pos[idx]=[2.5+0.05*j,10-i*0.05]
            p.rho[idx]=8
            p.vel[idx]=[0,0]
            p.mass[idx]=con.mass
            p.edge[idx]=1
    for i in range(196):
        for j in range(5):
            idx=N*N+i+5*200+196*(j+5)
            p.pos[idx]=[7-0.05*j,10-i*0.05]
            p.rho[idx]=8
            p.vel[idx]=[0,0]
            p.mass[idx]=con.mass
            p.edge[idx]=1
            
def animate():
    fig,ax=plt.subplots(figsize=(8,6))
    pos=p.pos.to_numpy()
    scatter=ax.scatter(pos[:,0],pos[:,1],c='b',s=1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.set_title("SPH Simulation")
    def update(frame):
        sph.solve()
        global pos
        pos=p.pos.to_numpy()
        scatter.set_offsets(pos)
        return scatter,
    ani=animation.FuncAnimation(fig,update,frames=1000,interval=30,blit=True)
    plt.show()
def taichi_ui():
    
if __name__ == "__main__":
    init()
    add_edge()
    p.grid_update()
    sph.replace_particle(p)
    sph.first_rho()
    sph.update_new_rho()
    sph.solve()
    for i in range(p.num_particles):
        print(f"{i} par rho={p.rho[i]} in {p.pos[i][0],p.pos[i][1]},pressure={p.pressure[i]}")
    animate()
