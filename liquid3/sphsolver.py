import taichi as ti
import particle as part
import config
PI=3.14159265358
con=config.Config()
gamma=7.0
h=con.h
eta=con.eta
dt=con.dt
N=con.num_particles
p=part.particle()
f_ext=ti.Vector([0.0,con.g])
new_rho=ti.field(dtype=ti.float32,shape=N)
grad_p=ti.Vector.field(2,dtype=ti.float32,shape=N)
laplace_v=ti.Vector.field(2,dtype=ti.float32,shape=N)
def replace_particle(particles):
    global p
    p=particles
@ti.func
def W(r):
    q=r/h
    res=0.0
    if q<=1.0:
        res=1.0-3.0/2.0*q*q+3.0/4.0*q*q*q
    elif q<=2.0:
        res=1.0/4.0*(2.0-q)*(2.0-q)*(2.0-q)
    return res*10.0/(7*PI*h*h)
@ti.func
def dW(r):
    q=r/h
    res=0.0
    if q<=1.0:
        res=-3.0*q+9.0/4.0*q*q
    elif q<=2.0:
        res=-3.0/4.0*(2.0-q)*(2.0-q)
    return res*10.0/(7*PI*h*h*h)
@ti.kernel
def first_rho():
    for i in range(p.num_particles):
        new_rho[i]=0
        temp=0.0
        for j in range(p.neighbour_count[i]):
            nb=p.neighbour[i*p.max_neighbour+j]
            r=(p.pos[i]-p.pos[nb]).norm()+0.01*h
            new_rho[i]+=p.mass[nb]*W(r)
            temp+=p.mass[nb]/p.rho[nb]*W(r)
        new_rho[i]/=temp
@ti.kernel
def update_rho():
    for i in range(N):
        delta=0.0
        temp=0.0
        new_rho[i]=p.rho[i]
        for j in range(p.neighbour_count[i]):
            nb=p.neighbour[i*p.max_neighbour+j]
            r=(p.pos[i]-p.pos[nb]).norm()+0.01*h
            delta+=p.mass[nb]*(p.vel[i]-p.vel[nb]).dot(p.pos[i]-p.pos[nb])/r*dW(r)
            temp+=p.mass[nb]/p.rho[nb]*W(r)
        new_rho[i]+=dt*delta
@ti.kernel
def update_pressure():
    for i in range(N):
        if p.neighbour_count[i]==0:
            p.pressure[i]=3.33
        else:
            p.pressure[i]=max(0,100*8.0/(3.0*gamma)*(ti.pow(p.rho[i]/8.0,gamma)-1))
@ti.kernel
def cal_grad_p():
    for i in range(N):
        grad_p[i]=0
        temp=0.0
        for j in range(p.neighbour_count[i]):
            nb=p.neighbour[j+i*p.max_neighbour]
            r=(p.pos[i]-p.pos[nb]).norm()+0.01*h
            grad_p[i]+=p.mass[nb]*(p.pos[i]-p.pos[nb])/r*dW(r)*(p.pressure[i]/(p.rho[i]*p.rho[i])+p.pressure[nb]/(p.rho[nb]*p.rho[nb]))
            temp+=p.mass[nb]/p.rho[nb]*W(r)
@ti.kernel
def cal_laplace_v():
    for i in range(N):
        laplace_v[i]=0
        temp=0.0
        for j in range(p.neighbour_count[i]):
            nb=p.neighbour[j+i*p.max_neighbour]
            r=(p.pos[i]-p.pos[nb]).norm()+0.01*h
            laplace_v[i]+=p.mass[nb]*4*con.eta/(p.rho[i]+p.rho[nb])*(p.vel[i]-p.vel[nb]).dot(p.pos[i]-p.pos[nb])/(r*r)*(p.pos[i]-p.pos[nb])/r*dW(r)
            temp+=p.mass[nb]/p.rho[nb]*W(r)
@ti.kernel
def cal_a():
    for i in range(N):
        beta=0
        if p.neighbour_count[i]>5:
            p.acc[i]=-grad_p[i]+laplace_v[i]+f_ext
        else:
            p.acc[i]=f_ext-grad_p[i]
@ti.kernel
def update():
    for i in range(N):
        if p.edge[i]==0:
            p.vel[i]+=dt*p.acc[i]
            p.pos[i]+=dt*p.vel[i]
        else:
            p.vel[i]=0
@ti.kernel
def update_new_rho():
    for i in range(N):
        if p.edge[i]==0:
            p.rho[i]=new_rho[i]
        else:
            if new_rho[i]>8:
                p.rho[i]=new_rho[i]
def solve():
    p.grid_update()
    update_rho()
    update_new_rho()
    update_pressure()
    cal_grad_p()
    cal_laplace_v()
    cal_a()
    update()