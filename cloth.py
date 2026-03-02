import taichi as ti
import numpy as np
import time
LENGTH=800
WIDTH=600
ti.init(arch=ti.cuda)
k=1000
dt=0.01
alpha=0.9
gamma=0.2
EPS=0.0001
ol1=1
ol2=ti.sqrt(2)
N=100
pos=ti.Vector.field(3,dtype=ti.float32,shape=N*N)
speed=ti.Vector.field(3,dtype=ti.float32,shape=N*N)
new_pos=ti.Vector.field(3,dtype=ti.float32,shape=N*N)
new_speed=ti.Vector.field(3,dtype=ti.float32,shape=N*N)
next_new_pos=ti.Vector.field(3,dtype=ti.float32,shape=N*N)
next_new_speed=ti.Vector.field(3,dtype=ti.float32,shape=N*N)
tri_indices=ti.field(dtype=ti.i32,shape=(2*(N-1)*(N-1)*3))
@ti.kernel
def init():
    for i,j in ti.ndrange(N,N):
        pos[i*N+j]=[i,j,0]
        speed[i*N+j]=[0,0,0]
        new_pos[i*N+j]=[i,j,0]
        new_speed[i*N+j]=[0,0,0]
        next_new_pos[i*N+j]=[i,j,0]
        next_new_speed[i*N+j]=[0,0,0]
@ti.kernel
def tri_init():
    for i,j in ti.ndrange(N-1,N-1):
        idx=(i*(N-1)+j)*2*3
        i0=i*N+j
        tri_indices[idx]=i0
        tri_indices[idx+1]=i0+1
        tri_indices[idx+2]=i0+N
        tri_indices[idx+3]=i0+1
        tri_indices[idx+4]=i0+N+1
        tri_indices[idx+5]=i0+N
@ti.kernel
def push():
    for i,j in ti.ndrange((75,100),(0,100)):
        speed[i*N+j][0]=1+speed[i*N+j][0]
        new_speed[i*N+j][0]=1+new_speed[i*N+j][0]
@ti.func
def dist(a,b):
    return ti.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)
@ti.func
def fx(a,b,ol):
    return k*(b[0]-a[0])*(1-ol/dist(a,b))
@ti.func
def fy(a,b,ol):
    return k*(b[1]-a[1])*(1-ol/dist(a,b))
@ti.func
def fz(a,b,ol):
    return k*(b[2]-a[2])*(1-ol/dist(a,b))+1
@ti.func
def delta_vx(idx1,idx2,ol):
    return fx(new_pos[idx1],new_pos[idx2],ol)
@ti.func
def delta_vy(idx1,idx2,ol):
    return fy(new_pos[idx1],new_pos[idx2],ol)
@ti.func
def delta_vz(idx1,idx2,ol):
    return fz(new_pos[idx1],new_pos[idx2],ol)
@ti.func
def total_next_vx(i,j):
    idx=i*N+j
    f=0.0
    if j!=0:
        f+=delta_vx(idx,idx-1,ol1)
    if j!=N-1:
        f+=delta_vx(idx,idx+1,ol1)
    if i!=0:
        f+=delta_vx(idx,idx-N,ol1)
    if i!=N-1:
        f+=delta_vx(idx,idx+N,ol1)
    if i!=0 and j!=0:
        f+=delta_vx(idx,idx-1-N,ol2)
    if i!=N-1 and j!=N-1:
        f+=delta_vx(idx,idx+1+N,ol2)
    if i!=0 and j!=N-1:
        f+=delta_vx(idx,idx+1-N,ol2)
    if i!=N-1 and j!=0:
        f+=delta_vx(idx,idx-1+N,ol2)
    f-=gamma*new_speed[idx][0]
    return f*dt+speed[idx][0]
@ti.func
def total_next_vy(i,j):
    idx=i*N+j
    f=0.0
    if j!=0:
        f+=delta_vy(idx,idx-1,ol1)
    if j!=N-1:
        f+=delta_vy(idx,idx+1,ol1)
    if i!=0:
        f+=delta_vy(idx,idx-N,ol1)
    if i!=N-1:
        f+=delta_vy(idx,idx+N,ol1)
    if i!=0 and j!=0:
        f+=delta_vy(idx,idx-1-N,ol2)
    if i!=N-1 and j!=N-1:
        f+=delta_vy(idx,idx+1+N,ol2)
    if i!=0 and j!=N-1:
        f+=delta_vy(idx,idx+1-N,ol2)
    if i!=N-1 and j!=0:
        f+=delta_vy(idx,idx-1+N,ol2)
    f-=gamma*new_speed[idx][1]
    return f*dt+speed[idx][1]
@ti.func
def total_next_vz(i,j):
    idx=i*N+j
    f=0.0
    if j!=0:
        f+=delta_vz(idx,idx-1,ol1)
    if j!=N-1:
        f+=delta_vz(idx,idx+1,ol1)
    if i!=0:
        f+=delta_vz(idx,idx-N,ol1)
    if i!=N-1:
        f+=delta_vz(idx,idx+N,ol1)
    if i!=0 and j!=0:
        f+=delta_vz(idx,idx-1-N,ol2)
    if i!=N-1 and j!=N-1:
        f+=delta_vz(idx,idx+1+N,ol2)
    if i!=0 and j!=N-1:
        f+=delta_vz(idx,idx+1-N,ol2)
    if i!=N-1 and j!=0:
        f+=delta_vz(idx,idx-1+N,ol2)
    f-=gamma*new_speed[idx][2]
    return f*dt+speed[idx][2]
@ti.kernel
def check()->ti.f32: # type: ignore
    max=0.0
    for i,j in ti.ndrange(N,N):
        if i==0:
            continue
        a=total_next_vx(i,j)
        b=new_speed[i*N+j][0]
        if abs(b-a)>max:
            max=abs(b-a)
        a=total_next_vy(i,j)
        b=new_speed[i*N+j][1]
        if abs(b-a)>max:
            max=abs(b-a)
        a=total_next_vz(i,j)
        b=new_speed[i*N+j][2]
        if abs(b-a)>max:
            max=abs(b-a)
        a=pos[i*N+j][0]+new_speed[i*N+j][0]*dt
        b=new_pos[i*N+j][0]
        if abs(b-a)>max:
            max=abs(b-a)
        a=pos[i*N+j][1]+new_speed[i*N+j][1]*dt
        b=new_pos[i*N+j][1]
        if abs(b-a)>max:
            max=abs(b-a)
        a=pos[i*N+j][2]+new_speed[i*N+j][2]*dt
        b=new_pos[i*N+j][2]
        if abs(b-a)>max:
            max=abs(b-a)
    return max
@ti.kernel
def iterate():
    for i,j in ti.ndrange(N,N):
        if i==0:
            continue
        last_speed=new_speed[i*N+j]
        next_new_speed[i*N+j][0]=alpha*total_next_vx(i,j)+(1-alpha)*new_speed[i*N+j][0]
        next_new_speed[i*N+j][1]=alpha*total_next_vy(i,j)+(1-alpha)*new_speed[i*N+j][1]
        next_new_speed[i*N+j][2]=alpha*total_next_vz(i,j)+(1-alpha)*new_speed[i*N+j][2]
        next_new_pos[i*N+j]=alpha*(dt*last_speed+pos[i*N+j])+(1-alpha)*new_pos[i*N+j]
@ti.kernel
def update_new():
    for i,j in ti.ndrange(N,N):
        new_pos[i*N+j]=next_new_pos[i*N+j]
        new_speed[i*N+j]=next_new_speed[i*N+j]    
@ti.kernel
def update():
    for i,j in ti.ndrange(N,N):
        pos[i*N+j]=new_pos[i*N+j]
        speed[i*N+j]=new_speed[i*N+j]
if __name__=="__main__":
    init()
    tri_init()
    window=ti.ui.Window("3D Mass-Spring System",(LENGTH,WIDTH))
    canvas=window.get_canvas()
    scene=window.get_scene()
    camera=ti.ui.Camera()
    camera.position(0,N/2,-100)
    camera.lookat(0,N/2,0)
    camera.up(0,1,0)
    while window.running:
        if(window.is_pressed(ti.ui.LMB)):
            push()
        camera.track_user_inputs(window, movement_speed=1, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        while(1):
            iterate()
            update_new()
            err=check()
            if err<EPS:
                break
        update()
        scene.point_light(pos=(-50, -50, 50), color=(1.0, 1.0, 1.0))
        scene.ambient_light((0.6, 0.6, 0.6))
        scene.mesh(vertices=pos,indices=tri_indices,color=(0.8, 0.8, 1.0),two_sided=True)
        canvas.scene(scene)
        window.show()