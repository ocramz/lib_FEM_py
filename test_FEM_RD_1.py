from lib_FEM.lib_FEM_RD import RD
Nex , Ney = [100, 100]                    
rmin = [0.0 , 0.0]
rmax = [Lx , Ly] = [20.0 , 20.0]
wwg, lwg = [1.0,  19.99]     
epsr1 = 3.4
epsr2 = 1.8 

M = FEM_PML(rmin, rmax, Nex, Ney)

M.init_x( )
M.x_background( 1/epsr2)
M.x_add_shape( 1/epsr1, M.rc, 'rect', [wwg, lwg])

M.plot_faceMesh( M.x)

##
xp = M.e2p( M.x , 1/epsr2) # face -> node bilinear interpolation 
xpc = np.clip(xp, 1/epsr1, 1) # clip values of interpolated 
# xpc = np.clip(xp, 1/epsr1, 1/epsr2)

# # PLOT
xr = M.plot_faceMesh( M.x)
# xpr = M.plot_nodeMesh( xp )
xpr = M.plot_nodeMesh( xpc )

print xr.shape
print xpcr.shape

iy = 10 # vertical slice
x2 = M.x_
x1 = M.xe_
y2 = xpr[:, iy]
y1 = xr[:, iy]
plt.plot(x1,y1, x2,y2)
plt.show()