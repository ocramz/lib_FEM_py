import matplotlib.pyplot as plt
import math
import numpy as np
import numpy.random as r
from numpy.linalg import norm
# from numpy import subtract 
# import pymc as mc
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import spsolve as spsolve
from scipy.sparse import *
import types
from itertools import *
from sympy import *
# from sympy.tensor import IndexedBase, Idxr
from sympy.utilities.lambdify import lambdify, implemented_function

class Mesh_rect2d(object):
    def __init__(s,rmin,rmax,Nex,Ney):
        if len(rmin)==len(rmax)==2 and rmax[0]>rmin[0] and rmax[1]>rmin[1] and Nex > 0 and Ney > 0:
            s.rmin = rmin
            s.rmax = rmax
            s.Lx = s.rmax[0] - s.rmin[0]
            s.Ly = s.rmax[1] - s.rmin[1]
            s.Nex = Nex
            s.Ney = Ney
            s.Ne = s.Nex*s.Ney
            s.Npx = s.Nex + 1
            s.Npy = s.Ney + 1
            s.Np = s.Npx * s.Npy
            s.lastie = s.Ne - 1
            s.lastip = s.Np - 1
            s.x0 = s.rmin[0]
            s.y0 = s.rmin[1]
            s.x1 = s.rmax[0]
            s.y1 = s.rmax[1]
            s.x_ = np.linspace(s.x0,s.x1,s.Npx)
            s.y_ = np.linspace(s.y0,s.y1,s.Npy)
            s.hx = s.x_[1]-s.x_[0]
            s.hy = s.y_[1]-s.y_[0]
            s.Npux = 1/s.hx
            s.xe_ = s.hx/2 + s.x_[:-1]
            s.ye_ = s.hy/2 + s.y_[:-1]
            s.ipx_ = np.arange( s.Npx )
            s.ipy_ = np.arange( s.Npy )
            s.ip_ = np.arange( s.Np )
            s.iex_ = np.arange( s.Nex )
            s.iey_ = np.arange( s.Ney )
            s.ie_ = np.arange( s.Ne )
            s.rc = [s.rmax[0]-s.Lx/2 , s.rmax[1]-s.Ly/2]
            s.q_ = s.nodes() # q_ is a GENERATOR 
            [s.X,s.Y] = s.meshgrid(s.x_, s.y_)
#             [s.Xe,s.Ye] = s.meshgrid(s.xe_, s.ye_)
#             s.e_ = s.elements()
#             s.q2 = s.nodes_rect(3,27) ## a rectangular region

    def interpolate2d_e2p(s, A, ix, iy):
        """bilinear intepolation e -> p"""
        Ae = s.hx * s.hy
        x , y = [ s.x_[ix], s.y_[iy] ]
        iex0 = ix # ..using these index arrays
        iex1 = iex0 + 1
        iey0 = iy
        iey1 = iey0 + 1 
#         print iex0,iey0,iex1,iey1
        Ia = A[ iey0, iex0 ]
        Ib = A[ iey1, iex0 ]
        Ic = A[ iey0, iex1 ]
        Id = A[ iey1, iex1 ]  
        x0 , y0 = [ s.xe_[iex0], s.ye_[iey0] ] 
        x1 , y1 = [ s.xe_[iex1], s.ye_[iey1] ]  
        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)
        return 1/Ae * (wa*Ia + wb*Ib + wc*Ic + wd*Id )
        
    def e2p(s, xe, b):
        if s.check_onFaceMesh(xe):
            ix = s.ipx_[1:-2] # inner nodes
            iy = s.ipy_[1:-2]
            xer = xe.reshape(s.Nex, s.Ney)
            ip = [i for i in s.ip_ if s.check_iprect(i, [ix[0],ix[-1]], [iy[0],iy[-1]])]
            xp = b * np.ones( s.Np )  ## BACKGROUND CONSTANT = b
            xp[ip] = s.interpolate2d_e2p( xer, ix, iy)
            return xp

    def meshgrid(s, x, y):
        try:
            len(x) and len(y)
            Lx = len(x) 
            Ly = len(y)
            x = np.reshape(x, (Lx,1))
            y = np.reshape(y, (1,Ly))
            return np.tile(x, [1, Ly]), np.tile(y, [Lx, 1])
        except:
            print "!!! invalid x, y"
                        
    def nodes(s):
        """face -> nodes generator"""
        for ie in range(s.Ne):
            iex, iey = s.ie2iexy(ie)
            yield s.iexy2nodes(iex, iey)
            
    def nodes_rect_xy(s, ie, ie1_, ie2_):
            ie1x, ie1y = ie1_
            ie2x, ie2y = ie2_
            if ie1x <= iex <= ie2x and ie1y <= iey <= ie2y:
                    yield s.iexy2nodes(iex, iey)
                                       
    def nodes_rect(s, ie1, ie2):
        """face -> nodes generator, in a rectangle with corner element indices ie1, ie2"""
        if s.ie_validq(ie1) and s.ie_validq(ie2):
            
            for ie in range(ie1, ie2):
                iex, iey = s.ie2iexy(ie)
                if ie1x <= iex <= ie2x and ie1y <= iey <= ie2y:
                    yield s.iexy2nodes(iex, iey)
    
    def iexy2nodes(s, iex, iey):
        """face nodes, len==4 """
        tl = (s.Nex + 1) * iey + iex 
        bl = tl + 1
        br = bl + s.Nex + 1
        tr = br - 1
        return [tl, bl, br, tr]
           
    def ip_validq(s, ip):
        return 0 <= ip <= s.Np

    def ie_validq(s, ie):
        return 0 <= ie <= s.Ne

    def contains(s, r):
        if isinstance(r, list) and len(r)==2:
            return ( r[0]-s.rmin[0]>=0 and r[1]-s.rmin[1]>=0 and r[0]-s.rmax[0]<=0 and r[1]-s.rmax[1]<=0 )
        else:
            print "!!! r must be a length 2 array"
            return False
            
    def ip2r(s, ip):
        """Cartesian coords from global node numbering"""
        if isinstance(ip, int) and s.ip_validq(ip):
            npx, npy = s.ip2ipxy(ip)
            return [ s.x_[npx] , s.y_[npy]]

    def ie2r(s, ie):
        """Cartesian coords from global elem numbering"""
        if isinstance(ie, int) and s.ie_validq(ie):
            nex, ney = s.ie2iexy(ie)
            return [ s.xe_[nex] , s.ye_[ney]]
    
    def ip2ipxy(s, ip):
        """x, y, node coords from global node numbering"""
        npx = np.mod(ip, s.Npx)
        return npx, (ip-npx)/s.Npx
        
    def ipxy2ip(s, ipx, ipy):
        return ipy*s.Npx + ipx

    def ie2iexy(s, ie):
        """x, y, face coords from global face numbering"""
        nex = np.mod(ie, s.Nex)
        return nex, (ie-nex)/s.Nex
    
    def r2i(s, r, t='e'):
        if (t == 'e' or t =='p'):
            N1 = s.Nex
            if t=='p':
                N1 = s.Npx
            xd = r[0] - s.x_[0]
            yd = r[1] - s.y_[0]
            return int( (math.floor(yd / s.hy) )* N1 + math.floor(xd / s.hx))
        else:
            print "!!! type must be either 'e' or 'p'"
            return None

# # #             
    def r_2i_rect(s, r_, r1, r2, t):
        if len(r_)>0 and len(r_[0])==2:
            if t == 'e':
                return [s.r2i_rect_iter(r, r1, r2, s.Nex) for r in r_]
            elif t == 'p':
                return [s.r2i_rect_iter(r, r1, r2, s.Npx) for r in r_]
            
    def r2i_rect_iter(s, r, r1, r2, N1):
        if r1[0]<r[0]<r2[0] and r1[1]<r[1]<r2[1]:
            xd = r[0] - s.x_[0]
            yd = r[1] - s.y_[0]
            yield int( (math.floor(yd / s.hy) )* s.Nex + math.floor(xd / s.hx))
# # #    
    def r2ie(s, r):
        """global face number from Cartesian coords"""
        xd = r[0] - s.x_[0]
        yd = r[1] - s.y_[0]
        return int( (math.floor(yd / s.hy) )* s.Nex + math.floor(xd / s.hx))
    
    def r2ip(s, r):
        """global node number from Cartesian coords"""
        xd = r[0] - s.x_[0]
        yd = r[1] - s.y_[0]
        return int( (math.floor(yd / s.hy) )* s.Npx + math.floor(xd / s.hx))
            
    def check_iprect(s, ip, ipbx, ipby):
        ipx, ipy =  s.ip2ipxy(ip)
        return ( ipbx[0]<=ipx<=ipbx[1] and ipby[0]<=ipy<=ipby[1] )
            
    def check_onFaceMesh(s, x):
        try:
            len(x)
            return len(x)==s.Ne
        except:
            return False
            
    def check_onNodeMesh(s, x):
        try:
            len(x)
            return len(x)==s.Np
        except:
            return False
        
    def check_x(s):
        """tests whether s.x \in [0,1]^s.Ne """
        if len(s.x)==s.Ne and min(s.x) >= 0 and max(s.x) <= 1:
            return True
        else:
            return False
    
    def check_x_loc(s, x_):
        """as check_x()
        NB: LOCAL version, supply design var. x_ \in [0,1]^s.Ne """
        if len(x_)==s.Ne and min(x_) >= 0 and max(x_) <= 1:
            return True
        else:
            return False
            
    def check_xref(s):
        """tests whether s.xref is an appropriate coefficient array"""
        if len(s.xref)==s.Ne and min(s.xref) >= 0 and max(s.xref) <= 1 :
            return True
        else:
            return False
            
    def check_x_bounds(s, emin, emax, vmin, vmax):
        if emax > emin > 0 and 1.0>=vmax>vmin>0:
            return True
        else:
            return False
            
    def set_x_bounds(s, emin, emax, vmin, vmax):
        if s.check_x_bounds(emin, emax, vmin, vmax):
            s.emin = emin
            s.emax = emax
            s.vmin = vmin
            s.vmax = vmax
        else:
            print "!!! ensure that 0 < emin < emax and 0 < vmin < vmax <= 1"
            
    def check_x_exists(s):
        if hasattr(s.x):
            return True
        else:
            return False
            
    def check_xref_exists(s):
        if hasattr(s.xref):
            return True
        else:
            return False
            
    def x_circle_radius(s, e, rc):
        return np.linalg.norm( np.subtract( s.ie2r(e) , rc) )
        
    def rect(s, ix_, iy_, N1, imin, imax, jmin, jmax):
        return [i+N1*j for (i,j) in cartes(ix_,iy_) if imin<=i<=imax and jmin<=j<=jmax]
        
    def nodes_rectr_2(s, r1, r2):
        ip1, ip2 = [s.r2ip(r1), s.r2ie(r2)]
        imin, jmin = s.ie2iexy(ip1)
        imax, jmax = s.ie2iexy(ip2)
        return s.rect( s.ix_, s.iy_, s.Npx, imin, imax, jmin, jmax)
        
    def elements_rectr_2(s, r1, r2):
        ie1, ie2 = [s.r2ie(r1), s.r2ie(r2)]
        imin, jmin = s.ie2iexy(ie1)
        imax, jmax = s.ie2iexy(ie2)
        return s.rect( s.iex_, s.iey_, s.Nex, imin, imax, jmin, jmax)
        
    def elements_rect(s, ie1, ie2):
        if s.ie_validq(ie1) and s.ie_validq(ie2):
            ie1x, ie1y = s.ie2iexy(ie1)
            ie2x, ie2y = s.ie2iexy(ie2)
            for ie in range(ie1, ie2):
                iex, iey = s.ie2iexy(ie)
                if ie1x <= iex <= ie2x and ie1y <= iey <= ie2y:
                    yield ie
                    
    def elements_rectr(s, r1, r2):
        if s.contains(r1) and s.contains(r2):
            ie1, ie2 = s.r2ie(r1), s.r2ie(r2)
            return [ie for ie in s.elements_rect(ie1, ie2)]
            
    def x_add_shape(s, c, rc, name, *args):
        """"""
        if s.check_x_exists and s.check_x and s.contains(rc) and c>0:
            x_ = s.x
#             print args
            if name == 'rect' and s.contains(args[0]):
                d_ = args[0]
                r1 = [rc[0]-d_[0]/2, rc[1]-d_[1]/2]
                r2 = [rc[0]+d_[0]/2, rc[1]+d_[1]/2]
                x_[ s.elements_rectr_2(r1, r2) ] = c
#             else:
#                 s.init_x
#                 s.x_add_shape(c, [0, 0], name, *args)
            if name == 'circle' and isinstance(args[0], float) and args[0]>0:
                radius = args[0]
                ie_ = [ ie for ie in range(s.Ne) if s.x_circle_radius(ie, rc) <= radius ]
                x_[ ie_ ] = c
                
            if name == 'ring' and isinstance(args[0], float) and args[0]>0 and args[1]>0:
                radius = args[0]
                w = args[1] # ring width
                r1 = radius-w/2
                r2 = radius+w/2
#                 print radius, w, r1, r2
                ie_ = [ ie for ie in range(s.Ne) if r1 <= s.x_circle_radius(ie, rc) <= r2 ]
                x_[ ie_ ] = c
            s.x = x_ # assign to s.x

    def xref_add_shape(s, c, rc, name, *args):
        """"""
        if s.check_xref_exists and s.check_xref and s.contains(rc) and c>0:
            if name == 'rect' and s.contains(args[0]):
                d_ = args[0]
                r1 = [rc[0]-d_[0]/2, rc[1]-d_[1]/2]
                r2 = [rc[0]+d_[0]/2, rc[1]+d_[1]/2]
                s.xref[ s.elements_rectr_2(r1, r2) ] = c

    def r_array(s, rc, N, a, rd):
        if s.contains(rc) and np.linalg.norm(rd)>0 and isinstance(N, int) and N>0 and isinstance(a, float) and a>0:
            rdn = rd / np.linalg.norm(rd)
            r_ =  [ [ rc[0]+ii*a*rdn[0], rc[1]+ii*a*rdn[1] ] for ii in range(N)] # HACK # would be better with an iterator here
            return r_ 
            
    def x_add_shape_array(s, c, rc, name, N, a, rd, d):
        """c > 0 inverse permittivity, rc = [rcx, rcy] array center
           a > 0 array pitch, rd = [rdx, rdy] direction array, d "name"-specific"""
        r_ = s.r_array(rc, N, a, rd)
        if s.contains(r_[-1]):
            for ri in r_:
                s.x_add_shape(c, ri, name, d)
                
    def plot_faceMesh(s, x):
        if s.check_onFaceMesh(x):
            xr = x.reshape(s.Ney,s.Nex).transpose()
            plt.imshow(xr)
            plt.colorbar()
            plt.show()
            interactive
#             return xr
        else:
            return None
            
    def plot_nodeMesh(s, x):
        if s.check_onNodeMesh(x):
            xr = x.reshape(s.Npy,s.Npx).transpose()
            plt.imshow(xr)
            plt.colorbar()
            plt.show()
            interactive
#             return xr
        else:
            return None
            
    def wrap(s, f, *args, **kwargs): # decorator for timeit()
       def w():
           return f(*args, **kwargs)
       return w
    # # w1 = wrap(M.eval_f, M.x, k0, Qpesx, Qpesy, n_ )
    # # w2 = wrap(M.powflux_b_norm, lam0, ipsens_ )
    # # t1 = timeit.timeit(w1, number=20 )
                
def ie_rectangle(rc, M, d_):
    if isinstance(M, FEM_PML):
        if len(rc)==len(d_)==2 and M.contains(rc) and 0<d_[0]<M.Lx and 0<d_[1]<M.Ly:
            r1 = [rc[0]-d_[0]/2, rc[1]-d_[1]/2]
            r2 = [rc[0]+d_[0]/2, rc[1]+d_[1]/2]
            return M.elements_rectr(r1, r2)
        else:
            return None
    else:
        print "!!! M must be a FEM_PML() "
        return None        

# class FEMesh_designvar:
#     def __init__(s, M, x):
#         if isinstance(M, FEM_PML):  
#             if len(x)==M.Ne: 
#                 s.x = x
#             else:
#                 raise 
#         else:
#             print '!! M not instance of FEM_PML'
#             return None
        
#
class FEM_PML(Mesh_rect2d):
    def __init__(self, rmin, rmax, Nex, Ney):
        Mesh_rect2d.__init__(self, rmin, rmax, Nex, Ney)
        self.elemmtx()
        self.nodes_full()
        self.sysmtx_idx()
        self.init_x()
        self.init_xref()
    def nodes_full(s):
        s.q = np.zeros([s.Ne, s.Nwe], dtype=int)
        for ii, pe in enumerate(s.q_):
            s.q[ii,] = pe
    def simp(s, p):
        """SIMP penalization of design variable s.x
        simp:: [0, 1] -> [emin, emax]
        penalize substitute s.x and s.xref IN PLACE"""
        if not( s.check_x() ):
            p = 1
        x = s.x
        xref = s.xref        
        s.xp = s.emin + (s.emax-s.emin)*x ** p # penalized 
        s.xrefp = s.emin + (s.emax-s.emin)*xref ** p
    def simp_loc(s, p, x_, xref_):
        """as simp()
        NB: LOCAL version: supply design var x_"""
        try:
            s.check_x_loc( x_ ) and s.check_x_loc( xref_ )
            xp = s.emin + (s.emax-s.emin)*x_ ** p # penalized 
            xrefp = s.emin + (s.emax-s.emin)*xref_ ** p
            return xp, xrefp
        except:
            print "!!! simp_loc() : invalid design variable!"
        
    def fill_x(self, x):
        if self.check_x:
            """one coefficient per element"""
            self.x = x
        else:
            s.init_x()
    def fill_xref(self, xref):
        if self.check_x:
            """one coefficient per element"""
            self.xref = xref
        else:
            s.init_xref()
    def init_x(self):
        self.x = np.ones(self.Ne, dtype=float)
    def init_xref(self):
        self.xref = np.ones(self.Ne, dtype=float)
    def x_background(self, c):
        if c>0 and c<=1:
            self.x = c * np.ones(self.Ne, dtype=float)
    def x_add_device_nanobeam95(s, xcore, xclad, wwg, lwg, rcwg, a, Nper, r, gap):
        """Fan95-style PhC nanobeam resonator:
        xcore, xclad: inverse relative permittiv.
        wwg, lwg : width and length of waveguide
        rcwg : center point of device [1 x 2]
        a, Nper, r : lattice pitch, # periods, hole radius
        gap : lattice defect size
        returnX : Boolean (to return design variable x instead of storing it as s.x, to used yet)"""
        s.x_add_shape( xcore, rcwg, 'rect', [wwg, lwg])     
        rc1 = [rcwg[0], rcwg[1]+(gap+a)/2]
        rc2 = [rcwg[0], rcwg[1]-(gap+a)/2]
        s.x_add_shape_array( xclad, rc1, 'circle', Nper, a, [0.0,1.0], r) 
        s.x_add_shape_array( xclad, rc2, 'circle', Nper, a, [0.0,-1.0], r)

    def pml(s, p, dtpml):
        if p>1 and 0.0 <= dtpml < 1.0: 
            rcx, rcy = s.rc
            tpml = min(s.Lx, s.Ly)*dtpml
            Lboxx, Lboxy = [s.Lx/2 - tpml, s.Ly/2 - tpml]
            s.sx = np.ones(s.Ne, dtype=complex)
            s.sy = np.ones(s.Ne, dtype=complex)
            s.Lambdaxx = np.ones(s.Ne, dtype=complex)
            s.Lambdayy = np.ones(s.Ne, dtype=complex)
            s.Lambdazz = np.ones(s.Ne, dtype=complex)
            for ii,ie in enumerate( range(s.Ne) ):
                rx, ry = s.ie2r(ie) 
                dx = abs(rx - rcx) - Lboxx # if dx > Lx/2 - tpml, we are in the PML
                dy = abs(ry - rcy) - Lboxy
                if dx >= 0:
                    s.sx[ii] = s.sx[ii] - 1j*(dx / tpml)**p
                if dy >= 0:
                    s.sy[ii] = s.sy[ii] - 1j*(dy / tpml)**p
                s.Lambdaxx[ii] = s.sx[ii]/s.sy[ii]   # required by Knablax (K1)
                s.Lambdayy[ii] = s.sy[ii]/s.sx[ii]   # " Knablay (K1)
                s.Lambdazz[ii] = s.sx[ii] * s.sy[ii] # " K2
#             
    def elemmtx(s):
        """element matrices, column stacked"""
        B = Lagrange_bilinear_rect2d()
        B.elementmtx(s.hx, s.hy)
        s.Nwe = B.Nwe # no. bases/element = no.nodes/element
        s.Nnpe = prod(B.Knablax0.shape)
        s.Kqx0_ = B.Kqx0.reshape(s.Nnpe,1)
        s.Kqy0_ = B.Kqy0.reshape(s.Nnpe,1)
        s.Knablax0_ = B.Knablax0.reshape(s.Nnpe,1)
        s.Knablay0_ = B.Knablay0.reshape(s.Nnpe,1)
        s.Kn0_ = B.Kn0.reshape(s.Nnpe,1)
#
    def sysmtx_idx(s):
        """COOrds in system mtx"""
        q_ = s.q.reshape(s.Ne * s.Nwe)
        s.I = np.kron( q_, np.ones(s.Nwe) )
        q_rep = np.tile(s.q, [1, s.Nwe])
        s.J = q_rep.reshape(s.Ne * s.Nnpe)
    
    def sysmtx(s):
        """ASSEMBLY of sys.mtx, PML (anisotropic)"""
        s.simp(s.psimp)
        K_effx1 = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        K_effy1 = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        K_eff2  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        Kqpex_eff  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        Kqpey_eff  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
#         K_effx10 = np.zeros(s.Nnpe * s.Ne, dtype=complex)
#         K_effy10 = np.zeros(s.Nnpe * s.Ne, dtype=complex)
#         K_eff20  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
#         Kqpex_eff0  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
#         Kqpey_eff0  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        for ii in range( s.Ne ):
            idx = np.ix_( np.arange( ii*s.Nnpe, (ii+1)*s.Nnpe))
            K_effx1.put(idx, s.xp[ii] * (1/s.Lambdaxx[ii]) * s.Knablax0_)
            K_effy1.put(idx, s.xp[ii] * (1/s.Lambdayy[ii]) * s.Knablay0_)
            K_eff2.put(idx, s.Lambdazz[ii] * s.Kn0_)
            Kqpex_eff.put(idx, s.xp[ii] * (1/s.Lambdaxx[ii]) * s.Kqx0_)
            Kqpey_eff.put(idx, s.xp[ii] * (1/s.Lambdayy[ii]) *s.Kqy0_)
#             K_effx10.put(idx, s.xref[ii] * (1/s.Lambdaxx[ii]) * s.Knablax0_)
#             K_effy10.put(idx, s.xref[ii] * (1/s.Lambdayy[ii]) * s.Knablay0_)
#             K_eff20.put(idx, s.Lambdazz[ii] * s.Kn0_)
#             Kqpex_eff0.put(idx, s.xref[ii] * (1/s.Lambdaxx[ii]) * s.Kqx0_)
#             Kqpey_eff0.put(idx, s.xref[ii] * (1/s.Lambdayy[ii]) *s.Kqy0_)
        try:
            s.K1 = coo_matrix((K_effx1 + K_effy1, (s.I, s.J)) ).tocsr()
            s.K2 = coo_matrix((K_eff2, (s.I, s.J)) ).tocsr()
            s.Qpex = coo_matrix((Kqpex_eff, (s.I, s.J))).tocsr()
            s.Qpey = coo_matrix((Kqpey_eff, (s.I, s.J))).tocsr()
#             # reference system matrices
#             s.K10 = coo_matrix((K_effx10 + K_effy10, (s.I, s.J)) ).tocsr()
#             s.K20 = coo_matrix((K_eff20, (s.I, s.J)) ).tocsr()
#             s.Qpex0 = coo_matrix((Kqpex_eff0, (s.I, s.J))).tocsr()
#             s.Qpey0 = coo_matrix((Kqpey_eff0, (s.I, s.J))).tocsr()
        except:
            print "!!! s.K1, s.K2, S.Qpex, s.Qpey not initialized"
            
    def sysmtx_loc(s, x_):
        """ASSEMBLY of sys.mtx, PML (anisotropic)
        NB: LOCAL version: need to supply design variable x_"""
        K_effx1 = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        K_effy1 = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        K_eff2  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        Kqpex_eff  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        Kqpey_eff  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        for ii in range( s.Ne ):
            idx = np.ix_( np.arange( ii*s.Nnpe, (ii+1)*s.Nnpe))
            K_effx1.put(idx, x_[ii] * (1/s.Lambdaxx[ii]) * s.Knablax0_)
            K_effy1.put(idx, x_[ii] * (1/s.Lambdayy[ii]) * s.Knablay0_)
            K_eff2.put(idx, s.Lambdazz[ii] * s.Kn0_)
            Kqpex_eff.put(idx, x_[ii] * (1/s.Lambdaxx[ii]) * s.Kqx0_)
            Kqpey_eff.put(idx, x_[ii] * (1/s.Lambdayy[ii]) *s.Kqy0_)
        try:
            K1 = coo_matrix((K_effx1 + K_effy1, (s.I, s.J)) ).tocsr()
            K2 = coo_matrix((K_eff2, (s.I, s.J)) ).tocsr()
            Qpex = coo_matrix((Kqpex_eff, (s.I, s.J))).tocsr()
            Qpey = coo_matrix((Kqpey_eff, (s.I, s.J))).tocsr()
            return K1, K2, Qpex, Qpey
        except:
            print "!!! K1, K2, Qpex, Qpey not initialized"
            
    def dK1dx(s):
        """d K / d x =  d K1 / dx
        
        (Hp: design var is homogeneous across each element )"""
        return ( s.Knablax0_ + s.Knablay0_ ).reshape([s.Nwe, s.Nwe])

    def update(s, x, freq):
        """fills design var.x, builds sys.mtx, solves K u = b
        NB. uses 'global' structure and state array s.x, s.u"""
        s.fill_x(x)
        s.fem_solve(freq) # assembly happens in here
        
    def update_loc(s, x, freq):
        u, K, K1, K2, Qpex, Qpey = s.fem_solve_loc(freq, x)
        
    def ie_designvar(s, r1, r2):
        """face indices of elements included in rectangle with corner coord arrays r1, r2"""
        ie_ = s.elements_rectr_2(r1, r2)
        return ie_

    def eval_f(s, x, freq, ips_ ):
        """cost function: Poynting flux
        x : design variable
        freq : freq.k0 wavenumber of solution, freq.lam0: wavelength
        ips_ : nodes defining Poynting flux plane"""
        s.update(x, freq) # updates variable and state
        return s.powflux_norm( freq , ips_)
        
    def eval_dfdx_cfd(s, x, freq, ips_, ie, eps):
        """response sentitivity by Central Finite Differences"""
        x1 = x
        x2 = x
        x1[ie] = x1[ie]-eps # perturbation
        x2[ie] = x2[ie]+eps
        p1 = s.eval_f(x1, freq, ips_)
        p2 = s.eval_f(x2, freq, ips_)
        dfdx_ie = (p2-p1)/(2*eps)
        return dfdx_ie
        
    def eval_dfdx(s, u, tau, ie):
        """cost function deriv wrt design variable array x
        NB: x (and therefore state s.u) must be already consistent, see s.update() """
        iqe = s.q[ie]
        dKdx = s.dK1dx() # deriv of sys.mtx wrt ANY design variable xi \in x
        return ( - ( tau[iqe] ).dot( dKdx.dot( u[iqe] )))
        
    def eval_dfdx_(s, x, u, tau, freq, ie_, ips_):
        """populate sensitivity array"""
        dfdx = np.zeros( s.Ne , dtype=complex)
        
        for e in ie_: # forall elements in design var.set
            dfdx[e] = s.eval_dfdx(u, tau, e)
        return dfdx
    
    def coforc_loc(s, freq, u, Qpex, Qpey):
        """co-forcing term : d/dx(x'Qx) = (Q + Q')x """
        Ax = Qpex + (Qpex.conjugate()).transpose() # d/dx(x'Qx) = (Q + Q')x
        Ay = Qpey + (Qpey.conjugate()).transpose()
        dfxdu = 1j * freq.lam0 * Ax.dot( u )
        dfydu = 1j * freq.lam0 * Ay.dot( u )
        return dfxdu, dfydu
        
    def coforc_sens_loc(s, freq, u, Qpex, Qpey, ips_):
        """co-forcing term, restricted to ips_ node set : d/dx(x'Qx) = (Q + Q')x """
        ipsgrid = np.ix_(ips_, ips_) 
        Qpexi = csc_matrix( np.zeros(Qpex) )
        Qpeyi = Qpey[ipsgrid]
        Ax = Qpex + (Qpex.conjugate()).transpose() # d/dx(x'Qx) = (Q + Q')x
        Ay = Qpey + (Qpey.conjugate()).transpose()
        dfxdu = 1j * freq.lam0 * Ax.dot( u )
        dfydu = 1j * freq.lam0 * Ay.dot( u )
        return dfxdu, dfydu
    
    
# #     function [u tau] = fe_adj(K1,K2,f,lam0,on_,A)
# #     % direct and adj soln.
# #     % if g = u'Qu then dg/du=(Q+Q')u
# #     [u K] = fe(K1,K2,f,lam0,on_);
# #     dg_u = 1i*lam0 * A * u; % deriv.of cost wrt state u AT SENSOR 
# #     tau = herm(K)\dg_u; % adjoint state tau
        
    def powflux_norm(s, freq, ips_):
        """ips_ : flux plane node index array"""
        Sxs, Sys = s.poynt_sens(freq, ips_) # restrict to sensor surface
        Sv = np.array( [sum(Sxs), sum(Sys)] , dtype=complex)
        return np.linalg.norm(Sv)

    def powflux_norm_loc(s, u, freq, ips_, Qpex, Qpey):
        """ips_ : flux plane node index array"""
        Sxs, Sys = s.poynt_sens_loc(u, freq, ips_, Qpex, Qpey) # restrict to sensor surface
        Sv = np.array( [sum(Sxs), sum(Sys)] , dtype=complex)
        return np.linalg.norm(Sv)
    
    def poynt_sens(s, freq, ips_):
        """Poynting vector S = [Sxs, Sys] restricted to surface defined by node array ips_ """
        ipsgrid = np.ix_(ips_, ips_) 
        Sxs = 1j * freq.lam0 * np.multiply(s.Qpex[ipsgrid].dot(s.u[ips_]) , np.conj(s.u[ips_]) )
        Sys = 1j * freq.lam0 * np.multiply(s.Qpey[ipsgrid].dot(s.u[ips_]) , np.conj(s.u[ips_]) )
        return Sxs, Sys
    
    def poynt_sens_loc(s, u, freq, ips_, Qpex, Qpey):
        """Poynting vector S = [Sxs, Sys] restricted to surface defined by node array ips_ 
        NB: local version: requires up-to-date state u (wrt updated design variable x)"""
        ipsgrid = np.ix_(ips_, ips_) 
        ui = u[ips_]
        Qpexi = Qpex[ipsgrid]
        Qpeyi = Qpey[ipsgrid]
        Sxs = 1j * freq.lam0 * np.multiply( Qpexi.dot( ui ) , np.conj( ui ) )
        Sys = 1j * freq.lam0 * np.multiply( Qpeyi.dot( ui ) , np.conj( ui ) )
        return Sxs, Sys #, ui, Qpexi, Qpeyi
        
    def poynt_loc(s, u, freq, Qpex, Qpey):
        """Poynting vector S = [Sxs, Sys]
        NB: local version: requires up-to-date state u (wrt updated design variable x)"""
        Sxs = 1j * freq.lam0 * np.multiply( Qpex.dot(u), u.conjugate() )
        Sys = 1j * freq.lam0 * np.multiply( Qpey.dot(u), u.conjugate() )
        return Sxs, Sys 
    
#     def frequ(s, k):
#         freq.k0 = k
#         freq.lam0 = 2*np.pi/k
#         return freq
    class Frequ(object):
        def __init__(s, k):
            s.k0 = k
            s.lam0 = 2*np.pi/k
    
    def poynt(s, freq):
        """returns Sx Sy = x, y components of the Poynting vector S"""
        Sx = 1j * freq.lam0 * np.multiply(s.Qpex.dot(s.u) , np.conj(s.u) )
        Sy = 1j * freq.lam0 * np.multiply(s.Qpey.dot(s.u) , np.conj(s.u) )
        return Sx, Sy
     
    def curl(s, uz):
        """curl(uz) ~ [ d_x(uz), -d_y(uz)]"""
        try:
            return s.Qpey.dot( uz ) , - s.Qpex.dot( uz ) 
        except:
            print "uz: nodal mesh"
        
    def sensormtx(s, ies_):
        """returns Qpesx, Qpesy = x, y flux sensor system matrices"""
        Kqpesx_eff  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        Kqpesy_eff  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        Kqpesx_eff0  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        Kqpesy_eff0  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        for ii in ies_:
            idx = np.ix_( np.arange( ii*s.Nnpe, (ii+1)*s.Nnpe))
            Kqpesx_eff.put(idx, s.x[ii] * (1/s.Lambdaxx[ii]) * s.Kqx0_)
            Kqpesy_eff.put(idx, s.x[ii] * (1/s.Lambdayy[ii]) *s.Kqy0_)
            Kqpesx_eff0.put(idx, s.xref[ii] * (1/s.Lambdaxx[ii]) * s.Kqx0_)
            Kqpesy_eff0.put(idx, s.xref[ii] * (1/s.Lambdayy[ii]) *s.Kqy0_)
        Qpesx = coo_matrix((Kqpesx_eff, (s.I, s.J))).tocsr()
        Qpesy = coo_matrix((Kqpesy_eff, (s.I, s.J))).tocsr()
        Qpesx0 = coo_matrix((Kqpesx_eff0, (s.I, s.J))).tocsr()
        Qpesy0 = coo_matrix((Kqpesy_eff0, (s.I, s.J))).tocsr()
        return Qpesx, Qpesy, Qpesx0, Qpesy0
        
    def sensormtx_loc(s, ies_, x_):
        """returns Qpesx, Qpesy = x, y flux sensor system matrices
        LOCAL version: need to supply design variable x_"""
        Kqpesx_eff  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        Kqpesy_eff  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        for ii in ies_:
            idx = np.ix_( np.arange( ii*s.Nnpe, (ii+1)*s.Nnpe))
            Kqpesx_eff.put(idx, x_[ii] * (1/s.Lambdaxx[ii]) * s.Kqx0_)
            Kqpesy_eff.put(idx, x_[ii] * (1/s.Lambdayy[ii]) *s.Kqy0_)
        Qpesx = coo_matrix((Kqpesx_eff, (s.I, s.J))).tocsr()
        Qpesy = coo_matrix((Kqpesy_eff, (s.I, s.J))).tocsr()
        return Qpesx, Qpesy
            
        
    def source_Hz_vertdipole(s, rsource, Npsource):
        """vertical dipole array that excites Hz component"""
#         s.contains(rsource) and s.isposint(Npsource)
        s.b = np.zeros( s.Np , dtype=complex)
        # # list of nodes of current element
        iesc = s.r2ie(rsource) # face containing source coords
        ipsc1 = s.q[iesc, 0] # top left and top right nodes used as ref.
        ipsc2 = s.q[iesc, 3]
        ips1 =   np.arange(ipsc1 - int(floor(Npsource/2)), ipsc1 + int(floor(Npsource/2)))
        ips2 =   np.arange(ipsc2 - int(floor(Npsource/2)), ipsc2 + int(floor(Npsource/2))) 
        s.b[ips1] = 1.0 # list of node pairs for B flux lines from " 
        s.b[ips2] = -1.0 
    
    def sysmtx_helmholtz(s, freq):
        s.sysmtx()
        s.K = s.K1 - freq.k0**2 * s.K2 # Helmholtz eqn.

    def sysmtx_helmholtz_loc(s, freq, x_):
        K1, K2, Qpex, Qpey = s.sysmtx_loc(x_)
        K = K1 - freq.k0**2 * K2 # Helmholtz eqn.
        return K, K1, K2, Qpex, Qpey
    
    def fem_solve(s, freq):
        s.sysmtx_helmholtz(freq)
        s.u = spsolve(s.K, s.b)
#     def fem_solve_reference(s, k0):
#         s.K0 = s.K10 - k0**2 * s.K20
#         s.u0 = spsolve(s.K0, s.b)  

    def fem_solve_loc(s, freq, x_, b):
        K, K1, K2, Qpex, Qpey = s.sysmtx_helmholtz_loc(freq, x_)
        u = spsolve(K, b)
        return u, K, K1, K2, Qpex, Qpey
    
    def fem_solve_cosolve_loc(s, freq, x_, b):
        """
        freq: frequ() object
        x_ : design variable
        b : forcing term vector
        
        NB : heavy lifting done in here:
        1 + 2 LU solves, assuming K is NOT Hermitian"""
        K, K1, K2, Qpex, Qpey = s.sysmtx_helmholtz_loc(freq, x_)
        Kt = K.transpose()
# # #         u, lu = s.solve_lu(K, b) # state u, lu fact. of primal problem
        u = s.solve_(K, b)
#         print "SOLVE done"
        cx, cy = s.coforc_loc( freq, u, Qpex, Qpey ) # co-forcing (= d cost function / d state)
# # #         taux, lutx = s.solve_lu(Kt, cx)
# # #         tauy, luty = s.solve_lu(Kt, cy)
        taux = s.solve_(Kt, cx)
        tauy = s.solve_(Kt, cy)
#         print "CO-SOLVES done"
        return u, taux, tauy, K, Kt, K1, K2, Qpex, Qpey #, lu, lutx, luty        

    def lu_fact(s, A):
        """returns SuperLU object"""
        return spla.splu(A)
        
    def solve_(s, A, b):
        """is spsolve faster than solving by LU factorization?"""
        return spsolve(A, b)
        
    def solve_lu(s, A, b):
        """solves A x = b using LU factorization
        returns x (numpy array) and lu (SuperLU object)"""
        lu = s.lu_fact(A, b)
        x = lu.solve(b)
        return x, lu
#         
# # # #     def solve_cosolve(s, A, At, b, c):
# # # #         """returns x, tau, lu, lut
# # # #         x := A x = b, tau := A^T tau = c using 2 LU factorizations
# # # #         lu, lut :: SuperLU 
# # # #         PROBLEM: LU fact. is not shared among solves, so this approach is not so cool yet"""
# # # #         lu = lu_fact(s, A)
# # # #         lut = lu_fact(s, At) # if we can prove that A is Hermitian, A == A^T so lut == lu
# # # #         x = lu.solve(b)
# # # #         tau = lut.solve(c)
# # # #         return x, tau, lu, lut
            
    def ip_sensor(s, ip0, Nip, n_):
        """node indices of field sensor plane
        ip0: center index
        Nip = # sensor nodes
        n_ = direction unit vector (grid-aligned i.e. either [0,1] (flux in y direction) or [1,0])"""
        if n_ == [0,1]:
            ip_ = np.array( np.arange(ip0 - int(floor(Nip/2)), ip0 + int(floor(Nip/2))) , dtype=int) 
        if n_ == [1,0]:
            ip_ = np.array( np.arange(ip0 - floor(Nip/2)*s.Npx, ip0 + floor(Nip/2)*s.Npx, s.Npx) , dtype=int) 
        return ip_
        
    def ie_sensor(s, ie0, Nie, n_):
        """face indices of field sensor plane
        ie0: center index
        Nie = # sensor faces
        n_ = direction unit vector (grid-aligned i.e. either [0,1] (flux in y direction) or [1,0])"""
        if n_ == [0,1]:
            ie_ = np.array( np.arange(ie0 - int(floor(Nie/2)), ie0 + int(floor(Nie/2))) , dtype=int) 
        if n_ == [1,0]:
            ie_ = np.array( np.arange(ie0 - floor(Nie/2)*s.Nex, ie0 + floor(Nie/2)*s.Nex, s.Nex) , dtype=int) 
        return ie_
        
    
    def u_plane(s, ips_):
        """u field at plane
        r : plane reference point [1 x 2]
        Nip : # sensor nodes
        n_ : direction unit vec."""
        
        return s.u[ ips_ ]
        
    def reshape_field(s, u):
        """plot field defined on node mesh"""
        return u.reshape(s.Npy, s.Npx).transpose()
        
    def contourf_field(s, u, cm):
        ur = s.reshape_field(u)
        try:
            plt.contourf( s.Y, s.X, ur, cmap = cm)
        except:
            print "!!! plot error!"
            
    def debug_msg(s, msg, flag):
        if flag == True and isinstance(msg, str):
            print msg
        
#     def S_param(s, r1, r2, Nip, n_):
#         u1 = u_plane(s, r1, Nip, n_)
#         u2 = u_plane(s, r2, Nip, n_)
#         return u2/u1
    def isposint(s, n):
        return (isinstance(n, int) and n>=0)
        
#     def plot_x_srcsens(s):
#         x2 = s.x
#         iesrc_ = s.ie_sensor(s.iesrc0, Npsrc, n_)
#         x2[iesens_] = 0
#         x2[iesrc_] = 0
#         M.plot_faceMesh(x2)

# # # PML
# # # \forall e in mesh -> epsr, sx, sy -> 1/epsr, sx/sy, sy/sx, sxsy -> assembly -> K1, K2
# # # NB: Knablax requires sx/sy, Knablay requires sy/sx, both require 1/epsr
class Lagrange_bilinear_rect2d:
    def __init__(s):
        s.x,  s.y,  s.hx,  s.hy = symbols('x y hx hy')
        s.Ae = s.hx*s.hy
        s.w1 = 1/ s.Ae*( s.hx/2 -  s.x)*( s.hy/2 -  s.y)
        s.w2 = 1/ s.Ae*( s.hx/2 +  s.x)*( s.hy/2 -  s.y)
        s.w3 = 1/ s.Ae*( s.hx/2 +  s.x)*( s.hy/2 +  s.y)
        s.w4 = 1/ s.Ae*( s.hx/2 -  s.x)*( s.hy/2 +  s.y)
        
        s.w_ = [ s.w1,  s.w2,  s.w3,  s.w4]
        s.Nwe = len(s.w_) # bases/element
        
        # # # # # \int ( \grad(Wi) x \grad(Wj) ), forall i, j , grad = [d/dx, d/dy]
        s.Knablax = [[integrate( diff(u,s.x)*diff(v,s.x) ,(s.x,-s.hx/2,s.hx/2), (s.y, -s.hy/2, s.hy/2)) for u in s.w_] for v in s.w_]
        s.Knablay = [[integrate( diff(u,s.y)*diff(v,s.y) ,(s.x,-s.hx/2,s.hx/2), (s.y, -s.hy/2, s.hy/2)) for u in s.w_] for v in 
        s.w_]
        s.Knabla = s.Knablax + s.Knablay
        # # # # # \int ( (Wi) x (Wj) ) , "
        s.Kn = [[integrate( u*v ,(s.x,-s.hx/2,s.hx/2), (s.y, -s.hy/2, s.hy/2)) for u in s.w_] for v in s.w_]
        # # # # # \int ( \grad(Wi) x (Wj) ) , "
        s.Kqx = [[integrate( diff(u,s.x)*v ,(s.x,-s.hx/2,s.hx/2), (s.y, -s.hy/2, s.hy/2)) for u in s.w_] for v in s.w_]
        s.Kqy = [[integrate( diff(u,s.y)*v ,(s.x,-s.hx/2,s.hx/2), (s.y, -s.hy/2, s.hy/2)) for u in s.w_] for v in s.w_]
        

    def elementmtx(s, hxnumeric, hynumeric):
        """element matrices Knabla0, Kn0, Kqx0, Kqy0"""
        s.Knablax0 = np.zeros([s.Nwe, s.Nwe])
        s.Knablay0 = np.zeros([s.Nwe, s.Nwe])
        s.Kn0 = np.zeros([s.Nwe, s.Nwe])
        s.Kqx0 = np.zeros([s.Nwe, s.Nwe])
        s.Kqy0 = np.zeros([s.Nwe, s.Nwe])
        for i in range(s.Nwe):
            for j in range(s.Nwe):
                f11 = lambdify( (s.hx,s.hy) , s.Knablax[i][j]) # shitty # HACK #
                s.Knablax0[i][j] = f11(hxnumeric, hynumeric)
                f12 = lambdify( (s.hx,s.hy) , s.Knablay[i][j]) 
                s.Knablay0[i][j] = f12(hxnumeric, hynumeric)
                f2 = lambdify( (s.hx,s.hy) , s.Kn[i][j])
                s.Kn0[i][j] = f2(hxnumeric, hynumeric)
                f3 = lambdify( (s.hx,s.hy) , s.Kqx[i][j])
                s.Kqx0[i][j] = f3(hxnumeric, hynumeric)
                f4 = lambdify( (s.hx,s.hy) , s.Kqy[i][j])
                s.Kqy0[i][j] = f4(hxnumeric, hynumeric)
#         return s.Knablax0, s.Knablay0, s.Kn0, s.Kqx0, s.Kqy0

class RD(FEM_PML):
    def __init__(self, rmin, rmax, Nex, Ney):
        FEM_PML.__init__(self, rmin, rmax, Nex, Ney)
        self.elemmtx()
        self.nodes_full()
        self.sysmtx_idx()

#         print self.Ne
        self.diffc = np.ones([self.Ne, 1])
#         self.diffc_init( 1 )
        self.sysmtx_nopml()

#     def diffc_init(s, c):
#         """diffusion coeff array (isotropic diff.)"""
#         s.diffc = np.ones([s.Ne, 1])
#         if isinstance(c, int) or isinstance(c, float):
#             s.diffc = c*s.diffc
        
            
    def sysmtx_nopml(s):
        """ASSEMBLY of sys.mtx, NO PML, scalar diffusion coeff diffc"""
        K_effx1 = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        K_effy1 = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        K_eff2  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        Kqpex_eff  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        Kqpey_eff  = np.zeros(s.Nnpe * s.Ne, dtype=complex)
        for ii in range( s.Ne ):
            idx = np.ix_( np.arange( ii*s.Nnpe, (ii+1)*s.Nnpe))
            K_effx1.put(idx, s.x[ii] * s.diffc[ii] * s.Knablax0_)
            K_effy1.put(idx, s.x[ii] * s.diffc[ii] * s.Knablay0_)
            K_eff2.put(idx, s.Kn0_)
            Kqpex_eff.put(idx, s.x[ii] * s.Kqx0_)
            Kqpey_eff.put(idx, s.x[ii] * s.Kqy0_)
        s.K1 = coo_matrix((K_effx1 + K_effy1, (s.I, s.J)) ).tocsr()
        s.K2 = coo_matrix((K_eff2, (s.I, s.J)) ).tocsr()


def plot_x(M):
    if isinstance(M, FEM_PML) and M.check_x_exists and M.check_x:
        xr = M.x.reshape(M.Ney,M.Nex).transpose()
        plt.imshow(xr)
        plt.colorbar()
        plt.show()
    else:
        return None
        
# def plot_x_markers(M, c, *ie__):
#     try:
#         if isinstance(M, FEM_PML) and M.check_x_exists and M.isnnegint
#             xtemp = M.x
#             for ie_ in ie__:
#                 
#     except:
#         print "!!! plot_x_markers() : error"
            