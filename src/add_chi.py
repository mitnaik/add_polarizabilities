import h5py 
import numpy as np
import scipy.spatial
import sys

def add_chi_q(chiq_x, chiq_add, comp_x, comp_add, B_x, B_add):
	nmtx_x = len(chiq_x[0,0,:,0,0] )
	# Convert components to tpiba
	# assuming alat is identical in the bot and top layer
	comp_x_cart = np.matmul(comp_x, B_x)
	comp_add_cart = np.matmul(comp_add, B_add)
	tree_main = scipy.spatial.cKDTree(comp_add_cart)
	dist, ind = tree_main.query(comp_x_cart)
	tol = 1e-5
	# Find intersection between x and add components
	cond_found = dist < tol
	for ii in range(nmtx_x):
		for jj in range(nmtx_x):
			# Add only if both G and G' components of chiq_add matches chiq_x
			if cond_found[ii] and cond_found[jj]:
				chiq_x[:,:,ii,jj,:] = chiq_x[:,:,ii,jj,:] + chiq_add[:,:,ind[ii],ind[jj],:]
	return chiq_x

f0_t = h5py.File("chi0mat_top.h5", 'r')
f_t = h5py.File("chimat_top.h5", 'r')
f0_b = h5py.File("chi0mat_bot.h5", 'r')
f_b = h5py.File("chimat_bot.h5", 'r')

#f0_x = h5py.File("eps0mat_x.h5", 'r')
#f_x = h5py.File("epsmat_x.h5", 'r')

fout0 = h5py.File("chi0mat_bi.h5", 'w')
fout = h5py.File("chimat_bi.h5", 'w')



intact_layer = 'bot'
if intact_layer == 'bot':
	f_x = f_b
	f0_x = f0_b
elif intact_layer == 'top':
	f_x = f_t
	f0_x = f0_t

# Copy mf and eps header from the intact layer files
fout0.copy(f0_x['mf_header'], 'mf_header')	
fout0.copy(f0_x['eps_header'], 'eps_header')	
fout.copy(f_x['mf_header'], 'mf_header')	
fout.copy(f_x['eps_header'], 'eps_header')	

# Create matrix
fout0.create_group('/mats')
fout.create_group('/mats')


fout0.create_dataset('/mats/matrix', shape=f0_x['/mats/matrix'].shape, dtype=f0_x['/mats/matrix'].dtype)
fout.create_dataset('/mats/matrix', shape=f_x['/mats/matrix'].shape, dtype=f_x['/mats/matrix'].dtype)


#fout0.create_dataset('/mats/matrix-diagonal', shape=f0_x['/mats/matrix-diagonal'].shape, dtype=f0_x['/mats/matrix-diagonal'].dtype)
#fout.create_dataset('/mats/matrix-diagonal', shape=f_x['/mats/matrix-diagonal'].shape, dtype=f_x['/mats/matrix-diagonal'].dtype)


nmtx_x = f_x['eps_header/gspace/nmtx'][()]
nmtx_max_x = f_x['eps_header/gspace/nmtx_max'][()]
nmtx0_max_x = f0_x['eps_header/gspace/nmtx_max'][()]
nmtx_t = f_t['eps_header/gspace/nmtx'][()]
nmtx_b = f_b['eps_header/gspace/nmtx'][()]

nq_x = f_x['eps_header/qpoints/nq'][()]
nq0_x = f0_x['eps_header/qpoints/nq'][()]
mat_x = f_x['mats/matrix'][()]
mat_t = f_t['mats/matrix'][()]
mat_b = f_b['mats/matrix'][()]
mat0_t = f0_t['mats/matrix'][()]
mat0_b = f0_b['mats/matrix'][()]
diag_t = f_t['mats/matrix-diagonal'][()]
diag_b = f_b['mats/matrix-diagonal'][()]
diag0_t = f0_t['mats/matrix-diagonal'][()]
diag0_b = f0_b['mats/matrix-diagonal'][()]


comp_x = f_x['mf_header/gspace/components'][:]
comp_b = f_b['mf_header/gspace/components'][:]
comp_t = f_t['mf_header/gspace/components'][:]


bvec_x = f_x['mf_header/crystal/bvec'][:]
bvec_b = f_b['mf_header/crystal/bvec'][:]
bvec_t = f_t['mf_header/crystal/bvec'][:]
print(nq_x, nmtx_max_x, len(comp_x))
#sys.exit()

chi0_bi = np.zeros((1, 1, 1, nmtx0_max_x, nmtx0_max_x, 2))
if intact_layer == 'bot':
	chi0_bi = mat0_b
	isrt_x = f0_x['eps_header/gspace/gind_eps2rho'][0] - 1
	print(isrt_x[0:30])
	isrt_t = f0_t['eps_header/gspace/gind_eps2rho'][0] - 1
	chi0_bi = add_chi_q(chi0_bi, mat0_t, comp_x[isrt_x], comp_t[isrt_t], bvec_x, bvec_t)
sys.exit()
#else:
#	chi0_bi = mat0_t

#chi0_bi[:,0,0,:,:,:] = mat0_W[:,0,0,:,:,:] + mat0_Mo[:,0,0,:,:,:]
chi_bi = np.zeros((nq_x, 1, 1, nmtx_max_x, nmtx_max_x, 2))
print(np.shape(mat_Mo), np.shape(mat_W), np.shape(mat_x))
#chi_bi[:,0,0,:,:,:] = mat_W[:,0,0,:,:,:] + mat_Mo[:,0,0,:,:,:]
for iq in range(nq_x):
	chi_bi[iq,0,0,:nmtx_x[iq], :nmtx_x[iq],:] = mat_W[iq,0,0,:nmtx_x[iq],:nmtx_x[iq],:] + mat_Mo[iq,0,0,:nmtx_x[iq],:nmtx_x[iq],:]
#print(nmtx_W[0],chi_bi[0,0,0,nmtx_max_x-1,0,0], mat_W[0,0,0,nmtx_max_x-1,0,0], mat_Mo[0,0,0,nmtx_max_x-1,0,0])

#for iq in range(nq_x):
#	chi0_bi[iq,0,0,:,:,0] = 
#		mat0_W[iq,0,0,:,:,0] + mat0_Mo[iq,0,0,:,:,0]
#	chi0_bi[iq,0,0,im,jm,1] = 
#		mat0_W[iq,0,0,:,:,1] + mat0_Mo[iq,0,0,:,:,1]
fout0['mats/matrix'][:] = chi0_bi
fout['mats/matrix'][:] = chi_bi

# Copy matrix-diagonal, matrix_subspace, matrix_eigenvec, matrix_fulleps0
fout0.copy(f0_x['mats/matrix-diagonal'], 'mats/matrix-diagonal')	
fout.copy(f0_x['mats/matrix-diagonal'], 'mats/matrix-diagonal')	
#fout0.copy(f0_x['mats/matrix_subspace'], 'mats/matrix_subspace')	
#fout.copy(f0_x['mats/matrix_subspace'], 'mats/matrix_subspace')	
#fout0.copy(f0_x['mats/matrix_eigenvec'], 'mats/matrix_eigenvec')	
#fout.copy(f0_x['mats/matrix_eigenvec'], 'mats/matrix_eigenvec')	
#fout0.copy(f0_x['mats/matrix_fulleps0'], 'mats/matrix_fulleps0')	
#fout.copy(f0_x['mats/matrix_fulleps0'], 'mats/matrix_fulleps0')	
