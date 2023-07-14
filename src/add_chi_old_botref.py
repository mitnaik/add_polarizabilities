import h5py 
import numpy as np
import scipy.spatial
import sys


def checkandmap_kgrid(kgr, kpts):
	# ---
	# Maps the points in kpts to those in 
	# kgr and returns the indices such that
	# kpts[ik] = kgr[ind[ik]]
	#  Crashes if match not found.
	# ---

	tree = scipy.spatial.cKDTree(kgr)
	dist, ind = tree.query(kpts, distance_upper_bound = 1e-6)
	for ik in range(len(kpts)):
		#print(kpts[ik], kgr[ind[ik]])
		if ind[ik] > len(kgr):
			print("k-point not found in grid: kpts[ik]", kpts[ik], ind[ik],kgr[ind[ik]])
			sys.exit()
	return ind

def repeat_and_index(kpts, b1, b2):
	# ---
	# This function adds one shell of Umklapp reciprocal lattice
	# vectors (b1, b2) to kpts. It also stores the original indices
	# of the kpts in labs_rep. This is useful to index the k-points 
	# in the original array.  
	# ---

	nk = len(kpts)
	labs = np.arange(nk)
	kpts_rep = kpts
	labs_rep = labs
	for i in range(-1,2):
		for j in range(-1,2):
			# Skip the (0,0) Umklapp
			if i == 0 and j==0:
				continue
			# Umklapp of (i,j)
			sh = i*b1 + j*b2
			kpts_rep = np.concatenate((kpts + sh, kpts_rep), axis = 0)
			labs_rep = np.concatenate((labs, labs_rep), axis = 0)
	return kpts_rep, labs_rep

def find_qmatch(qpts_x, qpts_add, B_x, B_add):
	# ---
	# This function finds maps q-points in qpts_x to 
	# qpts_add + Umklapps(in B_add). It returns the matching
	# q-points and the index of these matches (ind_match) in the 
	# original qpts_add array.
	# NOTE: B_x and B_add are assumed to be in tpba units with the 
	# SAME alat
	# ---

	# The mapping is done in cartesian (tpba) coordinates
	qpts_add_cart = np.matmul(qpts_add, B_add)
	# Repeat and store indices of qpts_add
	qpts_add_rep, lab_add_rep = repeat_and_index(qpts_add_cart, B_add[0], B_add[1])
	qpts_x_cart = np.matmul(qpts_x, B_x)
	indx = checkandmap_kgrid(qpts_add_rep, qpts_x_cart)
	qpts_match = qpts_add_rep[indx]
	ind_match = lab_add_rep[indx]
	return qpts_match, ind_match

#def find_qmatch(qpts_x, qpts_add, B_x, B_add):
#	gumk = []
#	for ig in range(-1,2):
#		for igp in range(-1,2): 
#			gumk.append(ig*B_add[0] + igp*B_add[1])
#	gumk = np.array(gumk)
#	qpts_add_cart = np.matmul(qpts_add, B_add)
#	qpts_x_cart = np.matmul(qpts_x, B_x)
#	tree= scipy.spatial.cKDTree(qpts_add_cart)
#	sort_qx = np.zeros(len(qpts_x)) - 1
#	for iqx in range(len(qpts_x)):
#		qx_cart = qpts_x_cart[iqx]
#		dist, ind = tree.query(qx_cart, k = 1)
#		print(dist, ind, iqx)
#		if dist > 1e-5:
#			for gv in gumk:
#				qpts_umk = qpts_add_cart + gv
#				tree_umk = scipy.spatial.cKDTree(qpts_add_cart)
#				d_umk, ind_umk = tree_umk.query(qx_cart, k = 1)
#				if d_umk < 1e-5:
#					sort_qx[iqx] = ind_umk
#					print(qx_cart, qpts_umk[ind_umk])
#					break
#				else:
#					continue
#			if sort_qx[iqx] == -1:
#				print("ERROR: q match NOT FOUND!!!", qx_cart)
#		else:
#			print(qx_cart, qpts_add_cart[ind])
#			sort_qx[iqx] =	ind
#	return sort_qx
		

def add_chi_q(chiq_x, chiq_add, qpt_x, qpt_add, comp_x, comp_add, B_x, B_add):
	# ---
	# This function adds the polarizability in chiq_add to 
	# that in chiq_x if the q+G, q+G' of x
	# matches that of add. 
	# comp_x contains the G vectors ordered according to chiq_x
	# comp_add contains the G vectors ordered according to chiq_add
	# NOTE: B_x and B_add are assumed to be in tpba units with the 
	# SAME alat
	# ---

	nmtx_x = len(comp_x)
	nmtx_add = len(comp_add)
	# Convert components to tpiba
	# assuming alat is identical in the bot and top layer
	qpt_x_cart = np.matmul(qpt_x, B_x)
	qpt_add_cart = np.matmul(qpt_add, B_add)
	comp_x_cart = np.matmul(comp_x, B_x)
	qpG_x_cart = comp_x_cart + qpt_x_cart
	comp_add_cart = np.matmul(comp_add, B_add)
	qpG_add_cart = comp_add_cart + qpt_add_cart
	print("nmtx_add:", nmtx_add, len(comp_add_cart))
	print("nmtx_x:", nmtx_x, len(comp_x_cart))
	tree_main = scipy.spatial.cKDTree(qpG_add_cart)
	dist, ind = tree_main.query(qpG_x_cart)
	tol = 1e-5
	# Find intersection between x and add components
	cond_found = dist < tol
	for ii in range(nmtx_x):
		#print(qpG_add_cart[ind[ii]], qpG_x_cart[ii], dist[ii])
		for jj in range(nmtx_x):
			# Add only if both G and G' components of chiq_add matches chiq_x
			if cond_found[ii] and cond_found[jj]:
				chiq_x[:,:,ii,jj,:] = chiq_x[:,:,ii,jj,:] \
					+ chiq_add[:,:,ind[ii],ind[jj],:]
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
fout0.copy(f0_x['mats'], 'mats')	
fout.copy(f_x['mf_header'], 'mf_header')	
fout.copy(f_x['eps_header'], 'eps_header')	
fout.copy(f_x['mats'], 'mats')	

# Create matrix
#fout0.create_group('/mats')
#fout.create_group('/mats')


#fout0.create_dataset('/mats/matrix', shape=f0_x['/mats/matrix'].shape, dtype=f0_x['/mats/matrix'].dtype)
#fout.create_dataset('/mats/matrix', shape=f_x['/mats/matrix'].shape, dtype=f_x['/mats/matrix'].dtype)


#fout0.create_dataset('/mats/matrix-diagonal', shape=f0_x['/mats/matrix-diagonal'].shape, dtype=f0_x['/mats/matrix-diagonal'].dtype)
#fout.create_dataset('/mats/matrix-diagonal', shape=f_x['/mats/matrix-diagonal'].shape, dtype=f_x['/mats/matrix-diagonal'].dtype)


nmtx_x = f_x['eps_header/gspace/nmtx'][:]
nmtx0_x = f0_x['eps_header/gspace/nmtx'][:]
qpts_x = f_x['eps_header/qpoints/qpts'][:]
qpts_b = f_b['eps_header/qpoints/qpts'][:]
qpts_t = f_t['eps_header/qpoints/qpts'][:]
qpts0_x = f0_x['eps_header/qpoints/qpts'][:]
qpts0_b = f0_b['eps_header/qpoints/qpts'][:]
qpts0_t = f0_t['eps_header/qpoints/qpts'][:]
nmtx_max_x = f_x['eps_header/gspace/nmtx_max'][()]
nmtx0_max_x = f0_x['eps_header/gspace/nmtx_max'][()]
nmtx_x = f_x['eps_header/gspace/nmtx'][:]
nmtx_t = f_t['eps_header/gspace/nmtx'][:]
nmtx_b = f_b['eps_header/gspace/nmtx'][:]
nmtx0_t = f0_t['eps_header/gspace/nmtx'][:]
nmtx0_b = f0_b['eps_header/gspace/nmtx'][:]

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
	nmtx0_x = nmtx0_x[0]
	nmtx0_t = nmtx0_t[0]
	isrt_x = f0_x['eps_header/gspace/gind_eps2rho'][0] - 1
	isrt_t = f0_t['eps_header/gspace/gind_eps2rho'][0] - 1
	isrt_x = isrt_x[:nmtx0_x]
	isrt_t = isrt_x[:nmtx0_t]
	print("isrts:", len(isrt_x), len(isrt_t))
	chi0_bi[0] = add_chi_q(chi0_bi[0], mat0_t[0], qpts0_x[0], qpts0_t[0], comp_x[isrt_x], comp_t[isrt_t], bvec_x, bvec_t)

#sys.exit()

#qgrid_t_rep, lab_t_rep = repeat_and_index(qgrid_t, b1_mo, b2_mo)

qgrid_match, ind_match = find_qmatch(qpts_x, qpts_t, bvec_x, bvec_t)
#else:
#	chi0_bi = mat0_t

#chi0_bi[:,0,0,:,:,:] = mat0_W[:,0,0,:,:,:] + mat0_Mo[:,0,0,:,:,:]
chi_bi = np.zeros((nq_x, 1, 1, nmtx_max_x, nmtx_max_x, 2))
print(np.shape(mat_b), np.shape(mat_t), np.shape(mat_x))
chi_bi = mat_x
#chi_bi[:,0,0,:,:,:] = mat_W[:,0,0,:,:,:] + mat_Mo[:,0,0,:,:,:]
for iqx in range(nq_x):
	isrt_x = f_x['eps_header/gspace/gind_eps2rho'][iqx] - 1
	isrt_t = f_t['eps_header/gspace/gind_eps2rho'][ind_match[iqx]] - 1
	#print("isrts:", len(isrt_x), len(isrt_t))
	isrt_x = isrt_x[:nmtx_x[iqx]]
	isrt_t = isrt_t[:nmtx_t[ind_match[iqx]]]
	chi_bi[iqx] = add_chi_q(chi_bi[iqx], mat_t[ind_match[iqx]], qpts_x[iqx], qpts_t[ind_match[iqx]], comp_x[isrt_x], comp_t[isrt_t], bvec_x, bvec_t)
	print(iqx)
	#chi_bi[iqx,0,0,:nmtx_x[iq], :nmtx_x[iq],:] = mat_W[iq,0,0,:nmtx_x[iq],:nmtx_x[iq],:] + mat_Mo[iq,0,0,:nmtx_x[iq],:nmtx_x[iq],:]
#print(nmtx_W[0],chi_bi[0,0,0,nmtx_max_x-1,0,0], mat_W[0,0,0,nmtx_max_x-1,0,0], mat_Mo[0,0,0,nmtx_max_x-1,0,0])

#for iq in range(nq_x):
#	chi0_bi[iq,0,0,:,:,0] = 
#		mat0_W[iq,0,0,:,:,0] + mat0_Mo[iq,0,0,:,:,0]
#	chi0_bi[iq,0,0,im,jm,1] = 
#		mat0_W[iq,0,0,:,:,1] + mat0_Mo[iq,0,0,:,:,1]
fout0['mats/matrix'][:] = chi0_bi
fout['mats/matrix'][:] = chi_bi

# Copy matrix-diagonal, matrix_subspace, matrix_eigenvec, matrix_fulleps0
#fout0.copy(f0_x['mats/matrix-diagonal'], 'mats/matrix-diagonal')	
#fout.copy(f_x['mats/matrix-diagonal'], 'mats/matrix-diagonal')	
#fout0.copy(f0_x['mats/matrix_subspace'], 'mats/matrix_subspace')	
#fout.copy(f0_x['mats/matrix_subspace'], 'mats/matrix_subspace')	
#fout0.copy(f0_x['mats/matrix_eigenvec'], 'mats/matrix_eigenvec')	
#fout.copy(f0_x['mats/matrix_eigenvec'], 'mats/matrix_eigenvec')	
#fout0.copy(f0_x['mats/matrix_fulleps0'], 'mats/matrix_fulleps0')	
#fout.copy(f0_x['mats/matrix_fulleps0'], 'mats/matrix_fulleps0')	
