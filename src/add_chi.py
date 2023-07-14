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


def main():

	# Read the bottom and top layer chi
	print("WARNING: The two layers are assumed to have the SAME value of alat!!!")
	f0_t = h5py.File("chi0mat_top.h5", 'r')
	f_t = h5py.File("chimat_top.h5", 'r')
	f0_b = h5py.File("chi0mat_bot.h5", 'r')
	f_b = h5py.File("chimat_bot.h5", 'r')
	
	fout0 = h5py.File("chi0mat_bi.h5", 'w')
	fout = h5py.File("chimat_bi.h5", 'w')
	
	if len(sys.argv) == 1:
		print("Please provide argument for intact layer: 'bot' or 'top'. \n This indicates which layer's q-points are kept intact. The other layer will be added to the intact layer")
		print("Aborting")
		sys.exit()
	else:
		intact_layer = sys.argv[1]

	# We define f_x to correspond to the intact layer and
	# f_add to the layer's chi that is being added
	if intact_layer == 'bot':
		f_x = f_b
		f0_x = f0_b
		f_add = f_t
		f0_add = f0_t
	elif intact_layer == 'top':
		f_x = f_t
		f0_x = f0_t
		f_add = f_b
		f0_add = f0_b
	
	# Copy mf and eps header from the intact layer files
	# to the bilayer chi
	fout0.copy(f0_x['mf_header'], 'mf_header')	
	fout0.copy(f0_x['eps_header'], 'eps_header')	
	fout0.copy(f0_x['mats'], 'mats')	
	fout.copy(f_x['mf_header'], 'mf_header')	
	fout.copy(f_x['eps_header'], 'eps_header')	

	# Note: We only modify mats/matrix later
	fout.copy(f_x['mats'], 'mats')	
	
	nmtx_x = f_x['eps_header/gspace/nmtx'][:]
	nmtx0_x = f0_x['eps_header/gspace/nmtx'][:]
	qpts_x = f_x['eps_header/qpoints/qpts'][:]
	qpts_add = f_add['eps_header/qpoints/qpts'][:]
	qpts0_x = f0_x['eps_header/qpoints/qpts'][:]
	qpts0_add = f0_add['eps_header/qpoints/qpts'][:]
	nmtx_max_x = f_x['eps_header/gspace/nmtx_max'][()]
	nmtx0_max_x = f0_x['eps_header/gspace/nmtx_max'][()]
	nmtx_x = f_x['eps_header/gspace/nmtx'][:]
	nmtx_max_add = f_add['eps_header/gspace/nmtx_max'][()]
	nmtx0_max_add = f0_add['eps_header/gspace/nmtx_max'][()]
	nmtx_add = f_add['eps_header/gspace/nmtx'][:]
	nmtx0_add = f0_add['eps_header/gspace/nmtx'][:]
	
	
	nq_x = f_x['eps_header/qpoints/nq'][()]
	nq0_x = f0_x['eps_header/qpoints/nq'][()]
	mat_x = f_x['mats/matrix'][()]
	mat0_x = f0_x['mats/matrix'][()]
	mat0_add = f0_add['mats/matrix'][()]
	
	nq_add = f_add['eps_header/qpoints/nq'][()]
	nq0_add = f0_add['eps_header/qpoints/nq'][()]
	mat_add = f_add['mats/matrix'][()]
	
	comp_x = f_x['mf_header/gspace/components'][:]
	comp_add = f_add['mf_header/gspace/components'][:]
	bvec_x = f_x['mf_header/crystal/bvec'][:]
	bvec_add = f_add['mf_header/crystal/bvec'][:]
	
	chi0_bi = np.zeros((1, 1, 1, nmtx0_max_x, nmtx0_max_x, 2))
	chi0_bi = mat0_x
	nmtx0_x = nmtx0_x[0]
	nmtx0_add = nmtx0_add[0]
	isrt_x = f0_x['eps_header/gspace/gind_eps2rho'][0] - 1
	isrt_add = f0_add['eps_header/gspace/gind_eps2rho'][0] - 1
	isrt_x = isrt_x[:nmtx0_x]
	isrt_add = isrt_add[:nmtx0_add]
	chi0_bi[0] = add_chi_q(chi0_bi[0], mat0_add[0], qpts0_x[0], qpts0_add[0], comp_x[isrt_x], comp_add[isrt_add], bvec_x, bvec_add)
	print("chi0mat of bilayer written")
	
	
	qgrid_match, ind_match = find_qmatch(qpts_x, qpts_add, bvec_x, bvec_add)
	
	chi_bi = np.zeros((nq_x, 1, 1, nmtx_max_x, nmtx_max_x, 2))
	chi_bi = mat_x

	for iqx in range(nq_x):
		print("Working on q-point:",iqx)
		isrt_x = f_x['eps_header/gspace/gind_eps2rho'][iqx] - 1
		isrt_add = f_add['eps_header/gspace/gind_eps2rho'][ind_match[iqx]] - 1
		isrt_x = isrt_x[:nmtx_x[iqx]]
		isrt_add = isrt_add[:nmtx_add[ind_match[iqx]]]
		# Make sure to send comp_x[isrt_x] and comp_add[isrt_add] which is the 
		# ordering of G-vectors in chi
		chi_bi[iqx] = add_chi_q(chi_bi[iqx], mat_add[ind_match[iqx]], qpts_x[iqx], qpts_add[ind_match[iqx]], comp_x[isrt_x], comp_add[isrt_add], bvec_x, bvec_add)
	
	fout0['mats/matrix'][:] = chi0_bi
	fout['mats/matrix'][:] = chi_bi
	print("chimat of bilayer written")
	
if __name__ == '__main__':
	main()
