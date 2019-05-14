"""
Module for doing analysis on the resulting trajectories

"""
import MDAnalysis as mda
from MDAnalysis.transformations.translate import *
from MDAnalysis.lib.distances import calc_angles
from MDAnalysis.analysis.distances import self_distance_array as sda
from MDAnalysis.analysis.distances import distance_array
from pmda.parallel import ParallelAnalysisBase
import scipy as sp
from scipy import stats
from tqdm import tqdm_notebook
from itertools import combinations

"""
Loads the trajectory of a particular production run

"""
def load_run(gas, trialnum, temp):
    directory = 'production/{}/trial_{}/{}K/dump.lammpstraj'.format(gas, trialnum, temp)
    u = mda.Universe(directory, format='lammpsdump')
    return u

"""
Read the log file from a particular production run

"""
def read_log(gas, trialnum, temp):
    directory = 'production/{}/trial_{}/{}K/npt_{}.out'.format(gas, trialnum, temp, temp)
    with open(directory) as f:
        lines = f.readlines()
    labels = []
    data = []
    read = False
    for line in lines:
        if line.startswith('Step'):
            read = True
            labels = line.split()
        elif line.startswith('Loop'):
            read = False
        elif read:
            data.append(np.array(line.split(), dtype=float))
    data = np.array(data).T
    return {k: v for k, v in zip(labels, data)}



"""
Provided an array of all the angles of a water oxygen and its neighboring oxygens,
computes the mean-AOP

"""
def aop(angles):
    co = np.cos(angles)
    ang = np.cos(109.47*np.pi/180)
    return np.mean((np.abs(co)*co + ang**2)**2) if angles.size > 0 else 1

"""
Nicely optimized AOP-calculating function,
operates on a particular universe ``u`` and timestep ``ts``
Gets most of its optimizations from only invoking the calc_angles function once, over
all necessary angles

Only counts waters that have 3.5 Angstrom neighbors in binning/calculating AOP

"""
def aops(oxy, ts):
    #oxy = u.select_atoms('type 3')
    ts1 = center_in_box(oxy, wrap=True)(ts)
    oxy.atoms.wrap()
    dists = sp.spatial.distance.squareform(sda(oxy.positions,
                                           oxy.dimensions))
    # in format (x-pos, aop)
    AOPs = np.ones((len(oxy), 2))
    all_inds = []
    num_each = []
    for i in range(len(oxy)):
        nearby = np.where(dists[i] <= 3.5)[0]
        nearby = nearby[nearby != i]
        if nearby.size > 1:
            AOPs[i, 0] = oxy[i].position[0]
            inds = np.array(list(combinations(nearby, 2)))
            combs = inds.shape[0]
            xyz_inds = np.insert(inds, 1, np.full(combs, i), axis=1)
            all_inds.append(xyz_inds)
            num_each.append(combs)
        elif nearby.size == 1:
            num_each.append(0)
            AOPs[i, 0] = oxy[i].position[0]
            AOPs[i, 1] = 0.1 # it has one neighbor, so just say it's liquid
        else:
            num_each.append(0)
            AOPs[i, 0] = np.nan
    all_pos = oxy.positions[np.concatenate(all_inds)]
    angles = calc_angles(all_pos[:, 0, :], all_pos[:, 1, :],
                         all_pos[:, 2, :], box=oxy.dimensions)
    i = 0
    # go through the angles and calculate AOP
    for j, num in enumerate(num_each):
        AOPs[j, 1] = aop(angles[i:i+num])
        i += num
    # remove the ones that were too far away to begin with
    # AOPs = AOPs[~np.isnan(AOPs).any(axis=1)]
    # left, right = np.min(oxy.positions[:, 0]), np.max(oxy.positions[:, 0])
    # AOP_means = stats.binned_statistic(AOPs[:, 0], AOPs[:, 1], bins=8)
    return AOPs

"""
Class for doing parallelized AOP analysis over an entire trajectory, using
the PMDA library

proides a nice ~3x speedup for 4 cores over previous version

"""
class AOPAnalysis(ParallelAnalysisBase):
    def __init__(self, atomgroup):
        self._ag = atomgroup
        super(AOPAnalysis, self).__init__(atomgroup[0].universe,
                                          self._ag)

    def _single_frame(self, ts, agroups):
        # REQUIRED
        # called for every frame. ``ts`` contains the current time step
        # and ``agroups`` a tuple of atomgroups that are updated to the
        # current frame. Return result of `some_function` for a single
        # frame
        return aops(agroups[0], ts)

    def _conclude(self):
        # REQUIRED
        # Called once iteration on the trajectory is finished. Results
        # for each frame are stored in ``self._results`` in a per block
        # basis. Here those results should be moved and reshaped into a
        # sensible new variable.
        self.results = np.vstack(self._results)

def aop_bins(aops):
    heat = np.zeros((aops.shape[0], 8))
    for i, aop in enumerate(aops):
        aop = aop[~np.isnan(aop).any(axis=1)]
        aop = aop[~(aop == 1).any(axis=1)]
        aop = aop[~(aop == 0.1).any(axis=1)]
        left, right = np.min(aop[:, 0]), np.max(aop[:, 0])
        #print(i, left, right)
        heat[i] = stats.binned_statistic(aop[:, 0], aop[:, 1], bins=8)[0]
    return heat

# tells how many methanes are (a) in the water region, (b) surrounded
# by non-clathrate waters, and (c) surrounded by clathrate waters
# gastype 1 is ch4, 5 is co2/n2
def gas_count(gas_sel, aops, ts):
    oxy_sel = gas_sel.universe.select_atoms('type 3')
    # bounds as determined by AOP
    aop = aops[~np.isnan(aops).any(axis=1)]
    aop = aop[~(aop == 1).any(axis=1)]
    aop = aop[~(aop == 0.1).any(axis=1)]


    # wrap the atom positions so water box is centered as in AOP
    ts1 = center_in_box(oxy_sel, wrap=True)(ts)
    oxy_sel.universe.atoms.wrap()

    #left, right = np.min(aop[:, 0]), np.max(aop[:, 0])
    left, right = np.min(oxy_sel.positions[:, 0]), np.max(oxy_sel.positions[:, 0])

    # start by determining counts of gas in each of 8 bins
    gas_pos = gas_sel.positions[:, 0]
    # these are the indices of the gases within the bounds of the water box
    gas_in_box = np.where(np.logical_and(np.min(oxy_sel.positions[:, 0]) <= gas_pos,
                          gas_pos <= np.max(oxy_sel.positions[:, 0])))[0]
    tot_counts = stats.binned_statistic(gas_pos[gas_in_box], gas_in_box,
                           statistic='count', bins=8, range=(left, right))
    tot_counts = tot_counts[0]

    # now we want to determine the number of these gases which are in
    # the hydrate
    # our criterion for "hydrate-like" is AOP < 0.05; criterion for gas
    # participating in the hydrate is >10 hydrate-like waters within 5.5A

    d = distance_array(gas_sel.positions, oxy_sel.positions,
                       box=oxy_sel.universe.dimensions)
    gasind, oxind = np.where(d <= 5.5)
    ox_aops = aops[:, 1][oxind]

    # array to store 1 if given gas is participating in clathrate,
    # 0 otherwise
    box_gas_clath = np.zeros_like(gas_in_box)

    for i, g in enumerate(gas_in_box):
        g_aops = ox_aops[gasind == g]
        g_clath_h2os = len(g_aops[g_aops < 0.05])
        if g_clath_h2os >= 10:
            box_gas_clath[i] = 1
    clath_counts = stats.binned_statistic(gas_pos[gas_in_box],
                                          box_gas_clath, statistic='sum',
                                          bins=8, range=(left, right))
    clath_counts = clath_counts[0]
    return np.stack((tot_counts, clath_counts))


"""
Class for doing parallelized gas count analysis over an entire trajectory, using
the PMDA library

"""
class GasCountAnalysis(ParallelAnalysisBase):
    def __init__(self, atomgroup, aops):
        self._ag = atomgroup
        self.aops = aops
        super(GasCountAnalysis, self).__init__(atomgroup[0].universe,
                                          self._ag)

    def _single_frame(self, ts, agroups):
        # REQUIRED
        # called for every frame. ``ts`` contains the current time step
        # and ``agroups`` a tuple of atomgroups that are updated to the
        # current frame. Return result of `some_function` for a single
        # frame
        return gas_count(agroups[0], self.aops[int(ts.time)], ts)

    def _conclude(self):
        # REQUIRED
        # Called once iteration on the trajectory is finished. Results
        # for each frame are stored in ``self._results`` in a per block
        # basis. Here those results should be moved and reshaped into a
        # sensible new variable.
        self.results = np.vstack(self._results)

"""
function to aggregate a metric over the three trials
aka takes the mean over all three trials of binned data

"""
def aggregate(metric, gas, temp):
    if metric == 'aop':
        data = []
        for trial in range(1, 4):
            aop = np.load('production/{}/trial_{}/{}K/aop.npy'.format(gas, trial, temp))
            data.append(aop_bins(aop))
    else:
        data = []
        for trial in range(1, 4):
            met = np.load('production/{}/trial_{}/{}K/{}.npy'.format(gas, trial, temp, metric))
            data.append(met)
    return np.stack(data).mean(axis=0)


def bin_avg_matrix(matrix):
    shape = matrix.shape
    rest = matrix[1:]
    res = np.zeros((int((matrix.shape[0]-1)/100), shape[1]))
    for i in range(int((matrix.shape[0]-1)/100)):
        res[i] = rest[i*100:(i+1)*100].mean(axis=0)
    return res
