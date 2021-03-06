"""
Nico Christianson and Will Weiter
Code for APCOMP 275 final project at Harvard
May 2019

Facilitates the automated production and molecular dynamics simulation of
methane clathrate structures interfaced with various gases, for the study of
methane recovery and gas sequestration in the clathrates.

"""

import ase
from ase import io, Atom
from ase.build import cut
from string import Template
from labutil.src.objects import *
from ase.io.lammpsrun import read_lammps_dump
import numpy as np

#fake: real
placeholder_dict = {}
#real: fake
rev_dict = {}


"""
Creates a methane clathrate structure as an ASE cell

size: gives number of unit cells in each dimension; can be either an integer
    (if you want a cubic structure) or a numpy array
    of three elements (corresponding to x, y, and z dimensions)

"""
def create_clathrate(size):
    x = io.read('jp111328v_si_001.cif')
    # need to reorder things for the tip4p water model, so read the cif file to do so
    with open('jp111328v_si_001.cif') as f:
        txt = f.readlines()
    read = False
    nums = []
    for line in txt:
        if line.startswith('_geom_bond_site_symmetry_2'):
            read = True
            continue
        elif read and line.startswith('O'):
            nums.append(int(line.split()[1][1:]))
        else:
            continue
    # now we reorder, because we need our water molecules to list O
    # first, followed by the two Hs, for the tip4p model
    # We also replace methane hydrogens with S as a placeholder in ase
    acc = x[0:1]
    acc.append(x[nums[0]-1])
    acc.append(x[nums[1]-1])
    for i in range(1, 46):
        acc.append(x[i])
        acc.append(x[nums[2*i]-1])
        acc.append(x[nums[2*i+1]-1])
    for i in range(138, len(x)):
        if x[i].symbol == 'C':
            acc.append(x[i])
        else:
            new_xi = x[i]
            new_xi.symbol = 'S' # replace methane hydrogen with s as placeholder
            acc.append(new_xi)
    # now let's create our supercell; need to use cut in order to make sure
    # atoms aren't removed
    cell_len = acc.cell[0, 0]
    m = np.identity(3) * size
    mh_super = cut(acc, m[0], m[1], m[2], tolerance=1e-5)
    # now let's move atoms if they were previously across the pbc
    i = 0
    while i < len(mh_super):
        if mh_super.numbers[i] == 8:
            natoms = 3
        elif mh_super.numbers[i] == 6:
            natoms = 5
        else:
            raise ValueError
        for d in range(3):
            pos_d = mh_super.positions[i:i+natoms, d]
            max_d = np.max(pos_d)
            pos_d[np.abs(pos_d - max_d) > cell_len/2] += acc.cell[d, d]
        i += natoms
    mh_super.positions[:, 1] += cell_len/2
    mh_super.positions[:, 1] %= (cell_len*size[1])
    # scale positions based off Tung et al, 2011
    mh_super.cell *= 23.74/(2*cell_len)
    mh_super.positions *= 23.74/(2*cell_len)
    return mh_super

"""
code to automatedly create a box of gas for appending to the clathrate cell
still in testing

"""
def create_gas(dimensions, gastype, outname, number):
    # creates and populates gas region to be appended to clathrate structure

    runpath = Dir(path=os.path.join('gas_coords', 'gases_input'))
    prepare_dir(runpath.path)

    packmol_in = open(runpath,"{}.inp".format(gastype), "w+")
    text = "tolerance 2.0\nfiletype pdb\n"
    text += "output {}.pdb\n\n".format(outname)
    text += "structure {}.pdb\n".format(gastype)
    text += "  number {}\n".format(number)
    text += "  inside box 2. 0. 0. "
    text += "{} {} {}".format(dimensions[0], dimensions[1], dimensions[2])
    text += "\nend structure"

    packmol_in.write(text)
    packmol_in.close()

    packmol_code = ExternalCode(path=os.eviron['PACKMOL_COMMAND'])
    packmol_run = "{} < {}".format(packmol_code, packmol_in.path)

    run_command(packmol_run)
    print("Maybe it worked?")

"""
expands the x-direction of a clathrate cell and fills
with gas molecules from file

clath: the ase clathrate structure
vac_size: length of the vacuum in the x-direction in angstroms
gas_pos_file: file directory of the gas positions to fill in
needs_ghostie: tells us whether we're using a linear gas model (such as
    n2 or h2) which requires the creation of a ghost atom in the center for
    charge reasons
"""
def fill_gas(clath, vac_size, gas_pos_file, needs_ghostie):
    original_xdim = clath.cell[0, 0].copy()
    # add our vacuum region
    clath.cell[0, 0] += vac_size
    # see what extra atomic numbers are available:
    avail_elts = list(set(range(1, 11)).difference(set(clath.get_atomic_numbers())))
    with open(gas_pos_file) as f:
        lines = f.readlines()
    # if we need a ghostie, then implement this
    if needs_ghostie:
        i = 0
        while i < len(lines):
            if lines[i].startswith('ATOM'):
                list1 = lines[i].split()
                list2 = lines[i+1].split()
                # get positions
                pos1 = np.array(list1[5:8], dtype=float)
                pos1[0] += original_xdim
                pos2 = np.array(list2[5:8], dtype=float)
                pos2[0] += original_xdim
                # get ghost atom position
                pos_ghost = (pos1 + pos2) / 2
                if list1[2] not in placeholder_dict.values():
                    ghost_number = avail_elts.pop(0)
                    gatom = Atom(ghost_number, pos_ghost)
                    # add to dicts
                    placeholder_dict[gatom.symbol] = 'Ghostie'
                    rev_dict['Ghostie'] = gatom.symbol
                    clath.append(gatom)

                    # now add the real atoms
                    atom_number = avail_elts.pop(0)
                    atom1 = Atom(atom_number, pos1)
                    atom2 = Atom(atom_number, pos2)
                    # add to dicts
                    placeholder_dict[atom1.symbol] = list1[2]
                    rev_dict[list1[2]] = atom1.symbol
                    clath.append(atom1)
                    clath.append(atom2)
                else:
                    # ghost atom first
                    gelt = rev_dict['Ghostie']
                    gatom = Atom(gelt, pos_ghost)
                    clath.append(gatom)
                    # now real atoms
                    elt1 = rev_dict[list1[2]]
                    elt2 = rev_dict[list2[2]]
                    atom1, atom2 = Atom(elt1, pos1), Atom(elt2, pos2)
                    clath.append(atom1)
                    clath.append(atom2)
                i += 2
            else:
                i += 1
    else:
        for line in lines:
            if line.startswith('ATOM'):
                linelst = line.split()
                # position of this new atom
                position = np.array(linelst[5:8], dtype=float)
                position[0] += original_xdim
                # if we haven't already made a placeholder element here
                if linelst[2] not in placeholder_dict.values():
                    atom_number = avail_elts.pop(0)
                    atom = Atom(atom_number, position)
                    # make the dictionaries so we can switch back and forth
                    placeholder_dict[atom.symbol] = linelst[2]
                    rev_dict[linelst[2]] = atom.symbol
                    clath.append(atom)
                else:
                    elt = rev_dict[linelst[2]]
                    atom = Atom(elt, position)
                    clath.append(atom)
    return clath


"""
write a lammps data topology file from an ase struc
has to replace ase placeholders with what should be the actual masses/kinds
also needs to infer and write bond and angle data, which was not included in the
cif or pdb files explicitly, and which ase doesn't give

struc: ase struc to convert
runpath: runpath to dump the file to

"""
def write_lammps_data(struc, runpath):
    """Make LAMMPS struc data"""

    # make the ol' switcheroo of sulfur to methane hydrogen
    struc.species['S'] = {'mass': 1.00794, 'kind': 4}
    struc.species['H']['kind'] = 2
    struc.species['O']['kind'] = 3
    struc.species['C']['kind'] = 1
    # and do it for our gas as well
    for i, fake in enumerate(placeholder_dict.keys()):
        if placeholder_dict[fake] == 'Ghostie':
            struc.species[fake] = {'mass': 1e-20, 'kind': 5+i}
        else:
            struc.species[fake] = {'mass': ase.Atom(placeholder_dict[fake]).mass,
                                   'kind': 5 + i} # make the ol' switcheroo
    # gotta real quick start by computing bond and angle info.
    # bond info:
    btxt = '\nBonds\n\n'
    i = 0
    bond_ind = 1
    while i < len(struc.positions):
        # H2O case
        if (struc.positions[i][0] == 'O' and
            struc.positions[i+1][0] == 'H' and struc.positions[i+2][0] == 'H'):
            # o-h is bond type 1; hydrogens must follow the oxygen
            bond_type = 1
            btxt += '{} {} {} {} # O:H\n'.format(bond_ind, bond_type, i+1, i+2)
            btxt += '{} {} {} {} # O:H\n'.format(bond_ind+1, bond_type, i+1, i+3)
            bond_ind += 2
            i += 3
        # CH4 case
        elif (struc.positions[i][0] == 'C' and
              struc.positions[i+1][0] == 'S' and struc.positions[i+2][0] == 'S' and
              struc.positions[i+3][0] == 'S' and struc.positions[i+4][0] == 'S'):
            # c-h is bond type 2
            bond_type = 2
            btxt += '{} {} {} {} # C:H\n'.format(bond_ind, bond_type, i+1, i+2)
            btxt += '{} {} {} {} # C:H\n'.format(bond_ind+1, bond_type, i+1, i+3)
            btxt += '{} {} {} {} # C:H\n'.format(bond_ind+2, bond_type, i+1, i+4)
            btxt += '{} {} {} {} # C:H\n'.format(bond_ind+3, bond_type, i+1, i+5)
            bond_ind += 4
            i += 5
        # we're only buliding this to allow CO2 or N2 or H2
        # CO2 case
        elif ('C' in rev_dict.keys() and
              struc.positions[i][0] == rev_dict['C'] and
              struc.positions[i+1][0] == rev_dict['O'] and
              struc.positions[i+2][0] == rev_dict['O']):
            bond_type = 3
            btxt += '{} {} {} {} # C:O\n'.format(bond_ind, bond_type, i+1, i+2)
            btxt += '{} {} {} {} # C:O\n'.format(bond_ind+1, bond_type, i+1, i+3)
            bond_ind += 2
            i += 3
        # N2 case (note, TraPPE n2 has a ghostie!)
        elif ('N' in rev_dict.keys() and
              struc.positions[i][0] == rev_dict['Ghostie'] and
              struc.positions[i+1][0] == rev_dict['N'] and
              struc.positions[i+2][0] == rev_dict['N']):
            bond_type = 3
            btxt += '{} {} {} {} # N:ghostie\n'.format(bond_ind, bond_type, i+1, i+2)
            btxt += '{} {} {} {} # N:ghostie\n'.format(bond_ind+1, bond_type, i+1, i+3)
            bond_ind += 2
            i += 3
        # H2 case (our h2 model also has a ghostie!)
        elif ('H' in rev_dict.keys() and
              struc.positions[i][0] == rev_dict['Ghostie'] and
              struc.positions[i+1][0] == rev_dict['H'] and
              struc.positions[i+2][0] == rev_dict['H']):
            bond_type = 3
            btxt += '{} {} {} {} # H:ghostie\n'.format(bond_ind, bond_type, i+1, i+2)
            btxt += '{} {} {} {} # H:ghostie\n'.format(bond_ind+1, bond_type, i+1, i+3)
            bond_ind += 2
            i += 3
        else:
            raise Exception('bond type not implemented!')
    # angle info:
    atxt = '\nAngles\n\n'
    i = 0
    angle_ind = 1
    while i < len(struc.positions):
        # H2O case
        if (struc.positions[i][0] == 'O' and
            struc.positions[i+1][0] == 'H' and struc.positions[i+2][0] == 'H'):
            # this is the H20 case, with hydrogens following the oxygen
            angle_type = 1
            atxt += '{} {} {} {} {} # H:O:H\n'.format(angle_ind, angle_type, i+2, i+1, i+3)
            angle_ind += 1
            i += 3
        #CH4 case
        elif (struc.positions[i][0] == 'C' and
              struc.positions[i+1][0] == 'S' and struc.positions[i+2][0] == 'S' and
              struc.positions[i+3][0] == 'S' and struc.positions[i+4][0] == 'S'):
            # this is the CH4 case
            angle_type = 2
            atxt += '{} {} {} {} {} # H:C:H\n'.format(angle_ind, angle_type, i+2, i+1, i+3)
            atxt += '{} {} {} {} {} # H:C:H\n'.format(angle_ind+1, angle_type, i+2, i+1, i+4)
            atxt += '{} {} {} {} {} # H:C:H\n'.format(angle_ind+2, angle_type, i+2, i+1, i+5)
            atxt += '{} {} {} {} {} # H:C:H\n'.format(angle_ind+3, angle_type, i+3, i+1, i+4)
            atxt += '{} {} {} {} {} # H:C:H\n'.format(angle_ind+4, angle_type, i+3, i+1, i+5)
            atxt += '{} {} {} {} {} # H:C:H\n'.format(angle_ind+5, angle_type, i+4, i+1, i+5)
            angle_ind += 6
            i += 5
        # we're only buliding this to allow CO2 or N2 or H2
        # CO2 case
        elif ('C' in rev_dict.keys() and
              struc.positions[i][0] == rev_dict['C'] and
              struc.positions[i+1][0] == rev_dict['O'] and
              struc.positions[i+2][0] == rev_dict['O']):
            angle_type = 3
            atxt += '{} {} {} {} {} # O:C:O\n'.format(angle_ind, angle_type, i+2, i+1, i+3)
            angle_ind += 1
            i += 3
        # N2 case (note, TraPPE n2 has a ghostie!)
        elif ('N' in rev_dict.keys() and
              struc.positions[i][0] == rev_dict['Ghostie'] and
              struc.positions[i+1][0] == rev_dict['N'] and
              struc.positions[i+2][0] == rev_dict['N']):
            angle_type = 3
            atxt += '{} {} {} {} {} # N:Ghostie:N\n'.format(angle_ind, angle_type, i+2, i+1, i+3)
            angle_ind += 1
            i += 3
        # H2 case (our h2 model also has a ghostie!)
        elif ('H' in rev_dict.keys() and
              struc.positions[i][0] == rev_dict['Ghostie'] and
              struc.positions[i+1][0] == rev_dict['H'] and
              struc.positions[i+2][0] == rev_dict['H']):
            angle_type = 3
            atxt += '{} {} {} {} {} # H:Ghostie:H\n'.format(angle_ind, angle_type, i+2, i+1, i+3)
            angle_ind += 1
            i += 3
        else:
            raise Exception('angle type not implemented!')
    # now that we know how many bonds and angles we have, let's write the header:
    datatxt = 'Header of the LAMMPS data file \n\n'
    datatxt += '{} atoms \n'.format(struc.n_atoms)
    datatxt += '{} bonds \n'.format(bond_ind-1)
    datatxt += '{} angles \n'.format(angle_ind-1)
    datatxt += '0 dihedrals\n0 impropers\n'
    # get an extra atom type for different ch4 hydrogen
    datatxt += '{} atom types\n'.format(struc.n_species)
    datatxt += '3 bond types\n'
    datatxt += '3 angle types\n'
    datatxt += '0 dihedral types\n0 improper types\n\n'


    datatxt += '0.0 {}  xlo xhi\n'.format(struc.cell[0][0])
    datatxt += '0.0 {}  ylo yhi\n'.format(struc.cell[1][1])
    datatxt += '0.0 {}  zlo zhi\n'.format(struc.cell[2][2])

    datatxt += '\nMasses \n\n'
    for mass, kind in sorted([tuple(x.values()) for x in struc.species.values()],
                          key=lambda x: x[1]):
        datatxt += '{}  {}\n'.format(kind, mass)
    # Write atom positions in angstroms
    datatxt += '\nAtoms # atomic \n\n'
    i = 0
    mol_id = 1
    while i < len(struc.positions):
        # H2O case
        if (struc.positions[i][0] == 'O' and
            struc.positions[i+1][0] == 'H' and struc.positions[i+2][0] == 'H'):
            # H2O case
            datatxt += '{} {} {} -1.04844 {:1.5f} {:1.5f} {:1.5f} # O\n'.format(i + 1, mol_id, struc.species['O']['kind'], *struc.positions[i][1])
            datatxt += '{} {} {} 0.52422 {:1.5f} {:1.5f} {:1.5f} # H\n'.format(i + 2, mol_id, struc.species['H']['kind'], *struc.positions[i+1][1])
            datatxt += '{} {} {} 0.52422 {:1.5f} {:1.5f} {:1.5f} # H\n'.format(i + 3, mol_id, struc.species['H']['kind'], *struc.positions[i+2][1])
            i += 3
            mol_id += 1
        #CH4 case
        elif (struc.positions[i][0] == 'C' and
              struc.positions[i+1][0] == 'S' and struc.positions[i+2][0] == 'S' and
              struc.positions[i+3][0] == 'S' and struc.positions[i+4][0] == 'S'):
            # CH4 case
            datatxt += '{} {} {} -0.24000 {:1.5f} {:1.5f} {:1.5f} # C\n'.format(i + 1, mol_id, struc.species['C']['kind'], *struc.positions[i][1])
            datatxt += '{} {} {} 0.06000 {:1.5f} {:1.5f} {:1.5f} # H\n'.format(i + 2, mol_id, struc.species['S']['kind'], *struc.positions[i+1][1])
            datatxt += '{} {} {} 0.06000 {:1.5f} {:1.5f} {:1.5f} # H\n'.format(i + 3, mol_id, struc.species['S']['kind'], *struc.positions[i+2][1])
            datatxt += '{} {} {} 0.06000 {:1.5f} {:1.5f} {:1.5f} # H\n'.format(i + 4, mol_id, struc.species['S']['kind'], *struc.positions[i+3][1])
            datatxt += '{} {} {} 0.06000 {:1.5f} {:1.5f} {:1.5f} # H\n'.format(i + 5, mol_id, struc.species['S']['kind'], *struc.positions[i+4][1])
            i += 5
            mol_id += 1
        #CO2 case
        elif ('C' in rev_dict.keys() and
              struc.positions[i][0] == rev_dict['C'] and
              struc.positions[i+1][0] == rev_dict['O'] and
              struc.positions[i+2][0] == rev_dict['O']):
            datatxt += '{} {} {} 0.6512 {:1.5f} {:1.5f} {:1.5f} # C\n'.format(i + 1, mol_id, struc.species[rev_dict['C']]['kind'], *struc.positions[i][1])
            datatxt += '{} {} {} -0.32560 {:1.5f} {:1.5f} {:1.5f} # O\n'.format(i + 2, mol_id, struc.species[rev_dict['O']]['kind'], *struc.positions[i+1][1])
            datatxt += '{} {} {} -0.32560 {:1.5f} {:1.5f} {:1.5f} # O\n'.format(i + 3, mol_id, struc.species[rev_dict['O']]['kind'], *struc.positions[i+2][1])
            i += 3
            mol_id += 1
        #N2 case - with Ghostie!
        elif ('N' in rev_dict.keys() and
              struc.positions[i][0] == rev_dict['Ghostie'] and
              struc.positions[i+1][0] == rev_dict['N'] and
              struc.positions[i+2][0] == rev_dict['N']):
            datatxt += '{} {} {} 0.964 {:1.5f} {:1.5f} {:1.5f} # Ghostie\n'.format(i + 1, mol_id, struc.species[rev_dict['Ghostie']]['kind'], *struc.positions[i][1])
            datatxt += '{} {} {} -0.482 {:1.5f} {:1.5f} {:1.5f} # N\n'.format(i + 2, mol_id, struc.species[rev_dict['N']]['kind'], *struc.positions[i+1][1])
            datatxt += '{} {} {} -0.482 {:1.5f} {:1.5f} {:1.5f} # N\n'.format(i + 3, mol_id, struc.species[rev_dict['N']]['kind'], *struc.positions[i+2][1])
            i += 3
            mol_id += 1
        #H2 case - with Ghostie!
        elif ('H' in rev_dict.keys() and
              struc.positions[i][0] == rev_dict['Ghostie'] and
              struc.positions[i+1][0] == rev_dict['H'] and
              struc.positions[i+2][0] == rev_dict['H']):
            datatxt += '{} {} {} 0.9864 {:1.5f} {:1.5f} {:1.5f} # Ghostie\n'.format(i + 1, mol_id, struc.species[rev_dict['Ghostie']]['kind'], *struc.positions[i][1])
            datatxt += '{} {} {} -0.4932 {:1.5f} {:1.5f} {:1.5f} # H\n'.format(i + 2, mol_id, struc.species[rev_dict['H']]['kind'], *struc.positions[i+1][1])
            datatxt += '{} {} {} -0.4932 {:1.5f} {:1.5f} {:1.5f} # H\n'.format(i + 3, mol_id, struc.species[rev_dict['H']]['kind'], *struc.positions[i+2][1])
            i += 3
            mol_id += 1
        else:
            raise Exception('atom type not implemented!')

#     for index, site in enumerate(struc.positions):
#         datatxt += '{} {} {:1.5f} {:1.5f} {:1.5f} \n'.format(index + 1, struc.species[site[0]]['kind'], *site[1])
    # now add in bond and angle text
    datatxt += btxt
    datatxt += atxt
    datatxt += '\n'

    datafile = os.path.join(runpath.path, 'lammps.data')
    write_file(datafile, datatxt)
    return File(path=datafile)

"""
Below, we have the force field data for co2, n2, and h2. explicit inter-atom-type
parameters are given for the interactions between co2 and h2o based off the parameters
used by tung et al. 2011

"""
co2_ff_data = """
pair_coeff 5 5 0.559 2.757 # C-C from CO2
pair_coeff 6 6 0.1600 3.033 # O-O from CO2
pair_coeff 5 6 0.09512 2.892 # C-O from CO2
pair_coeff 3 5 0.10612 2.955 #c from co2 - o from h2o
pair_coeff 2 5 0.04995 2.400 #c from co2 - h from h2o
pair_coeff 3 6 0.1790 3.034 #o from h2o - o from co2
pair_coeff 2 6 0.0851 2.480 # h from h2o - o from co2

bond_coeff 3 0.0 1.149 # co2
angle_coeff 3 295.411 180.0 # co2
"""

"""
We still assume geometric mixing with this trappe model; also, keep it as a rigid body
(tradeoffs here? recreates phase diagram well, seems worthwhile for when dealing with
gaseous and potential clathrate phase)

we set really high spring constants for bond and angle to ensure minimial distortion
during the minimization procedure, before they're made rigid

"""
n2_ff_data = """
pair_coeff 5 5 0.0 0.0 # Ghostie-Ghostie from N2
pair_coeff 6 6 0.07153 3.310 # N-N from N2

bond_coeff 3 10000 0.55
angle_coeff 3 10000 180.0 # approximating rigidity with huge spring constant
"""

"""
same caveats apply here as for n2 - rigid model by alavi et al.

"""
h2_ff_data = """
pair_coeff 5 5 0.06816 3.038 # Ghostie-Ghostie from H2
pair_coeff 6 6 0.0 0.0 # H-H from H2

bond_coeff 3 10000 0.3707
angle_coeff 3 10000 180.0 # approximating rigidity with huge spring constant
"""

"""
creates the lammps input file based off the relevant template in the templates folder

runpath: self-explanatory
intemplate: path for the template for the input file
inparam: extra parameters to substitute into the template

"""
def write_lammps_input(runpath, intemplate, inparam):
    subst = {'RUNPATH': runpath.path}

    with open(intemplate) as f:
        intemplate = f.read()

    inptxt = Template(intemplate).safe_substitute({**subst, **inparam})
    infile = TextFile(path=os.path.join(runpath.path, 'lammps.in'), text=inptxt)
    infile.write()
    return infile


"""
THIS WILL BE THE COMMAND THAT RUNS LAMMPS
EDIT IT AS NECESSARY TO TAKE ADVANTAGE OF CERTAIN OPTIMIZATIONS/MULTITHREADING
"""
#LAMMPS_RUN = "mpirun -np 4 {} -in {} -log {} > {}"
# THIS IS THE VERSION THAT SHOULD BE RUN ON GOOGLE CLOUD
LAMMPS_RUN = "mpirun -np 8 {} -sf omp -pk omp 1 -in {} -log {} > {}"


"""
Creates a structure file for the particular gas type

"""
def run_minimization(gastype, trial, run_cmd=LAMMPS_RUN):
    print('Begin clathrate-{} minimization'.format(gastype))
    lammps_code = ExternalCode(path=os.environ['LAMMPS_COMMAND'])

    #### RUN THE MINIMIZATION ---------------------------------------
    runpath = Dir(path=os.path.join('production', gastype,
                                    'trial_{}'.format(trial), 'minim'))
    prepare_dir(runpath.path)
    # make structure and data for it
    clath = create_clathrate(np.array([4, 2, 2]))


    if gastype == 'co2':
        clath = fill_gas(clath, 35.32, 'gas_coords/liquidco2.pdb', needs_ghostie=False)
    elif gastype == 'n2':
        clath = fill_gas(clath, 35.32, 'gas_coords/nitrogen.pdb', needs_ghostie=True)
    elif gastype == 'h2':
        clath = fill_gas(clath, 35.32, 'gas_coords/hydrogen.pdb', needs_ghostie=True)
    else:
        raise Exception('gas type not implemented!')

    struc = Struc(ase2struc(clath))
    datafile = write_lammps_data(struc, runpath)

    intemplate = ('templates/minim_template_liquid.txt' if gastype == 'co2'
                  else 'templates/minim_template_gas.txt')

    inparams = {'DATAINPUT': datafile.path, 'FFINPUT': globals()['{}_ff_data'.format(gastype)]}
    infile = write_lammps_input(runpath=runpath,
                                intemplate=intemplate, inparam=inparams)

    logfile = File(path=os.path.join(runpath.path, 'minim.log'))
    outfile = File(path=os.path.join(runpath.path, 'minim.out'))

    lammps_command = run_cmd.format(lammps_code.path, infile.path,
                                       logfile.path, outfile.path)
    run_command(lammps_command)
    print('End clathrate-{} minimization'.format(gastype))
    return outfile

# runs a quick, 20ps nvt simulation to relax stress, then heats up the system
# at 0.5K/ps and deposits restarts at each of the temps
def run_nvt(gastype, trial, run_cmd=LAMMPS_RUN):
    print('Begin clathrate-{} equilibration'.format(gastype))
    lammps_code = ExternalCode(path=os.environ['LAMMPS_COMMAND'])

    #### RUN THE NVT ---------------------------------------
    runpath = Dir(path=os.path.join('production', gastype,
                                    'trial_{}'.format(trial), 'nvt'))
    prepare_dir(runpath.path)

    equil_restart = Dir(path=os.path.join('production', gastype,
                                    'trial_{}'.format(trial), 'minim', 'restart_min.equil'))
    inparams = {'RESTARTFILE': equil_restart.path, 'FFINPUT': globals()['{}_ff_data'.format(gastype)],
                'SEED': int(np.abs(np.cos(trial)*10000))}
    templatefile = 'templates/nvt_template.txt' if gastype == 'co2' else 'templates/nvt_template_ghostie.txt'
    infile = write_lammps_input(runpath=runpath,
                                intemplate=templatefile, inparam=inparams)
    logfile = File(path=os.path.join(runpath.path, 'nvt.log'))
    outfile = File(path=os.path.join(runpath.path, 'nvt.out'))

    lammps_command = run_cmd.format(lammps_code.path, infile.path,
                                       logfile.path, outfile.path)
    run_command(lammps_command)
    print('End clathrate-{} equilibration'.format(gastype))
    return outfile

# production run!
def run_production(gastype, trial, temp, run_cmd=LAMMPS_RUN):
    print('Begin clathrate-{} production'.format(gastype))
    lammps_code = ExternalCode(path=os.environ['LAMMPS_COMMAND'])

    runpath = Dir(path=os.path.join('production', gastype,
                                    'trial_{}'.format(trial), '{}K'.format(temp)))
    prepare_dir(runpath.path)
    restarts_dir = Dir(path=os.path.join('production', gastype,
                                    'trial_{}'.format(trial), '{}K'.format(temp),
                                    'restarts'))
    prepare_dir(restarts_dir.path)

    # get the restart from the nvt/npt up to temp sims
    nvt_restart = Dir(path=os.path.join('production', gastype,
                                    'trial_{}'.format(trial), 'nvt',
                                    'restart_npt_{}.equil'.format(temp)))
    inparams = {'RESTARTFILE': nvt_restart.path, 'FFINPUT': globals()['{}_ff_data'.format(gastype)],
                'TEMP': temp}
    templatefile = 'templates/production_template.txt' if gastype == 'co2' else 'templates/production_template_ghostie.txt'
    infile = write_lammps_input(runpath=runpath,
                                intemplate=templatefile, inparam=inparams)
    logfile = File(path=os.path.join(runpath.path, 'npt_{}.log'.format(temp)))
    outfile = File(path=os.path.join(runpath.path, 'npt_{}.out'.format(temp)))

    lammps_command = run_cmd.format(lammps_code.path, infile.path,
                                       logfile.path, outfile.path)
    run_command(lammps_command)
    print('End clathrate-{} production'.format(gastype))
    return outfile

# just in case a restart is necessary
def run_restart(gastype, trial, temp, restarttime, run_cmd=LAMMPS_RUN):
    print('Begin clathrate-{} production restart'.format(gastype))
    lammps_code = ExternalCode(path=os.environ['LAMMPS_COMMAND'])

    runpath = Dir(path=os.path.join('production', gastype,
                                    'trial_{}'.format(trial), '{}K'.format(temp)))
    prepare_dir(runpath.path)
    restarts_dir = Dir(path=os.path.join('production', gastype,
                                    'trial_{}'.format(trial), '{}K'.format(temp),
                                    'restarts'))
    prepare_dir(restarts_dir.path)

    # get the restart from the nvt/npt up to temp sims
    npt_restart = Dir(path=os.path.join(restarts_dir.path,
                                    'restart.{}.prod'.format(restarttime)))
    inparams = {'RESTARTFILE': npt_restart.path, 'FFINPUT': globals()['{}_ff_data'.format(gastype)],
                'TEMP': temp}
    templatefile = 'templates/npt_restart_template.txt' if gastype == 'co2' else 'templates/npt_restart_template_ghostie.txt'
    infile = write_lammps_input(runpath=runpath,
                                intemplate=templatefile, inparam=inparams)
    logfile = File(path=os.path.join(runpath.path, 'npt_restart_{}.log'.format(temp)))
    outfile = File(path=os.path.join(runpath.path, 'npt_restart_{}.out'.format(temp)))

    lammps_command = run_cmd.format(lammps_code.path, infile.path,
                                       logfile.path, outfile.path)
    run_command(lammps_command)
    print('End clathrate-{} restart production'.format(gastype))
    return outfile

# main function; let's actually run some of these simulations, eh?
def main():
    # this runs the brief energy minimization for the clathrate w/ liquid co2
    # second input to the file is trial number; we want to run in triplicate, so
    # keep track of which ones you've already done
    # MINIMIZATION ONLY NEEDS TO BE DONE ONCE per gas and trial; we heat the
    # same structure up to 280, 285, and 290K
    out1 = run_nvt('n2', 1)

    # this runs a 20ps nvt simulation (1ps with a 1fs stepsize, then 19ps with
    # a 2.5fs stepsize) to "reduce stress" or whatever; then switches to NPT
    # at 6 MPa, and heats at 0.5K/ps up to 280, 285, and 290, dropping a restart
    # file at each of those
    # this only needs to be run ONCE per gas and trial; it generates data for
    # all three temperatures
    # out2 = run_nvt('co2', 1)

    # this is a production run at a particular temperature. As opposed to the
    # prior two functions, you actually need to run this once for each temperature.
    # out3 = run_production('co2', 1, 280)



if __name__ == '__main__':
    main()


