def generate_mopac_input_file(mol, filename, top_keywords = None, bottom_keywords = None, run = False, check_for_calc = True):
    if not filename.endswith(".mop"):
        filename = filename + ".mop"
    with open(filename, "w") as f:
        # writing top part
        if not top_keywords is None:
            top_string = ""
            for word in top_keywords:
                top_string += " " + word
            top_string += "\r\n"
            f.write(top_string)
            f.write("title\r\n")
            f.write("\n")
        # writing coords and atoms
        for atom, coord in zip(mol.atoms, mol.coords):
            s = atom
            for c in coord:
                s += " " + str(c) + " 1"
            s += "\r\n"
            f.write(s)
        # writing bottom part
        if not bottom_keywords is None:
            bottom_string = ""
            for word in bottom_keywords:
                bottom_string += " " + word
            bottom_string += "\r\n"
            f.write("\n")
            f.write(bottom_string)
    if run is True:
        if not check_for_calc:
            os.system(f'/opt/mopac/MOPAC2016.exe {filename}')
        elif not os.path.isfile(filename[:-3] + "out"):
            os.system(f'/opt/mopac/MOPAC2016.exe {filename}')

def read_mopac_file(filename):
    atoms = []; coords = []; OxyEnergy = None; ElecAffinity = None; ES = None; ET = None; TotalE = None; FreqVec = []; RateVec = []
    with open(filename, "r") as f:
        CoordsBlock = False; ESandETBlock = False; OxyStateBlock = False; VibBlock = False; VibVec = []
        for line in f.readlines():
            wordsvec = re.split(r" |\t|\n", line)
            wordsvec = list(filter(lambda a: a != '', wordsvec))
            if len(wordsvec) == 0:
                continue
            if "".join(wordsvec) == "CARTESIANCOORDINATES":
                CoordsBlock = True
                coords = []
                continue
            if CoordsBlock and not len(wordsvec) == 5:
                CoordsBlock = False
                continue
            if CoordsBlock and not wordsvec[0] == "NO.":
                atoms.append(wordsvec[1])
                coords.append([float(x) for x in wordsvec[2:]])
                continue
            if line == "  STATE       ENERGY (EV)        Q.N.  SPIN   SYMMETRY              POLARIZATION\n":
                ESandETBlock = True
                continue
            if ESandETBlock and (len(wordsvec) == 6 or len(wordsvec) == 9):
                if wordsvec[4] == "SINGLET":
                    ES = float(wordsvec[2])
                    continue
                elif wordsvec[4] == "TRIPLET":
                    ET = float(wordsvec[2])
                    continue
            if not ES == None and not ET == None and ESandETBlock:
                ESandETBlock = False
                continue
            try:
                if wordsvec[0] + wordsvec[1] == "IONIZATIONPOTENTIAL":
                    OxyEnergy = float(wordsvec[3])
                    continue
                elif wordsvec[0] + wordsvec[1] == "HOMOLUMO":
                    ElecAffinity = float(wordsvec[-1])
                    continue
                elif wordsvec[0] + wordsvec[1] == "ALPHASOMO":
                    ElecAffinity = float(wordsvec[-1])
                    continue
                elif wordsvec[0] + wordsvec[1] == "BETASOMO":
                    if float(wordsvec[-1]) < ElecAffinity:
                        ElecAffinity = float(wordsvec[-1])
                    continue
                elif wordsvec[0] + wordsvec[1] == "TOTALENERGY":
                    TotalE = float(wordsvec[3])
                    continue
            except IndexError:
                continue
            if line == "          DESCRIPTION OF VIBRATIONS\n":
                VibBlock = True
                continue
            if VibBlock and line == "           FORCE CONSTANT IN CARTESIAN COORDINATES (Millidynes/A)\n":
                VibBlock = False
                continue
            if VibBlock:
                if wordsvec[0] == "FREQUENCY":
                    VibVec = []
                    VibVec.append(float(wordsvec[1]))
                elif wordsvec[0] + wordsvec[1] == "EFFECTIVEMASS":
                    try:
                        VibVec.append(float(wordsvec[2]))
                    except:
                        continue
                elif wordsvec[0] + wordsvec[1] == "FORCECONSTANT":
                    VibVec.append(float(wordsvec[2]))
                if len(VibVec) == 3 and VibVec[2] > 0 and not VibVec[0] in FreqVec:
                    FreqVec.append(VibVec[0])
                    RateVec.append(1 / np.power((VibVec[1] * VibVec[2]), 0.25))
    return {
        "atoms": atoms,
        "coords": coords,
        "ES": ES,
        "ET": ET,
        "OxyEnergy": OxyEnergy,
        "ElecAffinity": ElecAffinity,
        "TotalE": TotalE,
        "FreqVec": FreqVec,
        "RateVec": RateVec
    }

def get_mols_from_CIF(path):
    my_structure = Structure.from_file(path)
    # begin ordering algorithm
    disordered_elements = []; disordered_sites = []; tot_nums = []
    for site in my_structure.sites:
        if site.is_ordered:
            continue
        else:
            element = str(site.species.elements[0])
            if element in disordered_elements:
                index = disordered_elements.index(element)
                disordered_sites[index].append(site)
                tot_nums[index] = tot_nums[index] + site.species.num_atoms
            else:
                disordered_sites.append([site])
                disordered_elements.append(element)
                tot_nums.append(site.species.num_atoms)
    for num, sites in zip(tot_nums, disordered_sites):
        pop_indicis = []
        for i in sorted(range(len(sites)), key=lambda i: sites[i].species.num_atoms)[int(num):]:
            pop_indicis.append(my_structure.sites.index(sites[i]))
        keep_indicis = []
        for i in sorted(range(len(sites)), key=lambda i: sites[i].species.num_atoms)[:int(num)]:
            keep_indicis.append(my_structure.sites.index(sites[i]))
        for i in keep_indicis:
            element = str(my_structure.sites[i].species.elements[0])
            my_structure.replace(i, element)
        my_structure.remove_sites(pop_indicis)
    # molecule finding
    sg = StructureGraph.with_local_env_strategy(my_structure, env_strategy())
    my_molecules = sg.get_subgraphs_as_molecules()
    # convert to Mol objects
    my_Mols = []
    for molecule in my_molecules:
        coords = []; atoms = []
        for site in molecule.sites:
            element = str(site.species.elements[0])
            atoms.append(element); coords.append(site.coords)
        mol = Mol()
        mol.atoms = atoms; mol.coords = coords; mol.EstimateBondmap(Set=True)
        my_Mols.append(mol)
    return my_Mols