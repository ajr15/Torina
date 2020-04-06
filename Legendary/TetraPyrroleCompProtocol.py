from pymatgen import Element
import os
import numpy as np
from scipy import integrate
import pandas as pd
from ..Base import MolUtils
from abc import ABC, abstractclassmethod
from rdkit import Chem
from Base import PropCalc

class TetraPyrrolePropCalc (PropCalc):
    def __init__(self, path, check_for_previous_calc = True, geo_opt = False, mopac_calc_method = "PM7"):
        self.path = path
        self.check_for_previous_calc = check_for_previous_calc
        self.geo_opt = geo_opt
        self.mopac_calc_method = mopac_calc_method

    def PreRunExternalSoftware(self, mol: MolUtils.Tetrapyrrole, mol_name, GlobalDict):
        ci_filename = os.path.join(self.path, mol_name + "_CI_CALC.mop")
        if not self.check_for_previous_calc:
            top_keywords = [self.mopac_calc_method]
            if self.geo_opt is True:
                top_keywords = top_keywords + ["NOSYM"]
            else:
                top_keywords = top_keywords + ["1SCF"]
            
            mol.generate_mopac_input_file(ci_filename, top_keywords, ["CIS", "C.I.=2", "MECI", "1SCF", "OLDGEO", "GEO-OK"])
            os.system(f'/opt/mopac/MOPAC2016.exe {ci_filename}')

        elif not os.path.isfile(ci_filename[:-4] + ".out"):
            top_keywords = [self.mopac_calc_method]
            if self.geo_opt is True:
                top_keywords = top_keywords + ["NOSYM"]
            else:
                top_keywords = top_keywords + ["1SCF"]
            
            mol.generate_mopac_input_file(ci_filename, top_keywords, ["CIS", "C.I.=2", "MECI", "1SCF", "OLDGEO", "GEO-OK"])
            os.system(f'/opt/mopac/MOPAC2016.exe {ci_filename}')
        
        force_filename = os.path.join(path, mol_name + "_FORCE_CALC.mop")
        if not self.check_for_previous_calc:
            mol.generate_mopac_input_file(force_filename, ["FORCE", "LET"])
            os.system(f'/opt/mopac/MOPAC2016.exe {force_filename}')
        elif not os.path.isfile(force_filename[:-4] + ".out"):
            mol.generate_mopac_input_file(force_filename, ["FORCE", "LET"])
            os.system(f'/opt/mopac/MOPAC2016.exe {force_filename}')

    def RateDict(self, mol: MolUtils.Tetrapyrrole, mol_name, GlobalDict) -> dict:
        ci_filename = os.path.join(self.path, mol_name + "_CI_CALC.out")
        force_filename = os.path.join(self.path, mol_name + "_FORCE_CALC.out")
        ci_dict = MolUtils.read_mopac_file(ci_filename)
        force_dict = MolUtils.read_mopac_file(force_filename)

        CenterNs, betas, mesos = mol.get_tetrapyrrole_atoms()
        RingAtoms = list(set(betas + mesos + CenterNs))

        steric_factor = self.CalcStericFactor(ci_dict["atoms"], ci_dict["coords"], 5.007, mesos)
        
        O2OxyE = 10.042875
        O2ES = 0.974165; O2ET = 3.606514
        CTSOargs = (5.007 + 1.976, ci_dict["OxyEnergy"] - ci_dict["ES"] - O2OxyE + O2ES, steric_factor, True)
        CTTOargs = (5.007 + 1.976, ci_dict["OxyEnergy"] - ci_dict["ET"] - O2OxyE + O2ET, steric_factor, True)
        ETargs = (5.007 + 1.976, O2ES - ci_dict["ET"], steric_factor, True)

        knames =      ["kISC",   "knrS",   "knrT",      "krS",    "krT",    "kCTSO",  "kCTTO",  "kET"]
        knorm =       [ 0.25,     1.333,    0.000428571, 1,        0.001,    0.1,      0.001,    0.1]
        NakedValues = [ 1083.727, 795.287,  598.510,     0.6535,   1.0162,   215.265,  119.950,  110.973]
        NormDict = dict([(name, norm / naked) for name, norm, naked in zip(knames, knorm, NakedValues)])

        return {
            "kISC": self.CalcKISC(ci_dict["atoms"], ci_dict["coords"], mesos, RingAtoms) * NormDict["kISC"],
            "krS": self.CalcKr(ci_dict["ES"]) * NormDict["krS"],
            "krT": self.CalcKr(ci_dict["ET"]) * NormDict["krT"],
            "knrS": self.CalcKnr(ci_dict["ES"], force_dict["FreqVec"], force_dict["RateVec"]) * NormDict["knrS"],
            "knrT": self.CalcKnr(ci_dict["ET"], force_dict["FreqVec"], force_dict["RateVec"]) * NormDict["knrT"],
            "kCTSO": self.CalcKMarcus(*CTSOargs) * NormDict["kCTSO"],
            "kCTTO": self.CalcKMarcus(*CTTOargs) * NormDict["kCTTO"],
            "kET": self.CalcKMarcus(*ETargs) * NormDict["kET"]
        }

    def PropDict(self, rate_dict):
        STerm = rate_dict["kCTSO"] + rate_dict["kISC"] + rate_dict["knrS"] + rate_dict["krS"]
        TTerm = rate_dict["krT"] + rate_dict["knrT"] + rate_dict["kCTTO"] + rate_dict["kET"]
        SuperOxideQY = (rate_dict["kCTSO"] / STerm) + (rate_dict["kISC"] * rate_dict["kCTTO"]) / (STerm * TTerm)
        SingletOxyQY = (rate_dict["kISC"] * rate_dict["kET"]) / (STerm + TTerm)
        
        return {
            "SuperOxideQY": SuperOxideQY,
            "SingletOxyQY": SingletOxyQY
        }

    def CalcKISC(self, atoms, coords, mesos, RingAtoms):
        center_coords = np.array([
            np.mean([coords[meso][0] for meso in mesos]),
            np.mean([coords[meso][1] for meso in mesos]),
            np.mean([coords[meso][2] for meso in mesos])
        ])
    
        dist_vec = [] # vector of distances of atoms from the center of the ring
        for coord in coords:
            dist_vec.append(np.sqrt(np.sum(np.square(np.array(coord) - center_coords))))
        ring_radius = np.mean([dist_vec[atom] for atom in RingAtoms])

        # finding normal vector to the 4N plane
        v1 = np.array(coords[mesos[0]]) - center_coords
        v2 = np.array(coords[mesos[1]]) - center_coords
        for coord in [coords[mesos[i]] for i in range(2, len(mesos))]:
            if round(np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) == 0:
                break
            else:
                v2 = np.array(coord) - center_coords

        normal_vec = np.array([
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        ])

        # calculating effective electric field
        def i1(m):
            return integrate.quad(lambda x: 1  / (np.power(1 - m * np.cos(x), 1.5)), 0, np.pi)[0]
        
        def i2(m):
            return integrate.quad(lambda x: np.cos(x) / (np.power(1 - m * np.cos(x), 1.5)), 0, np.pi)[0]

        field_vec = []
        for dist, coord in zip(dist_vec, coords):
            sin = np.sum(normal_vec * (np.array(coord) - center_coords)) / (dist * np.linalg.norm(normal_vec))
            cos = np.sqrt(1 - np.square(sin))
            rel_dist = dist / ring_radius

            m = 2 * rel_dist / (np.square(rel_dist) + 1) * sin
            field = 24 / np.square(ring_radius) * np.sqrt((rel_dist * sin / np.power(1 + np.square(rel_dist), 1.5) * i1(m) - i2(m) / np.power(1 + np.square(rel_dist),1.5))**2 + (rel_dist * cos / np.power(1 + np.square(rel_dist),1.5) * i1(m))**2)
            field_vec.append(field)

        kISC = 0
        for atom, field in zip(atoms, field_vec):
            Z = Element(atom).Z
            r = Element(atom).atomic_radius
            kISC = kISC + Z * (Z + np.square(r) * field)

        return kISC

    def CalcKr(self, E):
        return 1 / np.sqrt(E)

    def CalcKnr(self, dE, FreqVec, RateVec):
        knr = 0
        for freq, rate in zip(FreqVec, RateVec):
            if freq * 0.00012 <= dE and not freq == 0:
                ni = int(np.floor(2 * dE / (freq * 0.00012) - 1))
                knr = knr + 1 / (np.square(rate) / (np.sum([1 / i for i in range(1, ni + 1)])))
        return knr

    def CalcKMarcus(self, r, EnergyDiff, steric_factor, ToLogScale = False, Const = 230):
        ExpTerm = (EnergyDiff * 1.602 * 10 + r**2)**2 / (r**2 * 1.38 * 298) * 1000
        if ExpTerm < Const:
            k = steric_factor * np.exp(-ExpTerm)
        else:
            k = 0

        if ToLogScale is True:
            if np.isinf(np.log(k)):
                return 0
            else:
                if np.log(k) + Const < 0:
                    return 0
                else:
                    return np.log(k) + Const
        else:
            return k

    def CalcStericFactor(self, atoms, coords, reff, mesos):
        center_coords = np.array([
            np.mean([coords[meso][0] for meso in mesos]),
            np.mean([coords[meso][1] for meso in mesos]),
            np.mean([coords[meso][2] for meso in mesos])
        ])
    
        props = []
        for coord, atom in zip(coords, atoms):
            corr_coord = np.array(coord) - center_coords
            dist = np.linalg.norm(corr_coord)
            atomic_radius = Element(atom).atomic_radius
            if atomic_radius + reff > dist and not dist + atomic_radius < reff:
                alpha = np.arccos((-atomic_radius**2 + reff**2 + dist**2) / (2 * reff * dist))
                phi = corr_coord[2] / dist
                theta = corr_coord[0] / np.linalg.norm(corr_coord[:-1])
                props.append([alpha, phi, theta])

        reduced_surfaces = []; overlaps = []
        for i in range(len(props)):
            alpha1, phi1, theta1 = props[i]
            reduced_surface = alpha1**2
            for j in range(i + 1, len(props)):
                alpha2, phi2, theta2 = props[j]
                dphi = alpha1 + alpha2 - np.abs(phi1 - phi2)
                dtheta = alpha1 + alpha2 - np.abs(theta1 - theta2)
                if dphi > 0 and dtheta > 0:
                    reduced_surface = reduced_surface - dphi * dtheta
                    overlaps.append(dphi * dtheta)
        return 1 - (np.sum(reduced_surfaces) + np.sum(overlaps)) / (4 * np.pi**2)

class CNScanCalc (TetraPyrrolePropCalc):

    def __init__(self, path, NAKED_path, pair_data_file, s1_data_file, check_for_previous_calc = True, geo_opt = False, mopac_calc_method = "PM7"):
        self.path = path
        self.check_for_previous_calc = check_for_previous_calc
        self.geo_opt = geo_opt
        self.mopac_calc_method = mopac_calc_method
        self.NAKED_path = NAKED_path
        self.pair_data_file = pair_data_file
        self.s1_data_file = s1_data_file

    def GlobalPreCalc(self) -> dict:
        ci_filename = os.path.join(self.NAKED_path, "NAKED_CI_CALC.out")
        force_filename = os.path.join(self.NAKED_path, "NAKED_FORCE_CALC.out")
        ion_ci_filename = os.path.join(self.NAKED_path, "NAKED_ION_CI_CALC.out")
        ci_dict = MolUtils.read_mopac_file(ci_filename)
        ion_ci_dict = MolUtils.read_mopac_file(ion_ci_filename)
        force_dict = MolUtils.read_mopac_file(force_filename)
        mol = MolUtils.Tetrapyrrole()
        mol.from_file(os.path.join(self.NAKED_path, "NAKED.mol"))

        CenterNs, betas, mesos = mol.get_tetrapyrrole_atoms()
        RingAtoms = list(set(betas + mesos + CenterNs))

        steric_factor = self.CalcStericFactor(ci_dict["atoms"], ci_dict["coords"], 5.007, mesos)

        R1OxyE = 15.797749
        R1ES = 4.836235; R1ET = 4.751636
        R2OxyE = 12.708632

        return {
            "kISC": 0.25 / self.CalcKISC(ci_dict["atoms"], ci_dict["coords"], mesos, RingAtoms),
            "krS": 1.333 / self.CalcKr(ci_dict["ES"]),
            "krT": 0.000428571 / self.CalcKr(ci_dict["ET"]),
            "knrS": 1 / self.CalcKnr(ci_dict["ES"], force_dict["FreqVec"], force_dict["RateVec"]),
            "knrT": 0.001 / self.CalcKnr(ci_dict["ET"], force_dict["FreqVec"], force_dict["RateVec"]),
            "kCTSR1": 0.1 / self.CalcKMarcus(5.007 + 1.9, ci_dict["OxyEnergy"] - ci_dict["ES"] - R1OxyE + R1ES, steric_factor, True),
            "kCTTR1": 0.001 / self.CalcKMarcus(5.007 + 1.9, ci_dict["OxyEnergy"] - ci_dict["ET"] - R1OxyE + R1ET, steric_factor, True),
            "kCTR2": 1 / self.CalcKMarcus(5.007, R2OxyE - 1.3 - ion_ci_dict["OxyEnergy"], steric_factor, True),
            "kDecompCl": 1 / self.CalcKMarcus(5.007 + 3, 3.853099 - ion_ci_dict["OxyEnergy"], steric_factor, True, 700),
            "kDecompH2O": 1 / self.CalcKMarcus(5.007 + 2, 12.116089 - ion_ci_dict["OxyEnergy"], steric_factor, True)
        }
   
    def PreRunExternalSoftware(self, mol: MolUtils.Tetrapyrrole, mol_name, GlobalDict):
        mol.add_hydrogens()
        mol.MMGeoOpt()

        ci_filename = os.path.join(self.path, mol_name + "_CI_CALC.mop")
        if not self.check_for_previous_calc:
            top_keywords = [self.mopac_calc_method]
            if self.geo_opt is True:
                top_keywords = top_keywords + ["NOSYM"]
            else:
                top_keywords = top_keywords + ["1SCF"]
            
            mol.generate_mopac_input_file(ci_filename, top_keywords, ["CIS", "C.I.=2", "MECI", "1SCF", "OLDGEO", "GEO-OK"])
            os.system(f'/opt/mopac/MOPAC2016.exe {ci_filename}')

        elif not os.path.isfile(ci_filename[:-4] + ".out"):
            top_keywords = [self.mopac_calc_method]
            if self.geo_opt is True:
                top_keywords = top_keywords + ["NOSYM"]
            else:
                top_keywords = top_keywords + ["1SCF"]
            
            mol.generate_mopac_input_file(ci_filename, top_keywords, ["CIS", "C.I.=2", "MECI", "1SCF", "OLDGEO", "GEO-OK"])
            os.system(f'/opt/mopac/MOPAC2016.exe {ci_filename}')

        ion_ci_filename = os.path.join(self.path, mol_name + "_ION_CI_CALC.mop")
        if not self.check_for_previous_calc:
            top_keywords = [self.mopac_calc_method, "CHARGE=1"]
            if self.geo_opt is True:
                top_keywords = top_keywords + ["NOSYM"]
            else:
                top_keywords = top_keywords + ["1SCF"]
            
            mol.generate_mopac_input_file(ion_ci_filename, top_keywords, ["CIS", "C.I.=2", "MECI", "1SCF", "OLDGEO", "GEO-OK"])
            os.system(f'/opt/mopac/MOPAC2016.exe {ion_ci_filename}')

        elif not os.path.isfile(ion_ci_filename[:-4] + ".out"):
            top_keywords = [self.mopac_calc_method, "CHARGE=1"]
            if self.geo_opt is True:
                top_keywords = top_keywords + ["NOSYM"]
            else:
                top_keywords = top_keywords + ["1SCF"]
            
            mol.generate_mopac_input_file(ion_ci_filename, top_keywords, ["CIS", "C.I.=2", "MECI", "1SCF", "OLDGEO", "GEO-OK"])
            os.system(f'/opt/mopac/MOPAC2016.exe {ion_ci_filename}')
        
        force_filename = os.path.join(self.path, mol_name + "_FORCE_CALC.mop")
        if not self.check_for_previous_calc:
            mol.generate_mopac_input_file(force_filename, ["FORCE", "LET"])
            os.system(f'/opt/mopac/MOPAC2016.exe {force_filename}')
        elif not os.path.isfile(force_filename[:-4] + ".out"):
            mol.generate_mopac_input_file(force_filename, ["FORCE", "LET"])
            os.system(f'/opt/mopac/MOPAC2016.exe {force_filename}')

    def RateDict(self, mol: MolUtils.Tetrapyrrole, mol_name, GlobalDict) -> dict:
        ci_filename = os.path.join(self.path, mol_name + "_CI_CALC.out")
        force_filename = os.path.join(self.path, mol_name + "_FORCE_CALC.out")
        ion_ci_filename = os.path.join(self.path, mol_name + "_ION_CI_CALC.out")
        ci_dict = MolUtils.read_mopac_file(ci_filename)
        ion_ci_dict = MolUtils.read_mopac_file(ion_ci_filename)
        force_dict = MolUtils.read_mopac_file(force_filename)
        CenterNs, betas, mesos = mol.get_tetrapyrrole_atoms()
        RingAtoms = list(set(betas + mesos + CenterNs))

        steric_factor = self.CalcStericFactor(ci_dict["atoms"], ci_dict["coords"], 5.007, mesos)

        RateDict = {
            "kISC": self.CalcKISC(ci_dict["atoms"], ci_dict["coords"], mesos, RingAtoms) * GlobalDict["kISC"],
            "krS": self.CalcKr(ci_dict["ES"]) * GlobalDict["krS"],
            "krT": self.CalcKr(ci_dict["ET"]) * GlobalDict["krT"],
            "knrS": self.CalcKnr(ci_dict["ES"], force_dict["FreqVec"], force_dict["RateVec"]) * GlobalDict["knrS"],
            "knrT": self.CalcKnr(ci_dict["ET"], force_dict["FreqVec"], force_dict["RateVec"]) * GlobalDict["knrT"],
            "kDecompCl": self.CalcKMarcus(5.007 + 3, 3.853099 - ion_ci_dict["OxyEnergy"], steric_factor, True, 700) * GlobalDict["kDecompCl"] * 1,
            "kDecompH2O": self.CalcKMarcus(5.007 + 2, 12.116089 - ion_ci_dict["OxyEnergy"], steric_factor, True) * GlobalDict["kDecompH2O"] * 1.5
        }
        
        s1_data = pd.read_csv(self.s1_data_file, sep=' ')
        s1_data = s1_data.set_index("#")
        pair_data = pd.read_csv(self.pair_data_file, sep=' ')
        pair_data = pair_data.set_index("#")

        RateDict["s1_vals"] = s1_data.index
        RateDict["s2_num"] = len(pair_data.columns)

        err_vec = []
        for s1 in s1_data.index:
            s1OxyE = s1_data.loc[s1, "OxyE"]
            s1ES = s1_data.loc[s1, "ES"]
            s1ET = s1_data.loc[s1, "ET"]
            RateDict[f"kCTS{s1}"] = self.CalcKMarcus(5.007 + 1.9, ci_dict["OxyEnergy"] - ci_dict["ES"] - s1OxyE + s1ES, steric_factor, True) * GlobalDict["kCTSR1"] * 25
            RateDict[f"kCTT{s1}"] = self.CalcKMarcus(5.007 + 1.9, ci_dict["OxyEnergy"] - ci_dict["ES"] - s1OxyE + s1ET, steric_factor, True) * GlobalDict["kCTSR1"] * 25
            for s2 in pair_data.columns:
                if pair_data.loc[s1, s2] == "None":
                    err_vec.append([s1, int(s2)])
                    continue
                s2OxyE = float(pair_data.loc[s1, s2])
                RateDict[f"kCT{s1}-{s2}"] = self.CalcKMarcus(5.007, 0.9 * s2OxyE - ion_ci_dict["OxyEnergy"], steric_factor, True) * GlobalDict["kCTR2"]
        RateDict["error_pairs"] = err_vec
        return RateDict

    def PropDict(self, rate_dict):
        Dict = dict()
        for s1 in rate_dict["s1_vals"]:
            STerm = rate_dict[f"kCTS{s1}"] + rate_dict["kISC"] + rate_dict["knrS"] + rate_dict["krS"]
            TTerm = rate_dict["krT"] + rate_dict["knrT"] + rate_dict[f"kCTT{s1}"]
            R1QY = (rate_dict[f"kCTS{s1}"] / STerm) + (rate_dict["kISC"] * rate_dict[f"kCTT{s1}"]) / (STerm * TTerm)
            Dict[f"{s1}QY"] = R1QY
            for s2 in range(rate_dict["s2_num"]):
                if not [s1, s2] in rate_dict["error_pairs"]:
                    try:
                        Dict[f"{s1}-{s2}QY"] = R1QY * (rate_dict[f"kCT{s1}-{s2}"] / (rate_dict[f"kCT{s1}-{s2}"] + rate_dict["kDecompCl"] + rate_dict["kDecompH2O"]))
                    except ZeroDivisionError:
                        Dict[f"{s1}-{s2}QY"] = 0
                else:
                    Dict[f"{s1}-{s2}QY"] = "None"

        return Dict

class CNCompareToExp (CNScanCalc):
    
    def RateDict(self, mol: MolUtils.Tetrapyrrole, mol_name, GlobalDict) -> dict:
        ci_filename = os.path.join(path, mol_name + "_CI_CALC.out")
        force_filename = os.path.join(path, mol_name + "_FORCE_CALC.out")
        ion_ci_filename = os.path.join(path, mol_name + "_ION_CI_CALC.out")
        ci_dict = MolUtils.read_mopac_file(ci_filename)
        ion_ci_dict = MolUtils.read_mopac_file(ion_ci_filename)
        force_dict = MolUtils.read_mopac_file(force_filename)
        CenterNs, betas, mesos = mol.get_tetrapyrrole_atoms()
        RingAtoms = list(set(betas + mesos + CenterNs))

        steric_factor = self.CalcStericFactor(ci_dict["atoms"], ci_dict["coords"], 5.007, mesos)

        R1OxyE = 15.578809
        R1ES = 4.675247
        R1ET = 4.605599
        R2OxyE = 12.780939

        RateDict = {
            "kISC": self.CalcKISC(ci_dict["atoms"], ci_dict["coords"], mesos, RingAtoms) * GlobalDict["kISC"],
            "krS": self.CalcKr(ci_dict["ES"]) * GlobalDict["krS"],
            "krT": self.CalcKr(ci_dict["ET"]) * GlobalDict["krT"],
            "knrS": self.CalcKnr(ci_dict["ES"], force_dict["FreqVec"], force_dict["RateVec"]) * GlobalDict["knrS"],
            "knrT": self.CalcKnr(ci_dict["ET"], force_dict["FreqVec"], force_dict["RateVec"]) * GlobalDict["knrT"],
            "kDecompCl": self.CalcKMarcus(5.007 + 3, 3.853099 - ion_ci_dict["OxyEnergy"], steric_factor, True, 700) * GlobalDict["kDecompCl"] * 1,
            "kDecompH2O": self.CalcKMarcus(5.007 + 2, 12.116089 - ion_ci_dict["OxyEnergy"], steric_factor, True) * GlobalDict["kDecompH2O"] * 1.5,
            "kCTSR1": self.CalcKMarcus(5.007 + 1.9, ci_dict["OxyEnergy"] - ci_dict["ES"] - R1OxyE + R1ES, steric_factor, True) * GlobalDict["kCTSR1"] * 25,
            "kCTTR1": self.CalcKMarcus(5.007 + 1.9, ci_dict["OxyEnergy"] - ci_dict["ES"] - R1OxyE + R1ET, steric_factor, True) * GlobalDict["kCTSR1"] * 25,
            "kCTR2": self.CalcKMarcus(5.007, 0.9 * R2OxyE - ion_ci_dict["OxyEnergy"], steric_factor, True) * GlobalDict["kCTR2"]
        }

        return RateDict

    def PropDict(self, rate_dict, GlobalDict):
        STerm = rate_dict["kCTSR1"] + rate_dict["kISC"] + rate_dict["knrS"] + rate_dict["krS"]
        TTerm = rate_dict["krT"] + rate_dict["knrT"] + rate_dict["kCTTR1"]
        R1QY = (rate_dict["kCTSR1"] / STerm) + (rate_dict["kISC"] * rate_dict["kCTTR1"]) / (STerm * TTerm)
        try:
            R2QY = R1QY * (rate_dict["kCTR2"] / (rate_dict["kCTR2"] + rate_dict["kDecompCl"] + rate_dict["kDecompH2O"]))
        except ZeroDivisionError:
            R2QY = 0

        return {
            "R1QY": R1QY,
            "R2QY": R2QY
        }

class RDKitPropCalc (PropCalc):
    def __init__(self, rd_prop_funcs):
        self.prop_funcs = rd_prop_funcs

    def RateDict(self, mol, mol_name, Global_Dict):
        rdmol = mol.to_rdkit_Mol()
        rdmol.UpdatePropertyCache(strict = False)
        Chem.GetSymmSSSR(rdmol)
        if callable(self.prop_funcs):
            return {"prop": self.prop_funcs(rdmol)}
        else:
            res_dict = dict()
            for idx, prop_func in enumerate(self.prop_funcs):
                res_dict['prop' + str(idx + 1)] = prop_func(rdmol)
            return res_dict
    
    def PropDict(self, rate_dict):
        return rate_dict