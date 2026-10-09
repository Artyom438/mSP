#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import os, sys, re
import platform
from collections import Counter
import itertools
import numpy as np
import math
import random
import pickle


class Sequence(object):
    def __init__(self, file):
        self.file = file  # whole file path
        self.fasta_list = []  # 2-D list [sampleName, fragment, label, training or testing]
        self.sample_purpose = None  # 1-D ndarray, sample used as training dataset (True) or testing dataset(False)
        self.sequence_number = 0  # int: the number of samples
        self.sequence_type = ''  # DNA, RNA or Protein
        self.is_equal = False  # bool: sequence with equal length?
        self.minimum_length = 1  # int
        self.maximum_length = 0  # int
        self.minimum_length_without_minus = 1  # int
        self.maximum_length_without_minus = 0  # int
        self.error_msg = ''  # string

        self.fasta_list, self.sample_purpose, self.error_msg = self.read_fasta(self.file)
        self.sequence_number = len(self.fasta_list)

        if self.sequence_number > 0:
            self.is_equal, self.minimum_length, self.maximum_length, self.minimum_length_without_minus, self.maximum_length_without_minus = self.sequence_with_equal_length()
            self.sequence_type = self.check_sequence_type()


        else:
            self.error_msg = 'File format error.'

    def read_fasta(self, file):
        """
        read fasta sequence
        :param file:
        :return:
        """
        msg = ''
        if not os.path.exists(self.file):
            msg = 'Error: file %s does not exist.' % self.file
            return [], None, msg
        with open(file) as f:
            records = f.read()
        records = records.split('>')[1:]
        fasta_sequences = []
        for fasta in records:
            array = fasta.split('\n')
            header, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTUVWY-]', '-', ''.join(array[1:]).upper())
            header_array = header.split('|')
            name = header_array[0]
            label = header_array[1] if len(header_array) >= 2 else '0'
            label_train = header_array[2] if len(header_array) >= 3 else 'training'
            fasta_sequences.append([name, sequence, label, label_train])
        sample_purpose = np.array([item[3] == 'training' for item in fasta_sequences])
        return fasta_sequences, sample_purpose, msg

    def sequence_with_equal_length(self):
        """
        Check if fasta sequence is in equal length
        :return:
        """
        length_set = set()
        length_set_1 = set()
        for item in self.fasta_list:
            length_set.add(len(item[1]))
            length_set_1.add(len(re.sub('-', '', item[1])))

        length_set = sorted(length_set)
        length_set_1 = sorted(length_set_1)
        if len(length_set) == 1:
            return True, length_set[0], length_set[-1], length_set_1[0], length_set_1[-1]
        else:
            return False, length_set[0], length_set[-1], length_set_1[0], length_set_1[-1]

    def check_sequence_type(self):
        """
        Specify sequence type (Protein, DNA or RNA)
        :return:
        """
        tmp_fasta_list = []
        if len(self.fasta_list) < 100:
            tmp_fasta_list = self.fasta_list
        else:
            random_index = random.sample(range(0, len(self.fasta_list)), 100)
            for i in random_index:
                tmp_fasta_list.append(self.fasta_list[i])

        sequence = ''
        for item in tmp_fasta_list:
            sequence += item[1]

        char_set = set(sequence)
        if 5 < len(char_set) <= 21:
            for line in self.fasta_list:
                line[1] = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '-', line[1])
            return 'Protein'
        elif 0 < len(char_set) <= 5 and 'T' in char_set:
            return 'DNA'
        elif 0 < len(char_set) <= 5 and 'U' in char_set:
            for line in self.fasta_list:
                line[1] = re.sub('U', 'T', line[1])
            return 'RNA'
        else:
            return 'Unknown'

#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import sys, os, platform
import pickle

didna_list = ['Base stacking', 'Protein induced deformability', 'B-DNA twist', 'Dinucleotide GC Content', 'A-philicity',
              'Propeller twist', 'Duplex stability:(freeenergy)',
              'Duplex tability(disruptenergy)', 'DNA denaturation', 'Bending stiffness', 'Protein DNA twist',
              'Stabilising energy of Z-DNA', 'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH',
              'Breslauer_dS', 'Electron_interaction', 'Hartman_trans_free_energy', 'Helix-Coil_transition',
              'Ivanov_BA_transition', 'Lisser_BZ_transition', 'Polar_interaction', 'SantaLucia_dG',
              'SantaLucia_dH', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Stacking_energy',
              'Sugimoto_dG', 'Sugimoto_dH', 'Sugimoto_dS', 'Watson-Crick_interaction', 'Twist', 'Tilt', 'Roll',
              'Shift', 'Slide', 'Rise',
              'Clash Strength', 'Roll_roll', 'Twist stiffness', 'Tilt stiffness', 'Shift_rise',
              'Adenine content', 'Direction', 'Twist_shift', 'Enthalpy1', 'Twist_twist', 'Roll_shift',
              'Shift_slide', 'Shift2', 'Tilt3', 'Tilt1', 'Tilt4', 'Tilt2', 'Slide (DNA-protein complex)1',
              'Tilt_shift', 'Twist_tilt', 'Twist (DNA-protein complex)1', 'Tilt_rise', 'Roll_rise',
              'Stacking energy', 'Stacking energy1', 'Stacking energy2', 'Stacking energy3', 'Propeller Twist',
              'Roll11', 'Rise (DNA-protein complex)', 'Tilt_tilt', 'Roll4', 'Roll2', 'Roll3', 'Roll1',
              'Minor Groove Size', 'GC content', 'Slide_slide', 'Enthalpy', 'Shift_shift', 'Slide stiffness',
              'Melting Temperature1', 'Flexibility_slide', 'Minor Groove Distance',
              'Rise (DNA-protein complex)1', 'Tilt (DNA-protein complex)', 'Guanine content',
              'Roll (DNA-protein complex)1', 'Entropy', 'Cytosine content', 'Major Groove Size', 'Twist_rise',
              'Major Groove Distance', 'Twist (DNA-protein complex)', 'Purine (AG) content',
              'Melting Temperature', 'Free energy', 'Tilt_slide', 'Major Groove Width', 'Major Groove Depth',
              'Wedge', 'Free energy8', 'Free energy6', 'Free energy7', 'Free energy4', 'Free energy5',
              'Free energy2', 'Free energy3', 'Free energy1', 'Twist_roll', 'Shift (DNA-protein complex)',
              'Rise_rise', 'Flexibility_shift', 'Shift (DNA-protein complex)1', 'Thymine content', 'Slide_rise',
              'Tilt_roll', 'Tip', 'Keto (GT) content', 'Roll stiffness', 'Minor Groove Width', 'Inclination',
              'Entropy1', 'Roll_slide', 'Slide (DNA-protein complex)', 'Twist1', 'Twist3', 'Twist2', 'Twist5',
              'Twist4', 'Twist7', 'Twist6', 'Tilt (DNA-protein complex)1', 'Twist_slide', 'Minor Groove Depth',
              'Roll (DNA-protein complex)', 'Rise2', 'Persistance Length', 'Rise3', 'Shift stiffness',
              'Probability contacting nucleosome core', 'Mobility to bend towards major groove', 'Slide3',
              'Slide2', 'Slide1', 'Shift1', 'Bend', 'Rise1', 'Rise stiffness',
              'Mobility to bend towards minor groove']
tridna_list = ['Dnase I', 'Bendability (DNAse)', 'Bendability (consensus)', 'Trinucleotide GC Content',
               'Nucleosome positioning', 'Consensus_roll', 'Consensus-Rigid', 'Dnase I-Rigid', 'MW-Daltons',
               'MW-kg', 'Nucleosome', 'Nucleosome-Rigid']
dirna_list = ['Slide (RNA)', 'Adenine content', 'Hydrophilicity (RNA)', 'Tilt (RNA)', 'Stacking energy (RNA)',
              'Twist (RNA)', 'Entropy (RNA)', 'Roll (RNA)', 'Purine (AG) content', 'Hydrophilicity (RNA)1',
              'Enthalpy (RNA)1', 'GC content', 'Entropy (RNA)1', 'Rise (RNA)', 'Free energy (RNA)',
              'Keto (GT) content', 'Free energy (RNA)1', 'Enthalpy (RNA)', 'Guanine content', 'Shift (RNA)',
              'Cytosine content', 'Thymine content']
myDict = {
    'DAC': {'DNA': didna_list, 'RNA': dirna_list},
    'DCC': {'DNA': didna_list, 'RNA': dirna_list},
    'DACC': {'DNA': didna_list, 'RNA': dirna_list},
    'TAC': {'DNA': tridna_list, 'RNA': []},
    'TCC': {'DNA': tridna_list, 'RNA': []},
    'TACC': {'DNA': tridna_list, 'RNA': []},
    'PseDNC': {'DNA': didna_list, 'RNA': dirna_list},
    'PseKNC': {'DNA': didna_list, 'RNA': dirna_list},
    'PCPseDNC': {'DNA': didna_list, 'RNA': dirna_list},
    'PCPseTNC': {'DNA': tridna_list, 'RNA': []},
    'SCPseDNC': {'DNA': didna_list, 'RNA': dirna_list},
    'SCPseTNC': {'DNA': tridna_list, 'RNA': []},
}
myDictDefault = {
    'DAC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
            'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'DCC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
            'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'DACC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
             'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'TAC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
    'TCC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
    'TACC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
    'PseDNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
               'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'PseKNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
               'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'PCPseDNC': {
        'DNA': ['Base stacking', 'Protein induced deformability', 'B-DNA twist', 'A-philicity', 'Propeller twist',
                'Duplex stability:(freeenergy)', 'DNA denaturation', 'Bending stiffness', 'Protein DNA twist',
                'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH', 'Electron_interaction',
                'Hartman_trans_free_energy', 'Helix-Coil_transition', 'Lisser_BZ_transition', 'Polar_interaction',
                'SantaLucia_dG', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Sugimoto_dG', 'Sugimoto_dH',
                'Sugimoto_dS', 'Duplex tability(disruptenergy)', 'Stabilising energy of Z-DNA', 'Breslauer_dS',
                'Ivanov_BA_transition', 'SantaLucia_dH', 'Stacking_energy', 'Watson-Crick_interaction',
                'Dinucleotide GC Content', 'Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise'],
        'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'PCPseTNC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
    'SCPseDNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                 'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'SCPseTNC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
}
myKmer = {
    'DAC': 2, 'DCC': 2, 'DACC': 2,
    'TAC': 3, 'TCC': 3, 'TACC': 3
}
myDataFile = {
    'DAC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'DCC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'DACC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'TAC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
    'TCC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
    'TACC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
    'PseDNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'PseKNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'PCPseDNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'PCPseTNC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
    'SCPseDNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'SCPseTNC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
}

def check_acc_arguments(method, type, parameters):
    """
    Check auto-correlation parameters.
    :param method: i.e. DAC, DCC, DACC, TAC, TCC, TACC
    :param type: i.e. DNA, RNA
    :return:
    """
    kmer = myKmer[method]
    myIndex = []
    myProperty = {}
    dataFile = ''

    if type == 'DNA':
        if method in ('DAC', 'DCC', 'DACC'):
            myIndex = parameters['Di-DNA-Phychem'].strip().split(';')
        else:
            myIndex = parameters['Tri-DNA-Phychem'].strip().split(';')
    if type == 'RNA':
        if method in ('DAC', 'DCC', 'DACC'):
            myIndex = parameters['Di-RNA-Phychem'].strip().split(';')

    dataFile = myDataFile[method][type]
    with open(os.path.join('util', dataFile), 'rb') as f:
        myProperty = pickle.load(f)
    if len(myIndex) == 0 or len(myProperty) == 0:
        return myIndex, myProperty, kmer, False
    return myIndex, myProperty, kmer, True

def check_Pse_arguments(method, type, parameters):
    """
    Check auto-correlation parameters.
    :param method: i.e. PseDNC, PseKNC, PCPseDNC, PCPseTNC, SCPseDNC, SCPseTNC
    :param type: i.e. DNA, RNA
    :return:
    """
    myIndex = []
    myProperty = {}
    dataFile = ''

    if type == 'DNA':
        if method in ('PseDNC', 'PseKNC'):
            myIndex = parameters['Di-DNA-Phychem'].strip().split(';')
        if method in ('PCPseDNC', 'SCPseDNC'):
            myIndex = parameters['Di-DNA-Phychem'].strip().split(';')
        if method in ('PCPseTNC', 'SCPseTNC'):
            myIndex = parameters['Tri-DNA-Phychem'].strip().split(';')
    if type == 'RNA':
        if method in ('PseDNC', 'PseKNC'):
            myIndex = parameters['Di-RNA-Phychem'].strip().split(';')
        if method in ('PCPseDNC', 'SCPseDNC'):
            myIndex = parameters['Di-RNA-Phychem'].strip().split(';')

    dataFile = myDataFile[method][type]
    with open(os.path.join('util', dataFile), 'rb') as f:
        myProperty = pickle.load(f)

    if len(myIndex) == 0 or len(myProperty) == 0:
        return myIndex, myProperty, False

    return myIndex, myProperty, True

class Descriptor(Sequence):
    def __init__(self, file, kw):
        super(Descriptor, self).__init__(file=file)
        self.kw = kw  # dict
        self.encoding_array = np.array([])  # 2-D ndarray with column name and index name
        self.column = 0  # int
        self.row = 0  # int
        """ variable for ACC descriptors """
        self.myDiIndex = {
            'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3,
            'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7,
            'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11,
            'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15
        }
        self.myTriIndex = {
            'AAA': 0, 'AAC': 1, 'AAG': 2, 'AAT': 3,
            'ACA': 4, 'ACC': 5, 'ACG': 6, 'ACT': 7,
            'AGA': 8, 'AGC': 9, 'AGG': 10, 'AGT': 11,
            'ATA': 12, 'ATC': 13, 'ATG': 14, 'ATT': 15,
            'CAA': 16, 'CAC': 17, 'CAG': 18, 'CAT': 19,
            'CCA': 20, 'CCC': 21, 'CCG': 22, 'CCT': 23,
            'CGA': 24, 'CGC': 25, 'CGG': 26, 'CGT': 27,
            'CTA': 28, 'CTC': 29, 'CTG': 30, 'CTT': 31,
            'GAA': 32, 'GAC': 33, 'GAG': 34, 'GAT': 35,
            'GCA': 36, 'GCC': 37, 'GCG': 38, 'GCT': 39,
            'GGA': 40, 'GGC': 41, 'GGG': 42, 'GGT': 43,
            'GTA': 44, 'GTC': 45, 'GTG': 46, 'GTT': 47,
            'TAA': 48, 'TAC': 49, 'TAG': 50, 'TAT': 51,
            'TCA': 52, 'TCC': 53, 'TCG': 54, 'TCT': 55,
            'TGA': 56, 'TGC': 57, 'TGG': 58, 'TGT': 59,
            'TTA': 60, 'TTC': 61, 'TTG': 62, 'TTT': 63
        }

    """ Protein descriptors """
    ''' KNN descriptor '''
    def CalculateContent(self, myDistance, j, myLabelSets):
        content = []
        myDict = {}
        for i in myLabelSets:
            myDict[i] = 0
        for i in range(j):
            myDict[myDistance[i][0]] = myDict[myDistance[i][0]] + 1
        for i in myLabelSets:
            content.append(myDict[myLabelSets[i]] / j)
        return content

    ''' end Protein KNN descriptor '''

    def gapModel(self, fastas, myDict, gDict, gNames, ktuple, glValue):
        encodings = []
        header = ['SampleName', 'label']
        if ktuple == 1:
            header = header + [g + '_gap' + str(glValue) for g in gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                numDict = {}
                for j in range(0, len(sequence), glValue + 1):
                    numDict[gDict[myDict[sequence[j]]]] = numDict.get(gDict[myDict[sequence[j]]], 0) + 1

                for g in gNames:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        if ktuple == 2:
            header = header + [g1 + '_' + g2 + '_gap' + str(glValue) for g1 in gNames for g2 in gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                numDict = {}
                for j in range(0, len(sequence), glValue + 1):
                    if j + 1 < len(sequence):
                        numDict[gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + 1]]]] = numDict.get(
                            gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + 1]]], 0) + 1

                for g in [g1 + '_' + g2 for g1 in gNames for g2 in gNames]:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        if ktuple == 3:
            header = header + [g1 + '_' + g2 + '_' + g3 + '_gap' + str(glValue) for g1 in gNames for g2 in gNames for g3
                               in gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                numDict = {}
                for j in range(0, len(sequence), glValue + 1):
                    if j + 1 < len(sequence) and j + 2 < len(sequence):
                        numDict[gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + 1]]] + '_' + gDict[
                            myDict[sequence[j + 2]]]] = numDict.get(
                            gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + 1]]] + '_' + gDict[
                                myDict[sequence[j + 2]]], 0) + 1

                for g in [g1 + '_' + g2 + '_' + g3 for g1 in gNames for g2 in gNames for g3 in gNames]:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        return encodings

    def lambdaModel(self, fastas, myDict, gDict, gNames, ktuple, glValue):
        encodings = []
        header = ['SampleName', 'label']
        if ktuple == 1:
            header = header + [g + '_LC' + str(glValue) for g in gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                numDict = {}
                for j in range(0, len(sequence)):
                    numDict[gDict[myDict[sequence[j]]]] = numDict.get(gDict[myDict[sequence[j]]], 0) + 1

                for g in gNames:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        if ktuple == 2:
            header = header + [g1 + '_' + g2 + '_LC' + str(glValue) for g1 in gNames for g2 in gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                numDict = {}
                for j in range(0, len(sequence)):
                    if j + glValue < len(sequence):
                        numDict[gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + glValue]]]] = numDict.get(
                            gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + glValue]]], 0) + 1

                for g in [g1 + '_' + g2 for g1 in gNames for g2 in gNames]:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        if ktuple == 3:
            header = header + [g1 + '_' + g2 + '_' + g3 + '_LC' + str(glValue) for g1 in gNames for g2 in gNames for g3
                               in
                               gNames]
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                numDict = {}
                for j in range(0, len(sequence)):
                    if j + glValue < len(sequence) and j + 2 * glValue < len(sequence):
                        numDict[gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + glValue]]] + '_' + gDict[
                            myDict[sequence[j + 2 * glValue]]]] = numDict.get(
                            gDict[myDict[sequence[j]]] + '_' + gDict[myDict[sequence[j + glValue]]] + '_' + gDict[
                                myDict[sequence[j + 2 * glValue]]], 0) + 1

                for g in [g1 + '_' + g2 + '_' + g3 for g1 in gNames for g2 in gNames for g3 in gNames]:
                    code.append(numDict.get(g, 0))
                encodings.append(code)
        return encodings

    """ DNA and RNA descriptors """

    def kmerArray(self, sequence, k):
        kmer = []
        for i in range(len(sequence) - k + 1):
            kmer.append(sequence[i:i + k])
        return kmer

    def Kmer(self):
        try:
            fastas = self.fasta_list
            k = self.kw['kmer']
            upto = False
            normalize = True
            type = self.sequence_type

            encoding = []
            header = ['SampleName', 'label']
            NA = 'ACGT'
            if type in ("DNA", 'RNA'):
                NA = 'ACGT'
            else:
                NA = 'ACDEFGHIKLMNPQRSTVWY'

            if upto == True:
                for tmpK in range(1, k + 1):
                    for kmer in itertools.product(NA, repeat=tmpK):
                        header.append(''.join(kmer))
                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    count = Counter()
                    for tmpK in range(1, k + 1):
                        kmers = self.kmerArray(sequence, tmpK)
                        count.update(kmers)
                        if normalize == True:
                            for key in count:
                                if len(key) == tmpK:
                                    count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(header)):
                        if header[j] in count:
                            code.append(count[header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)
            else:
                for kmer in itertools.product(NA, repeat=k):
                    header.append(''.join(kmer))
                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    kmers = self.kmerArray(sequence, k)
                    count = Counter()
                    count.update(kmers)
                    if normalize == True:
                        for key in count:
                            count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(header)):
                        if header[j] in count:
                            code.append(count[header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encoding, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encoding
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def mismatch_count(self, seq1, seq2):
        mismatch = 0
        for i in range(min([len(seq1), len(seq2)])):
            if seq1[i] != seq2[i]:
                mismatch += 1
        return mismatch

    def Mismatch(self):
        try:
            k = self.kw['kmer']
            m = self.kw['mismatch']
            NN = 'ACGT'

            encoding = []
            header = ['SampleName', 'label']
            template_dict = {}
            for kmer in itertools.product(NN, repeat=k):
                header.append(''.join(kmer))
                template_dict[''.join(kmer)] = 0
            encoding.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                kmers = self.kmerArray(sequence, k)
                tmp_dict = template_dict.copy()
                for kmer in kmers:
                    for key in tmp_dict:
                        if self.mismatch_count(kmer, key) <= m:
                            tmp_dict[key] += 1
                code = [name, label] + [tmp_dict[key] for key in sorted(tmp_dict.keys())]
                encoding.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encoding, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encoding
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    """ Subsequence """
    # code from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4830532/bin/pone.0153268.s001.zip
    def GetKmerDict(self, piRNAletter, k):
        kmerlst = []
        partkmers = list(itertools.combinations_with_replacement(piRNAletter, k))
        for element in partkmers:
            elelst = set(itertools.permutations(element, k))
            strlst = [''.join(ele) for ele in elelst]
            kmerlst += strlst
        kmerlst = np.sort(kmerlst)
        kmerdict = {kmerlst[i]: i for i in range(len(kmerlst))}
        return kmerdict

    def GetSubsequenceProfile(self, instances, piRNAletter, k, delta):
        kmerdict = self.GetKmerDict(piRNAletter, k)
        X = []
        for sequence in instances:
            vector = self.GetSubsequenceProfileVector(sequence, kmerdict, k, delta)
            X.append(vector)
        X = np.array(X)
        return X

    def GetSubsequenceProfileVector(self, sequence, kmerdict, k, delta):
        vector = np.zeros((1, len(kmerdict)))
        sequence = np.array(list(sequence))
        n = len(sequence)
        index_lst = list(itertools.combinations(range(n), k))
        for subseq_index in index_lst:
            subseq_index = list(subseq_index)
            subsequence = sequence[subseq_index]
            position = kmerdict.get(''.join(subsequence))
            subseq_length = subseq_index[-1] - subseq_index[0] + 1
            subseq_score = 1 if subseq_length == k else delta ** subseq_length
            vector[0, position] += subseq_score
        return list(vector[0])

    def Subsequence(self):
        try:
            k = self.kw['kmer']
            delta = self.kw['delta']
            NN_list = ['A', 'C', 'G', 'T']

            encoding = []
            header = ['SampleName', 'label']
            for kmer in itertools.product(NN_list, repeat=k):
                header.append(''.join(kmer))
            encoding.append(header)

            instances = [elem[1] for elem in self.fasta_list]

            code = self.GetSubsequenceProfile(instances, NN_list, k, delta)
            info_list = np.array([[item[0], item[2]] for item in self.fasta_list])
            code = np.hstack((info_list, code))

            self.encoding_array = np.vstack((header, code))
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    """ end of Subsequence """

    def RC(self, kmer):
        myDict = {
            'A': 'T',
            'C': 'G',
            'G': 'C',
            'T': 'A'
        }
        return ''.join([myDict[nc] for nc in kmer[::-1]])

    def generateRCKmer(self, kmerList):
        rckmerList = set()
        myDict = {
            'A': 'T',
            'C': 'G',
            'G': 'C',
            'T': 'A'
        }
        for kmer in kmerList:
            rckmerList.add(sorted([kmer, ''.join([myDict[nc] for nc in kmer[::-1]])])[0])
        return sorted(rckmerList)

    def RCKmer(self):
        try:
            fastas = self.fasta_list
            k = self.kw['kmer']
            upto = False
            normalize = True

            encoding = []
            header = ['SampleName', 'label']
            NA = 'ACGT'

            if upto == True:
                for tmpK in range(1, k + 1):
                    tmpHeader = []
                    for kmer in itertools.product(NA, repeat=tmpK):
                        tmpHeader.append(''.join(kmer))
                    header = header + self.generateRCKmer(tmpHeader)
                myDict = {}
                for kmer in header[2:]:
                    rckmer = self.RC(kmer)
                    if kmer != rckmer:
                        myDict[rckmer] = kmer
                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    count = Counter()
                    for tmpK in range(1, k + 1):
                        kmers = self.kmerArray(sequence, tmpK)
                        for j in range(len(kmers)):
                            if kmers[j] in myDict:
                                kmers[j] = myDict[kmers[j]]
                        count.update(kmers)
                        if normalize == True:
                            for key in count:
                                if len(key) == tmpK:
                                    count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(header)):
                        if header[j] in count:
                            code.append(count[header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)
            else:
                tmpHeader = []
                for kmer in itertools.product(NA, repeat=k):
                    tmpHeader.append(''.join(kmer))
                header = header + self.generateRCKmer(tmpHeader)
                myDict = {}
                for kmer in header[2:]:
                    rckmer = self.RC(kmer)
                    if kmer != rckmer:
                        myDict[rckmer] = kmer

                encoding.append(header)
                for i in fastas:
                    name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                    kmers = self.kmerArray(sequence, k)
                    for j in range(len(kmers)):
                        if kmers[j] in myDict:
                            kmers[j] = myDict[kmers[j]]
                    count = Counter()
                    count.update(kmers)
                    if normalize == True:
                        for key in count:
                            count[key] = count[key] / len(kmers)
                    code = [name, label]
                    for j in range(2, len(header)):
                        if header[j] in count:
                            code.append(count[header[j]])
                        else:
                            code.append(0)
                    encoding.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encoding, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encoding
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def NAC(self):
        try:
            NA = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for i in NA:
                header.append(i)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                count = Counter(sequence)
                for key in count:
                    count[key] = count[key] / len(sequence)
                code = [name, label]
                for na in NA:
                    code.append(count[na])
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def DNC(self):
        try:
            base = 'ACGT'
            encodings = []
            dinucleotides = [n1 + n2 for n1 in base for n2 in base]
            header = ['SampleName', 'label'] + dinucleotides
            encodings.append(header)

            AADict = {}
            for i in range(len(base)):
                AADict[base[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                tmpCode = [0] * 16
                for j in range(len(sequence) - 2 + 1):
                    tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 4 + AADict[
                        sequence[j + 1]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]
                code = code + tmpCode
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def TNC(self):
        try:
            AA = 'ACGT'
            encodings = []
            triPeptides = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
            header = ['SampleName', 'label'] + triPeptides
            encodings.append(header)

            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                tmpCode = [0] * 64
                for j in range(len(sequence) - 3 + 1):
                    tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j + 1]] * 4 + AADict[sequence[j + 2]]] = tmpCode[
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j]] * 16 +
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j + 1]] * 4 +
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j + 2]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]
                code = code + tmpCode
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def ANF(self):
        try:
            if not self.is_equal:
                self.error_msg = 'ANF descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, len(self.fasta_list[0][1]) + 1):
                header.append('ANF.' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for j in range(len(sequence)):
                    code.append(sequence[0: j + 1].count(sequence[j]) / (j + 1))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def NCP(self):
        try:
            if not self.is_equal:
                self.error_msg = 'NCP descriptor need fasta sequence with equal length.'
                return False
            chemical_property = {
                'A': [1, 1, 1],
                'C': [0, 1, 0],
                'G': [1, 0, 0],
                'T': [0, 0, 1],
                'U': [0, 0, 1],
                '-': [0, 0, 0],
            }
            AA = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, len(self.fasta_list[0][1]) * 3 + 1):
                header.append('NCP.F' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    code = code + chemical_property.get(aa, [0, 0, 0])
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def ENAC(self):
        try:
            if not self.is_equal:
                self.error_msg = 'ENAC descriptor need fasta sequence with equal length.'
                return False
            window = self.kw['sliding_window']
            if self.minimum_length < window:
                self.error_msg = 'ENAC descriptor, all the sequence length should be larger than the sliding window'
                return False
            AA = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for w in range(1, len(self.fasta_list[0][1]) - window + 2):
                for aa in AA:
                    header.append('SW.' + str(w) + '.' + aa)
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for j in range(len(sequence)):
                    if j < len(sequence) and j + window <= len(sequence):
                        count = Counter(sequence[j:j + window])
                        for key in count:
                            count[key] = count[key] / len(sequence[j:j + window])
                        for aa in AA:
                            code.append(count[aa])
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def binary(self):
        try:
            if not self.is_equal:
                self.error_msg = 'binary descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, len(self.fasta_list[0][1]) * 4 + 1):
                header.append('BINARY.F' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    if aa == '-':
                        code = code + [0, 0, 0, 0]
                        continue
                    for aa1 in AA:
                        tag = 1 if aa == aa1 else 0
                        code.append(tag)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def CKSNAP(self):
        try:
            gap = self.kw['kspace']
            if self.minimum_length_without_minus < gap + 2:
                self.error_msg = 'CKSNAP - all the sequence length should be larger than the (gap value) + 2 = %s.' % (
                        gap + 2)
                return False

            AA = 'ACGT'
            encodings = []
            aaPairs = []
            for aa1 in AA:
                for aa2 in AA:
                    aaPairs.append(aa1 + aa2)

            header = ['SampleName', 'label']
            for g in range(gap + 1):
                for aa in aaPairs:
                    header.append(aa + '.gap' + str(g))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for g in range(gap + 1):
                    myDict = {}
                    for pair in aaPairs:
                        myDict[pair] = 0
                    sum = 0
                    for index1 in range(len(sequence)):
                        index2 = index1 + g + 1
                        if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                            index2] in AA:
                            myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                            sum = sum + 1
                    for pair in aaPairs:
                        code.append(myDict[pair] / sum)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def CalculateMatrix(self, data, order):
        matrix = np.zeros((len(data[0]) - 2, 64))
        for i in range(len(data[0]) - 2):  # position
            for j in range(len(data)):
                if re.search('-', data[j][i:i + 3]):
                    pass
                else:
                    matrix[i][order[data[j][i:i + 3]]] += 1
        return matrix

    def PSTNPss(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PSTNPss descriptor need fasta sequence with equal length.'
                return False

            fastas = []
            for item in self.fasta_list:
                if item[3] == 'training':
                    fastas.append(item)
                    fastas.append([item[0], item[1], item[2], 'testing'])
                else:
                    fastas.append(item)

            for i in fastas:
                if re.search('[^ACGT-]', i[1]):
                    self.error_msg = 'Illegal character included in the fasta sequences, only the "ACGT[U]" are allowed by this encoding scheme.'
                    return False

            encodings = []
            header = ['SampleName', 'label']
            for pos in range(len(fastas[0][1]) - 2):
                header.append('Pos.%d' % (pos + 1))
            encodings.append(header)

            positive = []
            negative = []
            positive_key = []
            negative_key = []
            for i in fastas:
                if i[3] == 'training':
                    if i[2] == '1':
                        positive.append(i[1])
                        positive_key.append(i[0])
                    else:
                        negative.append(i[1])
                        negative_key.append(i[0])

            nucleotides = ['A', 'C', 'G', 'T']
            trinucleotides = [n1 + n2 + n3 for n1 in nucleotides for n2 in nucleotides for n3 in nucleotides]
            order = {}
            for i in range(len(trinucleotides)):
                order[trinucleotides[i]] = i

            matrix_po = self.CalculateMatrix(positive, order)
            matrix_ne = self.CalculateMatrix(negative, order)

            positive_number = len(positive)
            negative_number = len(negative)

            for i in fastas:
                if i[3] == 'testing':
                    name, sequence, label = i[0], i[1], i[2]
                    code = [name, label]
                    for j in range(len(sequence) - 2):
                        if re.search('-', sequence[j: j + 3]):
                            code.append(0)
                        else:
                            p_num, n_num = positive_number, negative_number
                            po_number = matrix_po[j][order[sequence[j: j + 3]]]
                            if i[0] in positive_key and po_number > 0:
                                po_number -= 1
                                p_num -= 1
                            ne_number = matrix_ne[j][order[sequence[j: j + 3]]]
                            if i[0] in negative_key and ne_number > 0:
                                ne_number -= 1
                                n_num -= 1
                            code.append(po_number / p_num - ne_number / n_num)

                    encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def PSTNPds(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PSTNPds descriptor need fasta sequence with equal length.'
                return False

            fastas = []
            for item in self.fasta_list:
                if item[3] == 'training':
                    fastas.append(item)
                    fastas.append([item[0], item[1], item[2], 'testing'])
                else:
                    fastas.append(item)

            for i in fastas:
                if re.search('[^ACGT-]', i[1]):
                    self.error_msg = 'Illegal character included in the fasta sequences, only the "ACGT[U]" are allowed by this encoding scheme.'
                    return False

            for i in fastas:
                i[1] = re.sub('T', 'A', i[1])
                i[1] = re.sub('G', 'C', i[1])

            encodings = []
            header = ['SampleName', 'label']
            for pos in range(len(fastas[0][1]) - 2):
                header.append('Pos.%d' % (pos + 1))
            encodings.append(header)

            positive = []
            negative = []
            positive_key = []
            negative_key = []
            for i in fastas:
                if i[3] == 'training':
                    if i[2] == '1':
                        positive.append(i[1])
                        positive_key.append(i[0])
                    else:
                        negative.append(i[1])
                        negative_key.append(i[0])

            nucleotides = ['A', 'C']
            trinucleotides = [n1 + n2 + n3 for n1 in nucleotides for n2 in nucleotides for n3 in nucleotides]
            order = {}
            for i in range(len(trinucleotides)):
                order[trinucleotides[i]] = i

            matrix_po = self.CalculateMatrix(positive, order)
            matrix_ne = self.CalculateMatrix(negative, order)

            positive_number = len(positive)
            negative_number = len(negative)

            for i in fastas:
                if i[3] == 'testing':
                    name, sequence, label = i[0], i[1], i[2]
                    code = [name, label]
                    for j in range(len(sequence) - 2):
                        if re.search('-', sequence[j: j + 3]):
                            code.append(0)
                        else:
                            p_num, n_num = positive_number, negative_number
                            po_number = matrix_po[j][order[sequence[j: j + 3]]]
                            if i[0] in positive_key and po_number > 0:
                                po_number -= 1
                                p_num -= 1
                            ne_number = matrix_ne[j][order[sequence[j: j + 3]]]
                            if i[0] in negative_key and ne_number > 0:
                                ne_number -= 1
                                n_num -= 1
                            code.append(po_number / p_num - ne_number / n_num)

                    encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def EIIP(self):
        try:
            if not self.is_equal:
                self.error_msg = 'EIIP descriptor need fasta sequence with equal length.'
                return False
            fastas = self.fasta_list
            AA = 'ACGT'
            EIIP_dict = {
                'A': 0.1260,
                'C': 0.1340,
                'G': 0.0806,
                'T': 0.1335,
                '-': 0,
            }
            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, len(fastas[0][1]) + 1):
                header.append('F' + str(i))
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for aa in sequence:
                    code.append(EIIP_dict.get(aa, 0))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def TriNcleotideComposition(self, sequence, base):
        trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
        tnc_dict = {}
        for triN in trincleotides:
            tnc_dict[triN] = 0
        for i in range(len(sequence) - 2):
            tnc_dict[sequence[i:i + 3]] += 1
        for key in tnc_dict:
            tnc_dict[key] /= (len(sequence) - 2)
        return tnc_dict

    def PseEIIP(self):
        try:
            fastas = self.fasta_list
            for i in fastas:
                if re.search('[^ACGT-]', i[1]):
                    self.error_msg = 'Illegal character included in the fasta sequences, only the "ACGT-" are allowed by this PseEIIP scheme.'
                    return False
            base = 'ACGT'
            EIIP_dict = {
                'A': 0.1260,
                'C': 0.1340,
                'G': 0.0806,
                'T': 0.1335,
            }
            trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
            EIIPxyz = {}
            for triN in trincleotides:
                EIIPxyz[triN] = EIIP_dict[triN[0]] + EIIP_dict[triN[1]] + EIIP_dict[triN[2]]

            encodings = []
            header = ['SampleName', 'label'] + trincleotides
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                trincleotide_frequency = self.TriNcleotideComposition(sequence, base)
                code = code + [EIIPxyz[triN] * trincleotide_frequency[triN] for triN in trincleotides]
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def ASDC(self):
        try:
            self.encoding_array = np.array([])

            AA = 'ACGT'
            encodings = []
            aaPairs = []
            for aa1 in AA:
                for aa2 in AA:
                    aaPairs.append(aa1 + aa2)

            header = ['SampleName', 'label']
            header += [aa1 + aa2 for aa1 in AA for aa2 in AA]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                sum = 0
                pair_dict = {}
                for pair in aaPairs:
                    pair_dict[pair] = 0
                for j in range(len(sequence)):
                    for k in range(j + 1, len(sequence)):
                        if sequence[j] in AA and sequence[k] in AA:
                            pair_dict[sequence[j] + sequence[k]] += 1
                            sum += 1
                for pair in aaPairs:
                    code.append(pair_dict[pair] / sum)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def DBE(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'DBE descriptor need fasta sequence with equal length.'
                return False

            AA_dict = {
                'AA': [0, 0, 0, 0],
                'AC': [0, 0, 0, 1],
                'AG': [0, 0, 1, 0],
                'AT': [0, 0, 1, 1],
                'CA': [0, 1, 0, 0],
                'CC': [0, 1, 0, 1],
                'CG': [0, 1, 1, 0],
                'CT': [0, 1, 1, 1],
                'GA': [1, 0, 0, 0],
                'GC': [1, 0, 0, 1],
                'GG': [1, 0, 1, 0],
                'GT': [1, 0, 1, 1],
                'TA': [1, 1, 0, 0],
                'TC': [1, 1, 0, 1],
                'TG': [1, 1, 1, 0],
                'TT': [1, 1, 1, 1],
            }

            encodings = []
            header = ['SampleName', 'label']
            for i in range((len(self.fasta_list[0][1]) - 1) * 4):
                header.append('F%s' %(i + 1))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for j in range(len(sequence) - 1):
                    if sequence[j] + sequence[j + 1] in AA_dict:
                        code += AA_dict[sequence[j] + sequence[j + 1]]
                    else:
                        code += [0.5, 0.5, 0.5, 0.5]
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def LPDF(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'LPDF descriptor need fasta sequence with equal length.'
                return False

            encodings = []
            header = ['SampleName', 'label']
            header += ['LPDF_pos%s' % i for i in range(1, len(self.fasta_list[0][1]))]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                dinucleotide_dict = {
                    'AA': 0,
                    'AC': 0,
                    'AG': 0,
                    'AT': 0,
                    'A-': 0,
                    'CA': 0,
                    'CC': 0,
                    'CG': 0,
                    'CT': 0,
                    'C-': 0,
                    'GA': 0,
                    'GC': 0,
                    'GG': 0,
                    'GT': 0,
                    'G-': 0,
                    'TA': 0,
                    'TC': 0,
                    'TG': 0,
                    'TT': 0,
                    'T-': 0,
                    '-A': 0,
                    '-C': 0,
                    '-G': 0,
                    '-T': 0,
                    '--': 0,
                }
                for j in range(1, len(sequence)):
                    dinucleotide_dict[sequence[j] + sequence[j - 1]] += 1
                    code.append(dinucleotide_dict[sequence[j] + sequence[j - 1]] / j)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def DPCP(self):
        try:
            self.encoding_array = np.array([])

            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.kw['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.kw['Di-RNA-Phychem'].split(';')

            try:
                with open(os.path.join('util', file_name), 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            base = 'ACGT'
            encodings = []
            dinucleotides = [n1 + n2 + '_' + p_name for p_name in property_name for n1 in base for n2 in base]
            header = ['SampleName', 'label'] + dinucleotides
            encodings.append(header)

            AADict = {}
            for i in range(len(base)):
                AADict[base[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                tmpCode = [0] * 16
                for j in range(len(sequence) - 2 + 1):
                    tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 4 + AADict[
                        sequence[j + 1]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]

                for p_name in property_name:
                    normalized_code = []
                    for j in range(len(tmpCode)):
                        normalized_code.append(tmpCode[j] * float(property_dict[p_name][j]))
                    code += normalized_code
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def DPCP_type2(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'DPCP type2 descriptor need fasta sequence with equal length.'
                return False

            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.kw['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.kw['Di-RNA-Phychem'].split(';')

            try:
                with open(os.path.join('util', file_name), 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            AA = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for i in range(len(self.fasta_list[0][1]) - 1):
                for p_name in property_name:
                    header.append(p_name + '_pos%s' %(i+1))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in AA for aa2 in AA]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for p_name in property_name:
                    for j in range(len(sequence) - 1):
                        if sequence[j: j + 2] in AADict:
                            code.append(property_dict[p_name][AADict[sequence[j: j + 2]]])
                        else:
                            code.append(0)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def TPCP(self):
        try:
            self.encoding_array = np.array([])

            file_name = 'tridnaPhyche.data'
            property_name = self.kw['Tri-DNA-Phychem'].split(';')

            try:
                with open(os.path.join('util', file_name), 'rb') as handle:
                    property_dict = pickle.load(handle)
                property_name = property_dict.keys()
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            AA = 'ACGT'
            encodings = []
            triPeptides = [aa1 + aa2 + aa3 + '_' + p_name for p_name in property_name for aa1 in AA for aa2 in AA for aa3 in
                        AA]
            header = ['SampleName', 'label'] + triPeptides
            encodings.append(header)

            AADict = {}
            for i in range(len(AA)):
                AADict[AA[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                tmpCode = [0] * 64
                for j in range(len(sequence) - 3 + 1):
                    tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j + 1]] * 4 + AADict[sequence[j + 2]]] = tmpCode[
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j]] * 16 +
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j + 1]] * 4 +
                                                                                                                    AADict[
                                                                                                                        sequence[
                                                                                                                            j + 2]]] + 1
                if sum(tmpCode) != 0:
                    tmpCode = [i / sum(tmpCode) for i in tmpCode]

                for p_name in property_name:
                    normalized_code = []
                    for j in range(len(tmpCode)):
                        normalized_code.append(tmpCode[j] * float(property_dict[p_name][j]))
                    code += normalized_code
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def TPCP_type2(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'TPCP type2 descriptor need fasta sequence with equal length.'
                return False

            if self.sequence_type == 'DNA':
                file_name = 'tridnaPhyche.data'
                property_name = self.kw['Tri-DNA-Phychem'].split(';')

            try:
                with open(os.path.join('util', file_name), 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False

            AA = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for i in range(len(self.fasta_list[0][1]) - 2):
                for p_name in property_name:
                    header.append(p_name + '_pos%s' %(i+1))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for p_name in property_name:
                    for j in range(len(sequence) - 2):
                        if sequence[j: j + 3] in AADict:
                            code.append(property_dict[p_name][AADict[sequence[j: j + 3]]])
                        else:
                            code.append(0)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def MMI(self):
        try:
            NA = 'ACGT'
            dinucleotide_list = [a1 + a2 for a1 in NA for a2 in NA]
            trinucleotide_list = [a1 + a2 + a3 for a1 in NA for a2 in NA for a3 in NA]
            dinucleotide_dict = {}
            trinucleotide_dict = {}
            for elem in dinucleotide_list:
                dinucleotide_dict[''.join(sorted(elem))] = 0
            for elem in trinucleotide_list:
                trinucleotide_dict[''.join(sorted(elem))] = 0

            encodings = []
            header = ['SampleName', 'label']
            header += ['MMI_%s' % elem for elem in sorted(dinucleotide_dict.keys())]
            header += ['MMI_%s' % elem for elem in sorted(trinucleotide_dict.keys())]
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                f1_dict = {
                    'A': 0,
                    'C': 0,
                    'G': 0,
                    'T': 0,
                }
                f2_dict = dinucleotide_dict.copy()
                f3_dict = trinucleotide_dict.copy()

                for elem in sequence:
                    if elem in f1_dict:
                        f1_dict[elem] += 1
                for key in f1_dict:
                    f1_dict[key] /= len(sequence)

                for i in range(len(sequence) - 1):
                    if ''.join(sorted(sequence[i: i + 2])) in f2_dict:
                        f2_dict[''.join(sorted(sequence[i: i + 2]))] += 1
                for key in f2_dict:
                    f2_dict[key] /= (len(sequence) - 1)

                for i in range(len(sequence) - 2):
                    if ''.join(sorted(sequence[i: i + 3])) in f3_dict:
                        f3_dict[''.join(sorted(sequence[i: i + 3]))] += 1
                for key in f3_dict:
                    f3_dict[key] /= (len(sequence) - 2)

                for key in sorted(f2_dict.keys()):
                    if f2_dict[key] != 0 and f1_dict[key[0]] * f1_dict[key[1]] != 0:
                        code.append(f2_dict[key] * math.log(f2_dict[key] / (f1_dict[key[0]] * f1_dict[key[1]])))
                    else:
                        code.append(0)
                for key in sorted(f3_dict.keys()):
                    element_1 = 0
                    element_2 = 0
                    element_3 = 0
                    if f2_dict[key[0:2]] != 0 and f1_dict[key[0]] * f1_dict[key[1]] != 0:
                        element_1 = f2_dict[key[0:2]] * math.log(f2_dict[key[0:2]] / (f1_dict[key[0]] * f1_dict[key[1]]))
                    if f2_dict[key[0] + key[2]] != 0 and f1_dict[key[2]] != 0:
                        element_2 = (f2_dict[key[0] + key[2]] / f1_dict[key[2]]) * math.log(
                            f2_dict[key[0] + key[2]] / f1_dict[key[2]])
                    if f2_dict[key[1:3]] != 0 and f3_dict[key] / f2_dict[key[1:3]] != 0:
                        element_3 = (f3_dict[key] / f2_dict[key[1:3]]) * math.log(f3_dict[key] / f2_dict[key[1:3]])
                    code.append(element_1 + element_2 - element_3)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    ''' DNA/RNA KNN descriptor '''
    def SimN(self, a, b):
        score_matrix = [
            [ 2, -1, -1, -1, -1],  # A
            [-1,  2, -1, -1, -1],  # C
            [-1, -1,  2, -1, -1],  # G
            [-1, -1, -1,  2, -1],  # T
            [-1, -1, -1, -1,  2],  # -
        ]
        AA = 'ACGT-'
        myDict = {}
        for i in range(len(AA)):
            myDict[AA[i]] = i
        maxValue, minValue = 2, -1
        return (score_matrix[myDict[a]][myDict[b]] - minValue) / (maxValue - minValue)

    def CalculateDistanceN(self, sequence1, sequence2):
        if len(sequence1) != len(sequence2):
            self.error_msg = 'KNN descriptor need fasta sequence with equal length.'
            return 1
        distance = 1 - sum([self.SimN(sequence1[i], sequence2[i]) for i in range(len(sequence1))]) / len(sequence1)
        return distance

    def KNN(self):
        try:
            # clear
            self.encoding_array = np.array([])

            if not self.is_equal:
                self.error_msg = 'KNN descriptor need fasta sequence with equal length.'
                return False

            topK_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                        0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]

            training_data = []
            training_label = {}
            for i in self.fasta_list:
                if i[3] == 'training':
                    training_data.append(i)
                    training_label[i[0]] = int(i[2])
            tmp_label_sets = list(set(training_label.values()))

            topK_numbers = []
            for i in topK_values:
                topK_numbers.append(math.ceil(len(training_data) * i))

            # calculate pair distance
            distance_dict = {}
            for i in range(len(self.fasta_list)):
                name_seq1, sequence_1, label_1, usage_1 = self.fasta_list[i][0], self.fasta_list[i][1], self.fasta_list[i][2], self.fasta_list[i][3]
                for j in range(i+1, len(self.fasta_list)):
                    name_seq2, sequence_2, label_2, usage_2 = self.fasta_list[j][0], self.fasta_list[j][1], self.fasta_list[j][2], self.fasta_list[j][3]
                    if usage_1 == 'testing' and usage_2 == 'testing':
                        continue
                    else:
                        distance_dict[':'.join(sorted([name_seq1, name_seq2]))] = self.CalculateDistanceN(sequence_1, sequence_2)

            encodings = []
            header = ['#', 'label']
            for k in topK_numbers:
                for l in tmp_label_sets:
                    header.append('Top' + str(k) + '.label' + str(l))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                tmp_distance_list = []
                for j in range(len(training_data)):
                    if name != training_data[j][0]:
                        tmp_distance_list.append([int(training_data[j][2]), distance_dict.get(':'.join(sorted([name, training_data[j][0]])), 1)])

                tmp_distance_list = np.array(tmp_distance_list)
                tmp_distance_list = tmp_distance_list[np.lexsort(tmp_distance_list.T)]

                for j in topK_numbers:
                    code += self.CalculateContent(tmp_distance_list, j, tmp_label_sets)
                encodings.append(code)

            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    ''' ending DNA/RNA KNN descriptor '''

    def PS2(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PS2 descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            AA_list = [a1 + a2 for a1 in AA for a2 in AA]
            AA_dict = {}
            for i in range(len(AA_list)):
                AA_dict[AA_list[i]] = [0] * 16
                AA_dict[AA_list[i]][i] = 1

            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, (len(self.fasta_list[0][1]) - 1) * 16 + 1):
                header.append('BINARY.F' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for j in range(len(sequence) - 1):
                    code += AA_dict.get(sequence[j: j+2], [0]*16)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def PS3(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PS3 descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            AA_list = [a1 + a2 + a3 for a1 in AA for a2 in AA for a3 in AA]
            AA_dict = {}
            for i in range(len(AA_list)):
                AA_dict[AA_list[i]] = [0] * 64
                AA_dict[AA_list[i]][i] = 1

            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, (len(self.fasta_list[0][1]) - 2) * 64 + 1):
                header.append('BINARY.F' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for j in range(len(sequence) - 2):
                    code += AA_dict.get(sequence[j: j+3], [0]*64)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def PS4(self):
        try:
            if not self.is_equal:
                self.error_msg = 'PS4 descriptor need fasta sequence with equal length.'
                return False
            AA = 'ACGT'
            AA_list = [a1 + a2 + a3 + a4 for a1 in AA for a2 in AA for a3 in AA for a4 in AA]
            AA_dict = {}
            for i in range(len(AA_list)):
                AA_dict[AA_list[i]] = [0] * 256
                AA_dict[AA_list[i]][i] = 1

            encodings = []
            header = ['SampleName', 'label']
            for i in range(1, (len(self.fasta_list[0][1]) - 3) * 256 + 1):
                header.append('BINARY.F' + str(i))
            encodings.append(header)

            for i in self.fasta_list:
                name, sequence, label = i[0], i[1], i[2]
                code = [name, label]
                for j in range(len(sequence) - 3):
                    code += AA_dict.get(sequence[j: j+4], [0]*256)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Z_curve_9bit(self):
        try:
            encodings = []
            header = ['SampleName', 'label']
            for pos in range(1, 4):
                for elem in ['x', 'y', 'z']:
                    header.append('Pos_%s.%s' %(pos, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                pos1_dict = {}
                pos2_dict = {}
                pos3_dict = {}
                for i in range(len(sequence)):
                    if (i+1) % 3 == 1:
                        if sequence[i] in pos1_dict:
                            pos1_dict[sequence[i]] += 1
                        else:
                            pos1_dict[sequence[i]] = 1
                    elif (i+1) % 3 == 2:
                        if sequence[i] in pos2_dict:
                            pos2_dict[sequence[i]] += 1
                        else:
                            pos2_dict[sequence[i]] = 1
                    elif (i+1) % 3 == 0:
                        if sequence[i] in pos3_dict:
                            pos3_dict[sequence[i]] += 1
                        else:
                            pos3_dict[sequence[i]] = 1

                code += [
                    (pos1_dict.get('A', 0) + pos1_dict.get('G', 0) - pos1_dict.get('C', 0) - pos1_dict.get('T', 0)) / len(sequence), # x
                    (pos1_dict.get('A', 0) + pos1_dict.get('C', 0) - pos1_dict.get('G', 0) - pos1_dict.get('T', 0)) / len(sequence), # y
                    (pos1_dict.get('A', 0) + pos1_dict.get('T', 0) - pos1_dict.get('G', 0) - pos1_dict.get('C', 0)) / len(sequence)  # z
                    ]
                code += [
                    (pos2_dict.get('A', 0) + pos2_dict.get('G', 0) - pos2_dict.get('C', 0) - pos2_dict.get('T', 0)) / len(sequence),
                    (pos2_dict.get('A', 0) + pos2_dict.get('C', 0) - pos2_dict.get('G', 0) - pos2_dict.get('T', 0)) / len(sequence),
                    (pos2_dict.get('A', 0) + pos2_dict.get('T', 0) - pos2_dict.get('G', 0) - pos2_dict.get('C', 0)) / len(sequence)
                    ]
                code += [
                    (pos3_dict.get('A', 0) + pos3_dict.get('G', 0) - pos3_dict.get('C', 0) - pos3_dict.get('T', 0)) / len(sequence),
                    (pos3_dict.get('A', 0) + pos3_dict.get('C', 0) - pos3_dict.get('G', 0) - pos3_dict.get('T', 0)) / len(sequence),
                    (pos3_dict.get('A', 0) + pos3_dict.get('T', 0) - pos3_dict.get('G', 0) - pos3_dict.get('C', 0)) / len(sequence)
                    ]
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Z_curve_12bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for base in NN:
                for elem in ['x', 'y', 'z']:
                    header.append('%s.%s' %(base, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                pos_dict = {}
                for i in range(len(sequence) - 1):
                    if sequence[i: i+2] in pos_dict:
                        pos_dict[sequence[i: i+2]] += 1
                    else:
                        pos_dict[sequence[i: i+2]] = 1

                for base in NN:
                    code += [
                        (pos_dict.get('%sA' %base, 0) + pos_dict.get('%sG' %base, 0) - pos_dict.get('%sC' %base, 0) - pos_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # x
                        (pos_dict.get('%sA' %base, 0) + pos_dict.get('%sC' %base, 0) - pos_dict.get('%sG' %base, 0) - pos_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # y
                        (pos_dict.get('%sA' %base, 0) + pos_dict.get('%sT' %base, 0) - pos_dict.get('%sG' %base, 0) - pos_dict.get('%sC' %base, 0)) / (len(sequence) - 1)  # z
                        ]

                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Z_curve_36bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']

            for base in NN:
                for pos in range(1, 4):
                    for elem in ['x', 'y', 'z']:
                        header.append('Pos_%s_%s.%s' %(pos, base, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                pos1_dict = {}
                pos2_dict = {}
                pos3_dict = {}
                for i in range(len(sequence) - 1):
                    if (i+1) % 3 == 1:
                        if sequence[i: i+2] in pos1_dict:
                            pos1_dict[sequence[i: i+2]] += 1
                        else:
                            pos1_dict[sequence[i: i+2]] = 1
                    elif (i+1) % 3 == 2:
                        if sequence[i: i+2] in pos2_dict:
                            pos2_dict[sequence[i: i+2]] += 1
                        else:
                            pos2_dict[sequence[i: i+2]] = 1
                    elif (i+1) % 3 == 0:
                        if sequence[i: i+2] in pos3_dict:
                            pos3_dict[sequence[i: i+2]] += 1
                        else:
                            pos3_dict[sequence[i: i+2]] = 1

                for base in NN:
                    code += [
                        (pos1_dict.get('%sA' %base, 0) + pos1_dict.get('%sG' %base, 0) - pos1_dict.get('%sC' %base, 0) - pos1_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # x
                        (pos1_dict.get('%sA' %base, 0) + pos1_dict.get('%sC' %base, 0) - pos1_dict.get('%sG' %base, 0) - pos1_dict.get('%sT' %base, 0)) / (len(sequence) - 1), # y
                        (pos1_dict.get('%sA' %base, 0) + pos1_dict.get('%sT' %base, 0) - pos1_dict.get('%sG' %base, 0) - pos1_dict.get('%sC' %base, 0)) / (len(sequence) - 1)  # z
                        ]
                    code += [
                        (pos2_dict.get('%sA' %base, 0) + pos2_dict.get('%sG' %base, 0) - pos2_dict.get('%sC' %base, 0) - pos2_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos2_dict.get('%sA' %base, 0) + pos2_dict.get('%sC' %base, 0) - pos2_dict.get('%sG' %base, 0) - pos2_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos2_dict.get('%sA' %base, 0) + pos2_dict.get('%sT' %base, 0) - pos2_dict.get('%sG' %base, 0) - pos2_dict.get('%sC' %base, 0)) / (len(sequence) - 1)
                        ]
                    code += [
                        (pos3_dict.get('%sA' %base, 0) + pos3_dict.get('%sG' %base, 0) - pos3_dict.get('%sC' %base, 0) - pos3_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos3_dict.get('%sA' %base, 0) + pos3_dict.get('%sC' %base, 0) - pos3_dict.get('%sG' %base, 0) - pos3_dict.get('%sT' %base, 0)) / (len(sequence) - 1),
                        (pos3_dict.get('%sA' %base, 0) + pos3_dict.get('%sT' %base, 0) - pos3_dict.get('%sG' %base, 0) - pos3_dict.get('%sC' %base, 0)) / (len(sequence) - 1)
                        ]
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Z_curve_48bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for base in NN:
                for base1 in NN:
                    for elem in ['x', 'y', 'z']:
                        header.append('%s%s.%s' %(base, base1, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                pos_dict = {}
                for i in range(len(sequence) - 2):
                    if sequence[i: i+3] in pos_dict:
                        pos_dict[sequence[i: i+3]] += 1
                    else:
                        pos_dict[sequence[i: i+3]] = 1

                for base in NN:
                    for base1 in NN:
                        code += [
                            (pos_dict.get('%s%sA' %(base, base1), 0) + pos_dict.get('%s%sG' %(base, base1), 0) - pos_dict.get('%s%sC' %(base, base1), 0) - pos_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos_dict.get('%s%sA' %(base, base1), 0) + pos_dict.get('%s%sC' %(base, base1), 0) - pos_dict.get('%s%sG' %(base, base1), 0) - pos_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos_dict.get('%s%sA' %(base, base1), 0) + pos_dict.get('%s%sT' %(base, base1), 0) - pos_dict.get('%s%sG' %(base, base1), 0) - pos_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Z_curve_144bit(self):
        try:
            NN = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']

            for base in NN:
                for base1 in NN:
                    for pos in range(1, 4):
                        for elem in ['x', 'y', 'z']:
                            header.append('Pos_%s_%s%s.%s' %(pos, base, base1, elem))
            encodings.append(header)

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                pos1_dict = {}
                pos2_dict = {}
                pos3_dict = {}
                for i in range(len(sequence) - 2):
                    if (i+1) % 3 == 1:
                        if sequence[i: i+3] in pos1_dict:
                            pos1_dict[sequence[i: i+3]] += 1
                        else:
                            pos1_dict[sequence[i: i+3]] = 1
                    elif (i+1) % 3 == 2:
                        if sequence[i: i+3] in pos2_dict:
                            pos2_dict[sequence[i: i+3]] += 1
                        else:
                            pos2_dict[sequence[i: i+3]] = 1
                    elif (i+1) % 3 == 0:
                        if sequence[i: i+3] in pos3_dict:
                            pos3_dict[sequence[i: i+3]] += 1
                        else:
                            pos3_dict[sequence[i: i+3]] = 1

                for base in NN:
                    for base1 in NN:
                        code += [
                            (pos1_dict.get('%s%sA' %(base, base1), 0) + pos1_dict.get('%s%sG' %(base, base1), 0) - pos1_dict.get('%s%sC' %(base, base1), 0) - pos1_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos1_dict.get('%s%sA' %(base, base1), 0) + pos1_dict.get('%s%sC' %(base, base1), 0) - pos1_dict.get('%s%sG' %(base, base1), 0) - pos1_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos1_dict.get('%s%sA' %(base, base1), 0) + pos1_dict.get('%s%sT' %(base, base1), 0) - pos1_dict.get('%s%sG' %(base, base1), 0) - pos1_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                        code += [
                            (pos2_dict.get('%s%sA' %(base, base1), 0) + pos2_dict.get('%s%sG' %(base, base1), 0) - pos2_dict.get('%s%sC' %(base, base1), 0) - pos2_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos2_dict.get('%s%sA' %(base, base1), 0) + pos2_dict.get('%s%sC' %(base, base1), 0) - pos2_dict.get('%s%sG' %(base, base1), 0) - pos2_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos2_dict.get('%s%sA' %(base, base1), 0) + pos2_dict.get('%s%sT' %(base, base1), 0) - pos2_dict.get('%s%sG' %(base, base1), 0) - pos2_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                        code += [
                            (pos3_dict.get('%s%sA' %(base, base1), 0) + pos3_dict.get('%s%sG' %(base, base1), 0) - pos3_dict.get('%s%sC' %(base, base1), 0) - pos3_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # x
                            (pos3_dict.get('%s%sA' %(base, base1), 0) + pos3_dict.get('%s%sC' %(base, base1), 0) - pos3_dict.get('%s%sG' %(base, base1), 0) - pos3_dict.get('%s%sT' %(base, base1), 0)) / (len(sequence) - 2), # y
                            (pos3_dict.get('%s%sA' %(base, base1), 0) + pos3_dict.get('%s%sT' %(base, base1), 0) - pos3_dict.get('%s%sG' %(base, base1), 0) - pos3_dict.get('%s%sC' %(base, base1), 0)) / (len(sequence) - 2)  # z
                            ]
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def NMBroto(self):
        try:
            self.encoding_array = np.array([])

            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.kw['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.kw['Di-RNA-Phychem'].split(';')
            try:
                with open(os.path.join('util', file_name), 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False
            nlag = self.kw['nlag']

            # value normalization
            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

            base = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for p_name in property_name:
                for d in range(1, nlag + 1):
                    header.append(p_name + '.lag' + str(d))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in base for aa2 in base]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                N = len(sequence) - 1
                for p_name in property_name:
                    for d in range(1, nlag + 1):
                        try:
                            if N > nlag:
                                atsd = sum([float(property_dict[p_name][AADict[sequence[j: j+2]]]) * float(property_dict[p_name][AADict[sequence[j+d: j+d+2]]]) for j in range(N-d)]) / (N - d)
                            else:
                                atsd = 0
                        except Exception as e:
                            atsd = 0
                        code.append(atsd)
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Moran(self):
        try:
            self.encoding_array = np.array([])

            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.kw['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.kw['Di-RNA-Phychem'].split(';')
            try:
                with open(os.path.join('util', file_name), 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False
            nlag = self.kw['nlag']

            # value normalization
            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

            base = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for p_name in property_name:
                for d in range(1, nlag + 1):
                    header.append(p_name + '.lag' + str(d))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in base for aa2 in base]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                N = len(sequence) - 1
                for p_name in property_name:
                    pmean = sum([property_dict[p_name][AADict[sequence[i: i+2]]] for i in range(N)]) / N
                    for d in range(1, nlag + 1):
                        try:
                            Idup = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - pmean) * (property_dict[p_name][AADict[sequence[j+d: j+d+2]]] - pmean) for j in range(N-d)]) / (N - d)
                            Iddown = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - pmean) ** 2 for j in range(N-d)]) / N
                            code.append(Idup/Iddown)
                        except Exception as e:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def Geary(self):
        try:
            self.encoding_array = np.array([])

            if self.sequence_type == 'DNA':
                file_name = 'didnaPhyche.data'
                property_name = self.kw['Di-DNA-Phychem'].split(';')
            else:
                file_name = 'dirnaPhyche.data'
                property_name = self.kw['Di-RNA-Phychem'].split(';')
            try:
                with open(os.path.join('util', file_name), 'rb') as handle:
                    property_dict = pickle.load(handle)
            except Exception as e:
                self.error_msg = 'Could not find the physicochemical properties file.'
                return False
            nlag = self.kw['nlag']

            # value normalization
            for p_name in property_name:
                tmp = np.array(property_dict[p_name], dtype=float)
                pmean = np.average(tmp)
                pstd = np.std(tmp)
                property_dict[p_name] = [(elem - pmean) / pstd for elem in tmp]

            base = 'ACGT'
            encodings = []
            header = ['SampleName', 'label']
            for p_name in property_name:
                for d in range(1, nlag + 1):
                    header.append(p_name + '.lag' + str(d))
            encodings.append(header)

            AADict = {}
            AA_list = [aa1 + aa2 for aa1 in base for aa2 in base]
            for i in range(len(AA_list)):
                AADict[AA_list[i]] = i

            for elem in self.fasta_list:
                name, sequence, label = elem[0], re.sub('-', '', elem[1]), elem[2]
                code = [name, label]
                N = len(sequence) - 1
                for p_name in property_name:
                    pmean = sum([property_dict[p_name][AADict[sequence[i: i + 2]]] for i in range(N)]) / N
                    for d in range(1, nlag + 1):
                        try:
                            Cdup = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - property_dict[p_name][AADict[sequence[j+d: j+d+2]]]) ** 2 for j in range(N - d)]) / (2 * (N - d))
                            Cddown = sum([(property_dict[p_name][AADict[sequence[j: j+2]]] - pmean) ** 2 for j in range(N - d)]) / (N - 1)
                            code.append(Cdup / Cddown)
                        except Exception as e:
                            code.append(0)
                encodings.append(code)

            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def generatePropertyPairs(self, myPropertyName):
        pairs = []
        for i in range(len(myPropertyName)):
            for j in range(i + 1, len(myPropertyName)):
                pairs.append([myPropertyName[i], myPropertyName[j]])
                pairs.append([myPropertyName[j], myPropertyName[i]])
        return pairs

    def make_ac_vector(self, myPropertyName, myPropertyValue, kmer):
        try:
            fastas = self.fasta_list
            lag = self.kw['nlag']

            encodings = []
            myIndex = self.myDiIndex if kmer == 2 else self.myTriIndex
            header = ['SampleName', 'label']
            for p in myPropertyName:
                for l in range(1, lag + 1):
                    header.append('%s.lag%d' % (p, l))
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]

                for p in myPropertyName:
                    meanValue = 0
                    for j in range(len(sequence) - kmer + 1):
                        meanValue = meanValue + float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]])
                    meanValue = meanValue / (len(sequence) - kmer + 1)

                    for l in range(1, lag + 1):
                        acValue = 0
                        for j in range(len(sequence) - kmer - l + 1):
                            acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]]) - meanValue) * (
                                    float(myPropertyValue[p][myIndex[sequence[j + l:j + l + kmer]]]) - meanValue)
                        acValue = acValue / (len(sequence) - kmer - l + 1)
                        code.append(acValue)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def make_cc_vector(self, myPropertyName, myPropertyValue, kmer):
        try:
            fastas = self.fasta_list
            lag = self.kw['nlag']

            encodings = []
            myIndex = self.myDiIndex if kmer == 2 else self.myTriIndex
            if len(myPropertyName) < 2:
                self.error_msg = 'two or more property are needed for cross covariance (i.e. DCC and TCC) descriptors'
                return False
            propertyPairs = self.generatePropertyPairs(myPropertyName)
            header = ['SampleName', 'label'] + [n[0] + '-' + n[1] + '-lag.' + str(l) for n in propertyPairs for l in
                                                range(1, lag + 1)]
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]

                for pair in propertyPairs:
                    meanP1 = 0
                    meanP2 = 0
                    for j in range(len(sequence) - kmer + 1):
                        meanP1 = meanP1 + float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]])
                        meanP2 = meanP2 + float(myPropertyValue[pair[1]][myIndex[sequence[j: j + kmer]]])
                    meanP1 = meanP1 / (len(sequence) - kmer + 1)
                    meanP2 = meanP2 / (len(sequence) - kmer + 1)

                    for l in range(1, lag + 1):
                        ccValue = 0
                        for j in range(len(sequence) - kmer - l + 1):
                            ccValue = ccValue + (
                                    float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]]) - meanP1) * (
                                            float(
                                                myPropertyValue[pair[1]][myIndex[sequence[j + l:j + l + kmer]]]) - meanP2)
                        ccValue = ccValue / (len(sequence) - kmer - l + 1)
                        code.append(ccValue)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def make_acc_vector(self, myPropertyName, myPropertyValue, kmer):
        try:
            fastas = self.fasta_list
            lag = self.kw['nlag']

            encodings = []
            myIndex = self.myDiIndex if kmer == 2 else self.myTriIndex
            if len(myPropertyName) < 2:
                self.error_msg = 'two or more property are needed for cross covariance (i.e. DCC and TCC) descriptors'
                return False

            header = ['SampleName', 'label']
            for p in myPropertyName:
                for l in range(1, lag + 1):
                    header.append('%s.lag%d' % (p, l))
            propertyPairs = self.generatePropertyPairs(myPropertyName)
            header = header + [n[0] + '-' + n[1] + '-lag.' + str(l) for n in propertyPairs for l in range(1, lag + 1)]
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                ## Auto covariance
                for p in myPropertyName:
                    meanValue = 0
                    # for j in range(len(sequence) - kmer):
                    for j in range(len(sequence) - kmer + 1):
                        meanValue = meanValue + float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]])
                    # meanValue = meanValue / (len(sequence) - kmer)
                    meanValue = meanValue / (len(sequence) - kmer + 1)

                    for l in range(1, lag + 1):
                        acValue = 0
                        for j in range(len(sequence) - kmer - l + 1):
                            acValue = acValue + (float(myPropertyValue[p][myIndex[sequence[j: j + kmer]]]) - meanValue) * (
                                    float(myPropertyValue[p][myIndex[sequence[j + l:j + l + kmer]]]) - meanValue)
                        acValue = acValue / (len(sequence) - kmer - l + 1)
                        # print(acValue)
                        code.append(acValue)

                ## Cross covariance
                for pair in propertyPairs:
                    meanP1 = 0
                    meanP2 = 0
                    for j in range(len(sequence) - kmer + 1):
                        meanP1 = meanP1 + float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]])
                        meanP2 = meanP2 + float(myPropertyValue[pair[1]][myIndex[sequence[j: j + kmer]]])
                    meanP1 = meanP1 / (len(sequence) - kmer + 1)
                    meanP2 = meanP2 / (len(sequence) - kmer + 1)

                    for l in range(1, lag + 1):
                        ccValue = 0
                        for j in range(len(sequence) - kmer - l + 1):
                            ccValue = ccValue + (
                                    float(myPropertyValue[pair[0]][myIndex[sequence[j: j + kmer]]]) - meanP1) * (
                                            float(
                                                myPropertyValue[pair[1]][myIndex[sequence[j + l:j + l + kmer]]]) - meanP2)
                        ccValue = ccValue / (len(sequence) - kmer - l + 1)
                        code.append(ccValue)
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def get_kmer_frequency(self, sequence, kmer):
        baseSymbol = 'ACGT'
        myFrequency = {}
        for pep in [''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))]:
            myFrequency[pep] = 0
        for i in range(len(sequence) - kmer + 1):
            myFrequency[sequence[i: i + kmer]] = myFrequency[sequence[i: i + kmer]] + 1
        for key in myFrequency:
            myFrequency[key] = myFrequency[key] / (len(sequence) - kmer + 1)
        return myFrequency

    def correlationFunction(self, pepA, pepB, myIndex, myPropertyName, myPropertyValue):
        CC = 0
        for p in myPropertyName:
            CC = CC + (float(myPropertyValue[p][myIndex[pepA]]) - float(myPropertyValue[p][myIndex[pepB]])) ** 2
        return CC / len(myPropertyName)

    def correlationFunction_type2(self, pepA, pepB, myIndex, myPropertyName, myPropertyValue):
        CC = 0
        for p in myPropertyName:
            CC = CC + float(myPropertyValue[p][myIndex[pepA]]) * float(myPropertyValue[p][myIndex[pepB]])
        return CC

    def get_theta_array(self, myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
        thetaArray = []
        for tmpLamada in range(lamadaValue):
            theta = 0
            for i in range(len(sequence) - tmpLamada - kmer):
                theta = theta + self.correlationFunction(sequence[i:i + kmer],
                                                         sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer], myIndex,
                                                         myPropertyName, myPropertyValue)
            thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
        return thetaArray

    def get_theta_array_type2(self, myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
        thetaArray = []
        for tmpLamada in range(lamadaValue):
            for p in myPropertyName:
                theta = 0
                for i in range(len(sequence) - tmpLamada - kmer):
                    theta = theta + self.correlationFunction_type2(sequence[i:i + kmer],
                                                                   sequence[
                                                                   i + tmpLamada + 1: i + tmpLamada + 1 + kmer],
                                                                   myIndex,
                                                                   [p], myPropertyValue)
                thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
        return thetaArray

    def PseDNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.kw['lambdaValue']
            weight = self.kw['weight']
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName', 'label']
            for pair in sorted(myIndex):
                header.append(pair)
            for k in range(1, lamadaValue + 1):
                header.append('lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                dipeptideFrequency = self.get_kmer_frequency(sequence, 2)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pair in sorted(myIndex.keys()):
                    code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
                for k in range(17, 16 + lamadaValue + 1):
                    code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def PCPseDNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.kw['lambdaValue']
            weight = self.kw['weight']
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName', 'label']
            for pair in sorted(myIndex):
                header.append(pair)
            for k in range(1, lamadaValue + 1):
                header.append('lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                dipeptideFrequency = self.get_kmer_frequency(sequence, 2)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pair in sorted(myIndex.keys()):
                    code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
                for k in range(17, 16 + lamadaValue + 1):
                    code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def PCPseTNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.kw['lambdaValue']
            weight = self.kw['weight']
            encodings = []
            myIndex = self.myTriIndex

            header = ['SampleName', 'label']
            for tripeptide in sorted(myIndex):
                header.append(tripeptide)
            for k in range(1, lamadaValue + 1):
                header.append('lamada_' + str(k))
            encodings.append(header)

            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                tripeptideFrequency = self.get_kmer_frequency(sequence, 3)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 3)
                for pep in sorted(myIndex.keys()):
                    code.append(tripeptideFrequency[pep] / (1 + weight * sum(thetaArray)))
                for k in range(65, 64 + lamadaValue + 1):
                    code.append((weight * thetaArray[k - 65]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def SCPseDNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.kw['lambdaValue']
            weight = self.kw['weight']
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName', 'label']
            for pair in sorted(myIndex):
                header.append(pair)
            for k in range(1, lamadaValue * len(myPropertyName) + 1):
                header.append('lamada_' + str(k))

            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                dipeptideFrequency = self.get_kmer_frequency(sequence, 2)
                thetaArray = self.get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pair in sorted(myIndex.keys()):
                    code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
                for k in range(17, 16 + lamadaValue * len(myPropertyName) + 1):
                    code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def SCPseTNC(self, myPropertyName, myPropertyValue):
        try:
            fastas = self.fasta_list
            lamadaValue = self.kw['lambdaValue']
            weight = self.kw['weight']
            encodings = []
            myIndex = self.myTriIndex
            header = ['SampleName', 'label']
            for pep in sorted(myIndex):
                header.append(pep)
            for k in range(1, lamadaValue * len(myPropertyName) + 1):
                header.append('lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                tripeptideFrequency = self.get_kmer_frequency(sequence, 3)
                thetaArray = self.get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 3)
                for pep in sorted(myIndex.keys()):
                    code.append(tripeptideFrequency[pep] / (1 + weight * sum(thetaArray)))
                for k in range(65, 64 + lamadaValue * len(myPropertyName) + 1):
                    code.append((weight * thetaArray[k - 65]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    def PseKNC(self, myPropertyName, myPropertyValue):
        try:
            baseSymbol = 'ACGT'
            fastas = self.fasta_list
            lamadaValue = self.kw['lambdaValue']
            weight = self.kw['weight']
            kmer = self.kw['kmer']
            encodings = []
            myIndex = self.myDiIndex
            header = ['SampleName', 'label']
            header = header + sorted([''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))])
            for k in range(1, lamadaValue + 1):
                header.append('lamada_' + str(k))
            encodings.append(header)
            for i in fastas:
                name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
                code = [name, label]
                kmerFreauency = self.get_kmer_frequency(sequence, kmer)
                thetaArray = self.get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
                for pep in sorted([''.join(j) for j in list(itertools.product(baseSymbol, repeat=kmer))]):
                    code.append(kmerFreauency[pep] / (1 + weight * sum(thetaArray)))
                for k in range(len(baseSymbol) ** kmer + 1, len(baseSymbol) ** kmer + lamadaValue + 1):
                    code.append((weight * thetaArray[k - (len(baseSymbol) ** kmer + 1)]) / (1 + weight * sum(thetaArray)))
                encodings.append(code)
            self.encoding_array = np.array([])
            self.encoding_array = np.array(encodings, dtype=str)
            self.column = self.encoding_array.shape[1]
            self.row = self.encoding_array.shape[0] - 1
            del encodings
            if self.encoding_array.shape[0] > 1:
                return True
            else:
                return False
        except Exception as e:
            self.error_msg = str(e)
            return False

    """ functions """

    def get_header(self):
        if self.encoding_array.shape[0] > 1:
            return list(self.encoding_array[0])
        else:
            return None

    def get_data(self):
        if self.encoding_array.shape[0] >= 2:
            return self.encoding_array[1:, :]
        else:
            return None

    def save_descriptor(self, file_name):
        try:
            if self.row > 1 and self.column > 2:
                if file_name.endswith(".tsv"):
                    np.savetxt(file_name, self.encoding_array[1:, 1:], fmt='%s', delimiter='\t')
                    return True

                if file_name.endswith(".csv"):
                    np.savetxt(file_name, self.encoding_array[1:, 1:], fmt='%s', delimiter=',')
                    return True
                
                if file_name.endswith(".tsv1"):
                    np.savetxt(file_name, self.encoding_array, fmt='%s', delimiter='\t')
                    return Ture

                if file_name.endswith(".svm"):
                    with open(file_name, 'w') as f:
                        for line in self.encoding_array[1:]:
                            line = line[1:]
                            f.write('%s' % line[0])
                            for i in range(1, len(line)):
                                f.write('  %d:%s' % (i, line[i]))
                            f.write('\n')
                    return True

                if file_name.endswith(".arff"):
                    with open(file_name, 'w') as f:
                        f.write('@relation descriptor\n\n')
                        for i in range(1, len(self.encoding_array[0][2:]) + 1):
                            f.write('@attribute f.%d numeric\n' % i)
                        f.write('@attribute play {yes, no}\n\n')
                        f.write('@data\n')
                        for line in self.encoding_array[1:]:
                            line = line[1:]
                            for fea in line[1:]:
                                f.write('%s,' % fea)
                            if line[0] == '1':
                                f.write('yes\n')
                            else:
                                f.write('no\n')
        except Exception as e:
            return False
