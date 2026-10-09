import sys
import numpy as np
from Bio import SeqIO
from joblib import Parallel, delayed


from FExt.util import FeatureLib

def ReadFasta(inputFasta):
    seqIDs, sequences = [], []
    with open(inputFasta, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            seqIDs.append(record.id)
            sequences.append(record.seq)
    return seqIDs, sequences

def validate_dna_dict(sequences):
    valid_keys = {'A', 'C', 'T', 'G'}
    for seq in sequences:
        if not set(seq).issubset(valid_keys):
            return False
    return True

def calculate_descriptor(desc_fasta_file, desc_selected_descriptor, desc_seq_type, desc_default_para):
    try:
        descriptor = None

        if desc_fasta_file != '' and desc_selected_descriptor != '':
            descriptor = FeatureLib.Descriptor(desc_fasta_file, desc_default_para)
            if descriptor.error_msg == '' and descriptor.sequence_number > 0:
                if desc_selected_descriptor in ['DAC', 'TAC']:
                    my_property_name, my_property_value, my_kmer, ok = FeatureLib.check_acc_arguments(
                        desc_selected_descriptor, desc_seq_type, desc_default_para)
                    descriptor.make_ac_vector(my_property_name, my_property_value, my_kmer)
                elif desc_selected_descriptor in ['DCC', 'TCC']:
                    my_property_name, my_property_value, my_kmer, ok = FeatureLib.check_acc_arguments(
                        desc_selected_descriptor, desc_seq_type, desc_default_para)
                    descriptor.make_cc_vector(my_property_name, my_property_value, my_kmer)
                elif desc_selected_descriptor in ['DACC', 'TACC']:
                    my_property_name, my_property_value, my_kmer, ok = FeatureLib.check_acc_arguments(
                        desc_selected_descriptor, desc_seq_type, desc_default_para)
                    descriptor.make_acc_vector(my_property_name, my_property_value, my_kmer)
                elif desc_selected_descriptor in ['PseDNC', 'PseKNC', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC']:
                    my_property_name, my_property_value, ok = FeatureLib.check_Pse_arguments(
                        desc_selected_descriptor, desc_seq_type, desc_default_para)
                    cmd = 'descriptor.' + desc_selected_descriptor + '(my_property_name, my_property_value)'
                    eval(cmd)
                else:
                    cmd = 'descriptor.' + desc_selected_descriptor + '()'
                    eval(cmd)
            return descriptor.encoding_array
        else:
            print('Warning: Please check your input!')

    except Exception as e:
        print("Error: ", e)

def reset_defaults():
	desc_default_para = {
		'sliding_window': 5,
		'kspace': 3,
		'props': ['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102', 'CHOC760101', 'BIGC670101', 'CHAM810101', 'DAYM780201'],
		'nlag': 3,
		'weight': 0.05,
		'lambdaValue': 10,
		'PseKRAAC_model': 'g-gap',
		'g-gap': 2,
		'k-tuple': 2,
		'RAAC_clust': 1,
		'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101',
		'kmer': 2,
		'mismatch': 1,
		'delta': 0,
		'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',
		'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)',
		'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)',
		'distance': 0,
		'cp': 'cp(20)',
	}
	return desc_default_para

def MergeFeatures(features, featureOrder, resultPath):
	# dup_list = ['SampleName', 'label', 'lamada_1', 'lamada_2', 'lamada_3', 'lamada_4', 'lamada_5', 'lamada_6', 'lamada_7', 'lamada_8', 'lamada_9', 'lamada_10']
	df_list = []
	allFeatures = []

	for (fn, feature) in zip(featureOrder, features):
		print('processing :', fn)
		feature = feature[:, 2:]
		if fn in {'PseKNC3', 'PseKNC4', 'PseKNC5'}:
		    feature = feature[:, :-10]
		allFeatures.append(feature)

	allFeatures = np.hstack(allFeatures)[1:].astype(np.float64)
	print(f'allFeatures.shape {allFeatures.shape}')
	np.save(resultPath, allFeatures)

def GenFeatures(inputFasta, resultPath):
	desc_default_para = reset_defaults()

	feature_types = {
		"Kmer": {
			'kmer': [2, 3, 4, 5]
		},
		"PseKNC": {
			'kmer': [2, 3, 4, 5]
		},
		"PseEIIP": {},
		"DPCP": {
			'Di-DNA-Phychem': 'Base stacking;Protein induced deformability;B-DNA twist;Dinucleotide GC Content;A-philicity;Propeller twist;Duplex stability:(freeenergy);Duplex tability(disruptenergy);DNA denaturation;Bending stiffness;Protein DNA twist;Stabilising energy of Z-DNA;Aida_BA_transition;Breslauer_dG;Breslauer_dH;Breslauer_dS;Electron_interaction;Hartman_trans_free_energy;Helix-Coil_transition;Ivanov_BA_transition;Lisser_BZ_transition;Polar_interaction;SantaLucia_dG;SantaLucia_dH;SantaLucia_dS;Sarai_flexibility;Stability;Stacking_energy;Sugimoto_dG;Sugimoto_dH;Sugimoto_dS;Watson-Crick_interaction;Twist;Tilt;Roll;Shift;Slide;Rise;Clash Strength;Roll_roll;Twist stiffness;Tilt stiffness;Shift_rise;Adenine content;Direction;Twist_shift;Enthalpy1;Twist_twist;Roll_shift;Shift_slide;Shift2;Tilt3;Tilt1;Tilt4;Tilt2;Slide (DNA-protein complex)1;Tilt_shift;Twist_tilt;Twist (DNA-protein complex)1;Tilt_rise;Roll_rise;Stacking energy;Stacking energy1;Stacking energy2;Stacking energy3;Propeller Twist;Roll11;Rise (DNA-protein complex);Tilt_tilt;Roll4;Roll2;Roll3;Roll1;Minor Groove Size;GC content;Slide_slide;Enthalpy;Shift_shift;Slide stiffness;Melting Temperature1;Flexibility_slide;Minor Groove Distance;Rise (DNA-protein complex)1;Tilt (DNA-protein complex);Guanine content;Roll (DNA-protein complex)1;Entropy;Cytosine content;Major Groove Size;Twist_rise;Major Groove Distance;Twist (DNA-protein complex);Purine (AG) content;Melting Temperature;Free energy;Tilt_slide;Major Groove Width;Major Groove Depth;Wedge;Free energy8;Free energy6;Free energy7;Free energy4;Free energy5;Free energy2;Free energy3;Free energy1;Twist_roll;Shift (DNA-protein complex);Rise_rise;Flexibility_shift;Shift (DNA-protein complex)1;Thymine content;Slide_rise;Tilt_roll;Tip;Keto (GT) content;Roll stiffness;Minor Groove Width;Inclination;Entropy1;Roll_slide;Slide (DNA-protein complex);Twist1;Twist3;Twist2;Twist5;Twist4;Twist7;Twist6;Tilt (DNA-protein complex)1;Twist_slide;Minor Groove Depth;Roll (DNA-protein complex);Rise2;Persistance Length;Rise3;Shift stiffness;Probability contacting nucleosome core;Mobility to bend towards major groove;Slide3;Slide2;Slide1;Shift1;Bend;Rise1;Rise stiffness;Mobility to bend towards minor groove'
		},
		"TPCP": {},
		"Z_curve_48bit": {},
		"Z_curve_144bit": {}
	}

	print(f'inputFasta: {inputFasta}')
	seqIDs, sequences = ReadFasta(inputFasta)
	print(f'Sequences in inputFasta: {len(sequences)}')
	if validate_dna_dict(sequences) is False:
		raise RuntimeError('Invalid Fasta file')

	descriptorsTasks = {}
	for featype in feature_types.keys():
		if featype == "Kmer" or featype == "PseKNC":
			kmer_values = feature_types[featype]["kmer"]
			for kmer in kmer_values:

				desc_default_para['kmer'] = kmer
				descriptorsTasks[f"{featype}{kmer}"] = (inputFasta, featype, 'DNA', desc_default_para)
				desc_default_para = reset_defaults()
		else:
			if featype == "DPCP":
				desc_default_para['Di-DNA-Phychem'] = feature_types[featype]["Di-DNA-Phychem"]

			descriptorsTasks[f"{featype}"] = (inputFasta, featype, 'DNA', desc_default_para)
			desc_default_para = reset_defaults()

	featureOrder = ["Kmer2", "Kmer3", "Kmer4", "Kmer5", "PseKNC2", "PseKNC3", "PseKNC4", "PseKNC5", "PseEIIP", "DPCP", "TPCP", "Z_curve_144bit", "Z_curve_48bit"]
	descriptorsTasks = [descriptorsTasks[fn] for fn in featureOrder]
	descriptors = Parallel(n_jobs=-1)(delayed(calculate_descriptor)(desc_fasta_file, desc_selected_descriptor, desc_seq_type, desc_default_para)
	    for desc_fasta_file, desc_selected_descriptor, desc_seq_type, desc_default_para in descriptorsTasks)
	MergeFeatures(descriptors, featureOrder, resultPath)
