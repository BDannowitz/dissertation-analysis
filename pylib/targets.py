#!/usr/bin/python

import pandas as pd
import numpy as np

Na = 6.02214129e23

target_dict = {'LH2': {'TargPos': 1,
                       'Density[g/cm^3]': 0.0708,
                       'LengthPerLayer[cm]': 50.8,
                       'Layers': 1,
                       'A': 1,
		       'Z': 1,
		       'NIL[g/cm^2]': 52.0,
		       'MassNo[g/mol]': 1.00794},
               'LD2': {'TargPos': 3,
                       'Density[g/cm^3]': 0.1634,
                       'LengthPerLayer[cm]': 50.8,
                       'Layers': 1,
                       'A': 2,
		       'Z': 1,
		       'NIL[g/cm^2]': 71.8,
		       'MassNo[g/mol]': 2.01410},
               'Fe': {'TargPos': 5,
                       'Density[g/cm^3]': 7.874,
                       'LengthPerLayer[cm]': 0.635,
                       'Layers': 3,
                       'A': 56,
		       'Z': 26,
		       'NIL[g/cm^2]': 132.1,
		       'MassNo[g/mol]': 55.845},
               'C': {'TargPos': 6,
                     'Density[g/cm^3]': 1.802,
                     'LengthPerLayer[cm]': 1.10744,
                     'Layers': 3,
                     'A': 12,
		     'Z': 6,
		     'NIL[g/cm^2]': 85.8,
		     'MassNo[g/mol]': 12.0107},
               'W': {'TargPos': 7,
                     'Density[g/cm^3]': 19.300,
                     'LengthPerLayer[cm]': 0.3175,
                     'Layers': 3,
                     'A': 184,
		     'Z': 74,
		     'NIL[g/cm^2]': 191.9,
		     'MassNo[g/mol]': 183.84},
               'None': {'TargPos': 4,
                        'Density[g/cm^3]': None,
                        'LengthPerLayer[cm]': None,
                        'Layers': None,
                        'A': None,
		        'Z': None,
		        'NIL[g/cm^2]': None,
		        'MassNo[g/mol]': None},
               'Empty': {'TargPos': 2,
                         'Density[g/cm^3]': None,
                         'LengthPerLayer[cm]': None,
                         'Layers': None,
                         'A': None,
		         'Z': None,
		         'NIL[g/cm^2]': None,
		         'MassNo[g/mol]': None}}

target_df = pd.DataFrame(target_dict).T

def calc_values(df):
    df['Length[cm]'] = df['Layers'] * df['LengthPerLayer[cm]']
    df['NIL/D[cm]'] = df['NIL[g/cm^2]'] / df['Density[g/cm^3]']
    df['Density[/cm^3]'] = np.divide(df['Density[g/cm^3]'], df['MassNo[g/mol]']) * Na
    df['Density[mol/cm^3]'] = np.divide(df['Density[g/cm^3]'], df['MassNo[g/mol]'])
    df['IntLengths'] = np.divide(df['Length[cm]'], df['NIL/D[cm]'])
    df['AttenFactor'] =  np.divide((1.0-np.exp(-1.0 * df['IntLengths'])),df['IntLengths'])
    df['AttenLength[cm]'] = df['AttenFactor'] * df['Length[cm]'] 
    df['Scale'] = df['Density[mol/cm^3]'] * df['AttenLength[cm]'] * df['MassNo[g/mol]']

    return df

purity = {57: {'upper': 0.954, 'nominal': 0.904, 'lower': 0.904},
          62: {'upper': 0.943, 'nominal': 0.913, 'lower': 0.883},
          67: {'upper': 0.983, 'nominal': 0.953, 'lower': 0.923},
	  'all': {'upper': 0.90, 'nominal': 0.939, 'lower': 0.98},
}

def update_for_contam(df, roadset, bound='nominal'):
    p = purity[roadset][bound]
    d_dat = df.ix['LD2'].copy()
    h_dat = df.ix['LH2'].copy()
    for x in ['Density[g/cm^3]', 'MassNo[g/mol]']:
        df.loc['LD2'][x] = p * d_dat[x] + (1.0-p) * h_dat[x]
    df.loc['LD2']['NIL[g/cm^2]'] = np.divide(np.multiply(d_dat['NIL[g/cm^2]'],h_dat['NIL[g/cm^2]']),
					     (1.0-p)*d_dat['NIL[g/cm^2]']+p*h_dat['NIL[g/cm^2]'])

    return df
   
target_roadset_dict = {'pure': calc_values(target_df.copy()),
		       57: calc_values( update_for_contam( target_df.copy(), 57, bound='nominal' )),
		       62: calc_values( update_for_contam( target_df.copy(), 62, bound='nominal' )),
		       67: calc_values( update_for_contam( target_df.copy(), 67, bound='nominal' )),
		       'all': calc_values( update_for_contam( target_df.copy(), 'all' ))}

#############

targets = ['LH2', 'Empty', 'LD2', 'None', 'Fe', 'C', 'W']
cols = ['TargetPos', 'Density[g/cm^3]', 'Length[cm]',
        'LengthPerLayer[cm]', 'NLayers', 'NIL[g/cm^2]',
        'NIL/D[cm]', 'MassNo[g/mol]']
mc_target_df = pd.DataFrame(np.full([7, 8],np.nan),
                                 index=targets,
                                 columns=cols)
# Names
mc_target_df.index.name = 'targetPos'
mc_target_df['TargetPos'] = range(1,8)

# Raw values
#mc_target_df['Density[g/cm^3]'] = [0.07066, np.NaN, 0.15707, np.NaN,
#                                   7.87001216419734, 1.802, 19.25]
mc_target_df['Density[g/cm^3]'] = [0.07066, np.NaN, 0.169, np.NaN,
                                   7.87001216419734, 1.802, 19.25]
mc_target_df['LengthPerLayer[cm]'] = [50.8, 50.8, 50.8, np.NaN,
                                      0.635, 1.10744, 0.3175]
mc_target_df['NLayers'] = [1, 1, 1, np.NaN, 3, 3, 3]
mc_target_df['NIL[g/cm^2]'] = [52.0, np.NaN, 71.8, np.NaN, 132.1, 85.8, 191.9]
mc_target_df['MassNo[g/mol]'] = [1.01, np.NaN, 2.014, np.NaN, 55.845,
                                 12.01, 183.84]

# Calculated values
mc_target_df['Length[cm]'] = mc_target_df['NLayers']*mc_target_df['LengthPerLayer[cm]']
mc_target_df['NIL/D[cm]'] = mc_target_df['NIL[g/cm^2]']/mc_target_df['Density[g/cm^3]']
#mc_target_df['NIL/D[cm]'] = [(52.0/0.071), np.NaN, (71.8/0.169), np.NaN, 44.504, 16.968, np.NaN]
mc_target_df['Density[/cm^3]'] = np.divide(mc_target_df['Density[g/cm^3]'],
                                           mc_target_df['MassNo[g/mol]']) * Na
mc_target_df['Density[mol/cm^3]'] = np.divide(mc_target_df['Density[g/cm^3]'],
                                              mc_target_df['MassNo[g/mol]'])
mc_target_df['IntLengths'] = np.divide(mc_target_df['Length[cm]'], mc_target_df['NIL/D[cm]'])
mc_target_df['AttenLength[cm]'] = mc_target_df['NIL/D[cm]'] * (1-np.exp(-1 * \
                                  np.divide(mc_target_df['Length[cm]'], mc_target_df['NIL/D[cm]'])))
mc_target_df['Scale'] = mc_target_df['Density[mol/cm^3]'] * mc_target_df['AttenLength[cm]'] * \
                        mc_target_df['MassNo[g/mol]']

mc_target_dict = {'LH2': {'targpos': 1,
                          'density': (0.0708001281743171 / 1.01) * Na,
                          'length': 49.077,
                          'layers': 1,
                          'A': 1,
                          'scale': ((0.0708001281743171 / 1.01) * Na * 49.077 * 1 * 1)},
                  'LD2': {'targpos': 3,
                          'density': (0.169000261213386 / 2.014) * Na,
                          'length': 48.002,
                          'layers': 1,
                          'A': 2,
                          'scale': ((0.169000261213386 / 2.014) * Na * 48.002 * 1 * 2)},
                  'Fe': {'targpos': 5,
                         'density': (7.87001216419734 / 55.845) * Na,
                         'length': 1.801,
                         'layers': 3,
                         'A': 56,
                         'scale': ((7.87001216419734 / 55.845) * Na * 1.801 * 56)},
                  'C': {'targpos': 6,
                        'density': (2.267 / 12.01) * Na,
                        'length': 3.209,
                        'layers': 3,
                        'A': 12,
                        'scale': ((2.267 / 12.01) * Na * 3.209 * 12)},
                  'W': {'targpos': 7,
                        'density': (19.25 / 183.84) * Na,
                        'length': 0.9083,
                        'layers': 3,
                        'A': 183,
                        'scale': ((19.25 / 183.84) * Na * 0.9083 * 183)},
                  'None': {'targpos': 4,
                           'density': None,
                           'length': None,
                           'layers': None,
                           'A': None,
                           'scale': None},
                  'Empty': {'targpos': 2,
                            'density': None,
                            'length': None,
                            'layers': None,
                            'A': None,
                            'scale': None}}




#target_df['Density[g/cm^3]'] = [0.0708, np.NaN, 0.1545, np.NaN,
#                                7.874, 1.802, 19.300]
#target_df['LengthPerLayer[cm]'] = [50.8, 50.8, 50.8, np.NaN,
#                                   0.635, 1.10744, 0.3175]
#target_df['NLayers'] = [1, 1, 1, np.NaN, 3, 3, 3]
##target_df['NIL[g/cm^2]'] = [52.0, np.NaN, 71.8, np.NaN, 132.1, 85.8, 191.9]
#target_df['NIL[g/cm^2]'] = [52.0, np.NaN, 69.3, np.NaN, 132.1, 85.8, 191.9]
##target_df['MassNo[g/mol]'] = [1.00794, np.NaN, 2.0141, np.NaN, 55.845,
##                                      12.0107, 183.84]
#target_df['MassNo[g/mol]'] = [1.00794, np.NaN, 1.91751, np.NaN, 55.845,
#                              12.0107, 183.84]
#cols = ['TargetPos', 'Density[g/cm^3]', 'Length[cm]',
#        'LengthPerLayer[cm]', 'NLayers', 'NIL[g/cm^2]',
#        'NIL/D[cm]', 'MassNo[g/mol]']
#targets = ['LH2', 'Empty', 'LD2', 'None', 'Fe', 'C', 'W']
#target_df = pd.DataFrame(np.full([7, 8],np.nan),
#                                 index=targets,
#                                 columns=cols)
## Names
#target_df.index.name = 'targetPos'
#target_df['TargetPos'] = range(1,8)
#
## Raw values
##target_df['Density[g/cm^3]'] = [0.0708, np.NaN, 0.1634, np.NaN,
##                                7.874, 1.802, 19.300]
#
## Calculated values
