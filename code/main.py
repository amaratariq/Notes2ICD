from xmlrpc.client import boolean
import pandas as pd
import os
import ICD10_mapping
import ICD10_mapping_CDE
import embedding_generation

import argparse


header_data = '../data/'
header_res = './results/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notes_file', type=str, required=True, help='name of the excel sheet containin all notes in column titles NOTE_TEXT')
    parser.add_argument('--mode', type=str, required=True, help='partial matching or CDE')
    parser.add_argument('--RankedIndex', type=boolean, required=True, help='True or False, generate ranked retrievla index of groundtruth code, notes file must have DX_CODE_all column')
    
    
    args = parser.parse_args()
    
    print('reading notes')
    df = pd.read_excel(header_data+notes_file)
    print('parsing notes to extract final diagnoses recorded as free-text')
    df['EXT_PRIM_DX'] = df.NOTE_TEXT.apply(ICD10_mapping.extract_final_diagnosis)

    res_fname = 'notes_with_codes_based_on_'+'_'.join(mode.split())+'.xlsx'

    check_gt = RankedIndex and 'DX_CODE_all' in df.columns

    if mode == 'partial matching' or mode == 'full mathcing':
        print('{} based code matching'.format(mode))

        df_icd = pd.read_csv(header_data+'ICD10/MayoClinicProblemList_Epic_concatenated.csv')
        codes = df_icd.DX_CODE.values
        codes = np.array([c.upper() if type(c) is str else c for c in codes])
        icd10 = [process_desc(c) for c in df_icd.DIAGNOSIS_DESCRIPTION_all.values]
        print('ICD10 vocbaulary loaded and processed')
        sys.stdout.flush()

        
        if mode=='full matching':
            df_w_codes = ICD10_mapping.code_mapping(df, icd10, check_gt = check_gt,  full=True, verbose=False)
        else:
            df_w_codes = ICD10_mapping.code_mapping(df, icd10, check_gt = check_gt,  full=False, verbose=False)

        df_w_codes.to_csv(header_res+res_fname)

    elif mode == 'CDE':
        dct = embedding_generation.generate_embedding(df)
        df_w_codes = ICD10_mapping.code_mapping(df, dct)
        df_w_codes.to_csv(header_res+res_fname)
    else:
        print('mode {} not implemented'.format(mode))


if __name__ == "__main__":
    main()