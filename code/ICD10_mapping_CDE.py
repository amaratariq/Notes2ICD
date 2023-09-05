import pandas as pd
import numpy as np
import pickle as pkl
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

def code_mapping(df, dct_in, check_gt):

    print('input notes embeddings:\t', dct_in['mat'].shape, len(dct_in['keys']))

    dct_out = pkl.load(open('ICD10/ICD10_vocab_new_pipe_split_az_ed_new.pkl', 'rb'))
    sim_all = cos_sim(dct_in['mat'].reshape(-1, 768), dct_out['mat'].reshape(-1, 768)).squeeze()

    keys_out = np.array(dct_out['keys'])
    keys_out = np.array([k.upper() for k in keys_out])
    for i in range(len(df)):
        txt = df.at[df.index[i], 'EXT_PRIM_DX']
        if check_gt:
            dx_name = df.at[df.index[i], 'DX_NAME_all']#'PRIM_DX_NAME']
            dx_code = df.at[df.index[i], 'DX_CODE_all'].upper()#'PRIM_DX_ICD10'].upper()

        if type(txt) is str:
            keys = txt.split('|')

            if check_gt:
                codes = dx_code.split('|')
                matches = [1000000 for c in codes]
            top_matches = []
            for key in keys:
                idx = list(dct_in['keys']).index(key)

                sim_i = sim_all[idx,:]

                idx_sort = np.argsort(sim_i)
                idx_sort = idx_sort[::-1]

                top_codes = list(keys_out[idx_sort])
                top_matches.append(keys_out[idx_sort[0]])

                if check_gt:
                    matches = [min(matches[i], top_codes.index(codes[i].upper())) if codes[i].upper() in top_codes else matches[i] for i in range(len(codes))]
                
            if check_gt:
                matches = [str(m) for m in matches]
                matches = '|'.join(matches)
                df.at[df.index[i], 'MatchIdx_CSE'] = matches
            top_matches = '|'.join(top_matches)
            df.at[df.index[i], 'TopMatches'] = top_matches
        
    return df

    
