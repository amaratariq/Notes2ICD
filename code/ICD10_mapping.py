import pandas as pd
import os
import numpy as np
import sys 
import pickle as pkl
import re
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from datetime import datetime, timedelta


header = '../../bucket/Amara/ED_Coding/Data/'
 
        



def remove_signature(txt):
    tags = ['i saw the']#, 'eletronically signed', 'signed by', 'i have personally', 'i personally', 'i have seen', 
            #'i have reviewed', '. i saw', '. i reviewed', 'please page', 'please refer', 'please contact', 'reviewed and summarized', 'documents from previous', 'please feel free' ,'. page ', '  page ', 'thank you for', 'please do not hesitate']
    if type(txt) is str and len(txt)>0:
        for t in tags:
            if t in txt.lower():
                idx = txt.lower().index(t)
                txt = txt[:idx]
        toks = txt.split('   ')
        toks = [t.strip() for t in toks if len(t.strip())>0]
        if len(toks)>1: #if there is only 1 token, do nto removae all
            last_tok = toks[-1]
            tags2 = ['l.i.c.s.w', 'm.s.w', 'p.t', 'p.a', 'm.d', 'd.o', 'l.g.s.w', 'resident', 'pa-c', 'dr.', 'l.s.w', 'm.b.b.s', 'm.b.', 'b.ch.', 'md', 'pgy-']
            if any(t in last_tok.lower() for t in tags2)==True:
                idx = txt.lower().index(last_tok.lower())
                txt = txt[:idx]
        if len(toks)>2: #if there are only 2 token, do nto removae all
            last_tok = toks[-2]   #sometine doctors name appear twice
            if last_tok in txt and any(t in last_tok.lower() for t in tags2)==True:
                idx = txt.lower().index(last_tok.lower())
                txt = txt[:idx]

    return txt

def extract_final_diagnosis(txt, verbose=False):
    tags = ['l.i.c.s.w', 'm.s.w', 'p.t', 'p.a', 'm.d', 'd.o', 'l.g.s.w', 'resident', 'pa-c', 'dr.', 'l.s.w', 'm.b.b.s', 'm.b.', 'b.ch.', 'as of', 'md', 'pgy-']
    rout_all = []
    if type(txt) is str:
        txt = txt.lower() #re.sub('\s+',' ',txt.lower())
        txt_all = txt.split('|')
        
        for txt in txt_all:
            eflag = False
            txt = remove_signature(txt)
            out = re.findall(r"final (diagnos(i|e)s|problem(s)):(.*?)(\s\s\s+|disposition|medication|impression|care handoff)(:|\s\s+)",txt) #tab or heading of next section
            midx = 3
            if verbose and len(out)>0:
                print('RULE1')
            if len(out)==0:
                out = re.findall(r"final (diagnos(i|e)s|problem(s)):(.*)(\s\s\s+)",txt) #check for start of new section with multiple spaces'
                midx=-2
                if len(out)==0:
                    out = re.findall(r"final (diagnos(i|e)s|problem(s)):(.*)",txt)  # check till the end of note
                    midx = -1
                if verbose and len(out)>0:
                    print('RULE2')
                if len(out)==0:
                    out = re.findall(r"final (diagnos(i|e)s|problem(s)) (is|are) (.*)(\s\s\s+)",txt) ##check for start of new section with multiple spaces
                    midx=-2
                    if len(out)==0:
                        out = re.findall(r"final (diagnos(i|e)s|problem(s)) (is|are) (.*)",txt) # check till the end of note
                        midx = -1
                    eflag = True
                    if verbose and len(out)>0:
                        print('RULE3')
#                 if len(out)==0:
#                     out = re.findall("impression(s)*:(\s+)*(.+?)\s\s\s\s+",txt.lower())
#                     midx = -1
#                 else:
#                     eflag=True
            if len(out)>0:
                if verbose:
                    print('Matched:\n'+out[0][midx], end='\n\n')
                rout = re.sub(' as of [0-9]*/[0-9]*/[0-9]* [0-9]*', '', out[0][midx])
                if verbose:
                    print('Date Removed:\n'+rout, end='\n\n')
                rout = rout.strip()
                if verbose:
                    print('Spaces removed:\n'+rout, end='\n\n')
#                 if eflag:
#                     rout = re.sub(' resident ', '', rout)
#                     rout = re.split("\s\s+", rout)[:-2]
#                 else:
                #rout = re.sub("[0123456789]", " ", rout)
                rout = re.split("\s\s+", rout)
                if verbose:
                    print('Split:\n',rout, end='\n\n')
                
                rout = [r.strip(r"[0123456789]*.").strip() for r in rout]
                if verbose:
                    print('Bullets Removed:\n',rout, end='\n\n')
                rout = [r for r in rout if len(re.sub(r'[^a-zA-Z]', '', r))>0]
                if verbose:
                    print('Non-alpha Removed:\n',rout, end='\n\n')
                rout = [r for r in rout if (any(t in r for t in tags)==True)==False]
                if verbose:
                    print('Signature Removed:\n',rout, end='\n\n')
                rout_all = rout_all+ rout
                    
    rout_all = list(np.unique(np.array(rout_all)))
    if len(rout_all)>0:
        return '|'.join(rout_all)


def process_desc(x):
    #tokenize description or diagnosis
    xx = x.split('|')
    out = []
    for x in xx:
        if type(x) is str:
            x = re.sub(r"[()-.]", "", x.lower())
            x = x.strip()
            out.append(word_tokenize(x))#x.split(' ')
    return out



def overlap_new_full(listA, listB):
    olap = 0
    if len(listB)>0: 
        margin = 2*len(listB)
        if margin<len(listA):
            for i in range(0, len(listA)-margin):
                o = 100*(len(set(listA[i:i+margin]) & set(listB)) / len(set(listB)|set(listA[i:i+margin]))) 
                if o>=100:
                    olap=o
        else:
            o = 100*(len(set(listA) & set(listB)) / len(set(listB) | set(listA)))
            if o>=100:
                    olap=o
    return olap


def overlap_new_partial(listA, listB):
    olap = 0
    if len(listB)>0: 
        margin = 2*len(listB)
        if margin<len(listA):
            for i in range(0, len(listA)-margin):
                o = 100*(len(set(listA[i:i+margin]) & set(listB)) / len(set(listB)|set(listA[i:i+margin]))) 
                if o>olap:
                    olap=o
        else:
            o = 100*(len(set(listA) & set(listB)) / len(set(listB) | set(listA)))
            if o>olap:
                    olap=o
    return olap

def map_to_ICD10_multi_gt_row_wise(diag, gt = None, icd10, check_gt = True, full=False, verbose=False):
    #icd10 - list of all icd10 descriptions processed through process_desc()
    #diag: | separated list of diagnosis
    #gt = | separated list of ground truth - None not allowed
    dct = {}
    
    match_idx = None
    top_matches = None
    if type(diag) is str and type(gt) is str:
        if check_gt:
            gt = list(gt.split('|'))
            match_idx = [1000000 for g in gt]
        
        diag_list = diag.split('|')    
        top_matches = []
        for d in diag_list:    
            d = d.replace('/', ' ')
            if verbose:
                print('DIAGNOSIS:\t', d)

            else:
                d = re.sub(r"[()-.]", "", d.lower())
                diag_toks = process_desc(d)[0] #process desc expects a list of diagnoses, if processing only one, pick the first list of toks
                if verbose:
                    print('TOKENS:\t', diag_toks)
            
                if full:
                    m = np.array([max([overlap_new_full(diag_toks, xx) for xx in x]) for x in icd10])
                else:
                    m = np.array([max([overlap_new_partial(diag_toks, xx) for xx in x]) for x in icd10])
                n = np.argsort(m)
                n = n[::-1]
                dct[d] = {'match': m[n], 'index':n}
                top_matches.append(codes[n[0]])
                if check_gt:
                    for g in gt:
                        if verbose:
                            print('checking ', g, ' in matches of ', d)
                        if g in codes:
                            match_idx[gt.index(g)] = min(match_idx[gt.index(g)], list(codes[n]).index(g))
                    if verbose:
                        print(match_idx)
        if check_gt:
            match_idx = [str(m) for m in match_idx]
            match_idx = '|'.join(match_idx)
        top_matches = '|'.join(top_matches)
    return match_idx, top_matches

def code_mapping(df, icd10, check_gt=False, full=False, verbose=False):
    for i, j in df.iterrows():
            diag = df.at[i, 'EXT_PRIM_DX']
            gt = df.at[i, 'DX_CODE_all']
            if type(gt) is str:
                gt = gt.upper()
            if mode == 'partial matching':
                matches, top_matches = map_to_ICD10_multi_gt_row_wise(diag, gt, check_gt = check_gt,  full=False, verbose=verbose)
            else:
                matches, top_matches = map_to_ICD10_multi_gt_row_wise(diag, gt, check_gt = check_gt,  full=True, verbose=verbose)
            if check_gt:
                df.at[i, 'MatchIdx'] = matches
            df.at[i, 'TopMatches'] = top_matches
    return df