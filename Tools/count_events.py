#from ROOT import TFile                                                                                                                                                                                                                          
import uproot
import sys
import os
import concurrent.futures
import multiprocessing
import numpy as np
executor = concurrent.futures.ThreadPoolExecutor()

def worker(roo):
    t=uproot.open(roo)['Events']

    gw = t.array('genWeight', executor=executor, blocking=True)
    p_count= sum(gw>=0.0)
    n_count= sum(gw<0.0)
    return [p_count, n_count]
 
def process(sample, target_file, pool):
    with open(target_file) as roo_files:
        roo_list  = [line.strip('\n') for line in roo_files.readlines()]
        #
        result_list = np.array(pool.map(worker, ( roo for roo in roo_list)))
        p_count, n_count = sum(result_list[:,0]), sum(result_list[:,1])
        #
        print('\nFor sample: {}'.format(sample))
        print('p_events: {}'.format(p_count))
        print('n_events: {}\n'.format(n_count))

def main(cfg_file,get_s=None):
    with open(sys.argv[1]) as cfg_file:
        pool = multiprocessing.Pool(processes=8) # have to define this here
        for sample in cfg_file.readlines():
            if '#' is sample[0]: continue
            if sample.replace('\n','') is '' : continue
            if get_s is not None and get_s not in sample: continue
            s_name, s_dir, s_file, = [ _.strip() for _ in sample.split(',')[:3] ]
            if not os.path.exists(s_dir+'/'+s_file) : raise NameError(s_dir+'/'+s_file)

            process(s_name, s_dir+'/'+s_file, pool)

if __name__ == '__main__':

    cfg_file = sys.argv[1]
    sample=None
    if len(sys.argv) == 3: sample=sys.argv[2]
    if not os.path.exists(sys.argv[1]) : raise NameError(sys.argv[1])
    #
    main(cfg_file,sample)
    
