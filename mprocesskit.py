import os
from multiprocessing import Pool

def system_run_process(final_cmd):
    debug = True
    if debug:
        print('[system_run_process] ' + final_cmd)
    os.system(final_cmd)  # blocks
    print('[system_run_process] done')


def blocking_multiproc_launch_nprocs(cmd_str, process_count=1):
    final_cmd_strings = []
    for port in range(0, process_count):
        final_cmd_strings.append(cmd_str)

    print(final_cmd_strings)

    # this breaks very cleanly :)
    blocking_multiproc_launch(final_cmd_strings)


def blocking_multiproc_launch(final_cmd_strings,poolname='bmps_pool'):
    ''' input: a bunch of command strings.
    runs them all in parallel
    '''
    print('final_cmd_strings:',final_cmd_strings)
    pool = Pool(processes=len(final_cmd_strings))
    try:
        print('[{}] multiprocessing.pool.map begin...'.format(poolname))
        pool.map(system_run_process, final_cmd_strings) #blocks
        print('[{}] multiprocessing.pool.map ok'.format(poolname))
    except:
        print('[{}] multiprocessing.pool.close begin...'.format(poolname))
        pool.close()
        print('[{}] multiprocessing.pool.close ok'.format(poolname))
        pool.join()
        print('[{}] multiprocessing.pool.join ok'.format(poolname))

def _test():
    blocking_multiproc_launch(['sleep 4;echo "hi"','sleep 10;echo "hi"'])

import sys
import fire
junk=sys.argv.pop(0)
proc_arg=sys.argv.pop(0)
proc_count = int(proc_arg) # remove call to this prog.
if proc_count<1 or proc_count>20:
    print("Invalid proc count "+proc_arg)
    exit(1)
cmd = " ".join(sys.argv)
blocking_multiproc_launch_nprocs(cmd, process_count=proc_count)

# eg: python mprocesskit.py 3 python train_qrdqn_offline.py

