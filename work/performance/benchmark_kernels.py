"""Separate-process CPU-kernel benchmarks with exact complete-state comparison.

Example: .venv/bin/python -m work.performance.benchmark_kernels --steps 180
Optional --reference points to an earlier source tree containing ecosystem/.
No profiler, champion publication or persistent simulation output is produced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

CASES = {
    'dqn': ('dqn.json', False), 'ppo': ('ppo.json', False),
    'a2c': ('a2c.json', False), 'reinforce': ('reinforce.json', False),
    'grpo': ('grpo.json', False), 'recurrent_ppo': ('recurrent_ppo.json', False),
    'qrdqn': ('qrdqn.json', False), 'maskable_ppo': ('maskable_ppo.json', False),
    'mappo': ('mappo.json', False), 'ppo_large': ('ppo_large_predator.json', False),
    'valley_record': ('enclosed_valleys.json', True),
}


def worker(args):
    if args.reference:sys.path.insert(0,str(args.reference.resolve()))
    from ecosystem.config import load
    from ecosystem.runner import Experiment
    from ecosystem.recording import Recorder
    try:
        from ecosystem.kernels import status
        backend = status()
    except ImportError:backend = dict(backend='reference', requested='reference')
    filename, record = CASES[args.case]
    config = load(Path('environments')/filename)
    config['seed'] = 42
    config['experiment'].update(runs=1,world_seed_policy='increment')
    config['world']['max_steps'] = args.steps
    started = time.perf_counter();experiment = Experiment(config)
    initialization_seconds = time.perf_counter()-started
    with tempfile.TemporaryDirectory(dir='.cache/native-build') as temporary:
        folder = Path(temporary)
        started = time.perf_counter()
        writer = Recorder(folder, config, 'train') if record else None
        if writer:writer.begin_run(experiment)
        while not experiment.done:
            experiment.step()
            if writer:writer.capture(experiment,force=experiment.finished_run)
        if writer:writer.end_run(experiment);writer.close()
        seconds = time.perf_counter()-started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = dict(case=args.case,backend=backend,seconds=seconds,
                      initialization_seconds=initialization_seconds,
                      peak_rss_mib=peak/(1024**2 if sys.platform=='darwin' else 1024),
                      output_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),
                      completed_steps=experiment.world.tick, updates=experiment.total_updates,
                      founder_population={s:c['initial_population'] for s,c in config['species'].items()},
                      algorithms={s:config['brains'][c['brain']]['type'] for s,c in config['species'].items()},
                      metrics=experiment.world.metrics())
        digest=hashlib.sha256()
        for part in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(experiment.checkpoint()):
            digest.update(part.encode())
        result['state_sha256']=digest.hexdigest()
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=180)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    p.add_argument('--output',type=Path,default=Path('.cache/native-build/benchmark.json'))
    p.add_argument('--reference',type=Path)
    p.add_argument('--backends',nargs='+',choices=['python','native'],default=['python','native'])
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--case',choices=list(CASES),help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.steps<=0 or args.repeats<=0:p.error('steps and repeats must be positive')
    Path('.cache/native-build').mkdir(parents=True,exist_ok=True)
    if args.worker:return worker(args)
    results=[];modes=list(dict.fromkeys(args.backends))+(['reference'] if args.reference else [])
    for case in args.cases:
        for repeat in range(args.repeats):
            order=modes[repeat%len(modes):]+modes[:repeat%len(modes)]
            for mode in order:
                command=[sys.executable,'-m','work.performance.benchmark_kernels','--worker',
                         '--case',case,'--steps',str(args.steps)]
                if mode=='reference':command+=['--reference',str(args.reference)]
                env=dict(os.environ,WTF_KERNELS='python' if mode=='reference' else mode,
                         PYTHONDONTWRITEBYTECODE='1')
                run=subprocess.run(command,env=env,capture_output=True,text=True)
                if run.returncode:
                    sys.stderr.write(run.stderr)
                    raise RuntimeError(f'Benchmark worker failed: {case}, {mode}, repeat {repeat+1}; completed samples retained')
                result=json.loads(run.stdout);result.update(mode=mode,repeat=repeat+1)
                results.append(result)
                print(f'{case} {repeat+1}/{args.repeats} {mode}: {result["seconds"]:.3f}s, {result["updates"]} updates',flush=True)
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(dict(status='running',samples=results),indent=2)+'\n')
    summary={}
    for case in args.cases:
        samples=[s for s in results if s['case']==case]
        if len({s['state_sha256'] for s in samples})!=1:
            raise RuntimeError(f'Complete state differs across backends/repeats: {case}; raw samples retained')
        row={mode:{key:statistics.median(s[key] for s in samples if s['mode']==mode)
                   for key in ('seconds','initialization_seconds','peak_rss_mib','output_bytes')}
             for mode in modes}
        if 'python' in row and 'native' in row:
            row['speedup_vs_python']=row['python']['seconds']/row['native']['seconds']
        if 'reference' in row and 'native' in row:
            row['speedup_vs_reference']=row['reference']['seconds']/row['native']['seconds']
        row['state_sha256']=samples[0]['state_sha256'];row['updates']=samples[0]['updates']
        row['completed_steps']=samples[0]['completed_steps'];summary[case]=row
    report=dict(status='complete',python=platform.python_version(),platform=platform.platform(),
                machine=platform.machine(),steps=args.steps,repeats=args.repeats,
                method='Alternating separate processes; seed 42; full preset populations; one episode; medians; RSS before semantic digest; initialization reported separately',
                summary=summary,samples=results)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
