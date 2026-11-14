"""Alternate reference and optimized processes; record median local timings."""
import json
import statistics
import subprocess
import sys
from pathlib import Path

root=Path('work/performance')
for repeat in (2,3):
    for case in ('simulate','record','checkpoint'):
        versions=('before','after') if repeat==2 else ('after','before')
        for version in versions:
            command=[sys.executable]
            if version=='before':
                command+=['-c','import sys,runpy;sys.path.insert(0,"work/performance/reference");runpy.run_module("work.performance.benchmark",run_name="__main__")']
            else:
                command+=['-m','work.performance.benchmark']
            command+=['--case',case,'--steps','180','--output',str(root/version/f'{case}_{repeat}.json')]
            result=subprocess.run(command,check=True,capture_output=True,text=True)
            print(version,case,repeat,result.stdout.strip(),flush=True)
summary={}
for case in ('simulate','record','checkpoint'):
    sets={version:[json.loads(p.read_text()) for p in sorted((root/version).glob(f'{case}*.json'))] for version in ('before','after')}
    assert all(len(values)==3 for values in sets.values())
    assert len({v['state_sha256'] for values in sets.values() for v in values})==1
    summary[case]={version:{key:statistics.median(v[key] for v in values) for key in ('simulation_seconds','peak_rss_mib','output_bytes')+(('checkpoint_seconds',) if case=='checkpoint' else ())} for version,values in sets.items()}
summary['method']='Median of 3 separate-process runs per case/version; seed 42, 180 steps, 48 prey, 12 predators, Python 3.9.6. RSS measured before computing semantic digest. Before/after complete-state digests match.'
(root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
