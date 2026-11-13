"""Plot the actual Episode/rew_tracking_lin_vel TensorBoard metric."""
import argparse
import csv
import json
from pathlib import Path

from supermimic_metrics import EventAccumulator, plt
import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison",type=Path,required=True,help="Directory containing comparison.json")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.comparison.resolve()
    report=json.loads((root/'comparison.json').read_text())
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    directories={'fastsac':Path(report['baseline']['checkpoint']).parent,
                 'supermimic':Path(report['evaluations']['supermimic_longer'][0]['checkpoint']).parent}
    rows=[]
    checks={}
    for name,directory in directories.items():
        cfg=yaml.safe_load((directory/'holosoma_config.yaml').read_text())
        weight=cfg['reward']['terms']['tracking_lin_vel']['weight']
        dt=cfg['simulator']['config']['sim']['control_decimation']/cfg['simulator']['config']['sim']['fps']
        events=list(directory.glob('events.out.tfevents.*'))
        assert len(events)==1
        acc=EventAccumulator(str(events[0]),size_guidance={'scalars':0});acc.Reload()
        series={key:{p.step:p.value for p in acc.Scalars(tag)} for key,tag in {
            'reward':'Episode/rew_tracking_lin_vel','raw':'RawEpisode/raw_rew_tracking_lin_vel',
            'samples':'Train/num_samples'}.items()}
        errors=[]
        for iteration,value in sorted(series['reward'].items()):
            samples=int(series['samples'][iteration])
            if name=='fastsac':
                assert samples==iteration*cfg['training']['num_envs']
                samples+=cfg['training']['num_envs']
                if samples>report['baseline']['training_samples']:
                    continue
            raw=series['raw'][iteration]
            errors.append(abs(value-raw*weight*dt))
            rows.append(dict(algorithm=name,training_seed=cfg['training']['seed'],iteration=iteration,
                             real_transitions=samples,linear_velocity_tracking_reward=value,
                             raw_linear_velocity_tracking_score=raw))
        assert max(errors)<1e-5,(name,max(errors))
        checks[name]=dict(weight=weight,dt=dt,max_raw_scaling_difference=max(errors),
                          max_episode_length_s=cfg['simulator']['config']['sim']['max_episode_length_s'])
    with (output/'linear_velocity_tracking_reward.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for filename,limit in [('linear_velocity_tracking_reward',report['supermimic']['matched_samples']),
                           ('linear_velocity_tracking_reward_full',report['supermimic']['training_samples'])]:
        figure,axis=plt.subplots(figsize=(10,5.7),dpi=180)
        for name,color,label in [('fastsac','#1369b3','FastSAC (existing), seed 1'),
                                 ('supermimic','#a44c12','SuperMimic, seed 2')]:
            # Permit the single vector-step accounting difference at the common endpoint.
            selected=[r for r in rows if r['algorithm']==name and r['real_transitions']<=limit+4096]
            axis.plot([r['real_transitions']/1e6 for r in selected],
                      [r['linear_velocity_tracking_reward'] for r in selected],label=label,color=color,linewidth=2)
        axis.set(xlabel='Real training transitions (millions)',ylabel='Linear velocity tracking reward',
                 xlim=(0,(limit+4096)/1e6),ylim=(0,1.3 if limit>21000000 else 1.05))
        axis.grid(alpha=.2);axis.legend(loc='upper left')
        axis.set_title('G1 velocity tracking: training reward\nTensorBoard: Episode/rew_tracking_lin_vel',fontsize=13)
        figure.text(.5,.015,'Task weight and dt included; episode sum / 20 s. Actual logged points, no smoothing.',ha='center',fontsize=9)
        figure.tight_layout(rect=(0,.04,1,1))
        figure.savefig(output/(filename+'.png'));figure.savefig(output/(filename+'.svg'));plt.close(figure)
    summary=dict(tag='Episode/rew_tracking_lin_vel',meaning='Mean logged episode sum of weighted dt-scaled reward divided by the configured 20 s episode horizon; not per-surviving-step mean',
                 checks=checks,points=len(rows),sample_difference_at_comparison=-4096,
                 scope='Single training seed per algorithm; training metric with exploration and curriculum')
    (output/'linear_velocity_tracking_reward.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
