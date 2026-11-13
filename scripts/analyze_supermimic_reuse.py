"""Plot real-sample learning curves while preserving the reused baseline's provenance."""
import argparse
import csv
import json
import math
from pathlib import Path

from supermimic_metrics import EventAccumulator, OTHER_TAGS, SCORE_TAG, plt
import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    root=args.directory.resolve()
    report=json.loads((root/"comparison.json").read_text())
    output=root/"analysis"
    output.mkdir(exist_ok=True)
    candidates=report["evaluations"]
    label="supermimic_longer" if "supermimic_longer" in candidates else "supermimic_matched"
    paths={"fastsac":Path(report["baseline"]["checkpoint"]),
        "supermimic":Path(candidates[label][0]["checkpoint"])}
    limits={"fastsac":report["baseline"]["training_samples"],
        "supermimic":candidates[label][0]["training_transitions"]}
    configs={name:yaml.safe_load((path.parent/"holosoma_config.yaml").read_text()) for name,path in paths.items()}
    for field in ("simulator","robot","terrain","observation","action","command","reward",
                  "termination","randomization","curriculum","scene","sensor","plugin"):
        assert configs["fastsac"].get(field)==configs["supermimic"].get(field), field
    curves=[]
    for algorithm,path in paths.items():
        events=sorted(path.parent.glob("events.out.tfevents.*"))
        assert len(events)==1, "Explicitly handle multiple event files rather than silently select one"
        accumulator=EventAccumulator(str(events[0]),size_guidance={"scalars":0})
        accumulator.Reload()
        available=set(accumulator.Tags()["scalars"])
        tags={"linear_velocity_tracking":SCORE_TAG,**OTHER_TAGS}
        series={key:{point.step:point.value for point in accumulator.Scalars(tag)} for key,tag in tags.items() if tag in available}
        assert "simulator_transitions" in series and "linear_velocity_tracking" in series
        for iteration,value in series["linear_velocity_tracking"].items():
            if iteration not in series["simulator_transitions"] or not math.isfinite(value):
                continue
            samples=int(series["simulator_transitions"][iteration])
            if algorithm=="fastsac":
                assert samples==iteration*configs[algorithm]["training"]["num_envs"]
                samples+=configs[algorithm]["training"]["num_envs"]
            if samples>limits[algorithm]:
                continue
            curves.append(dict(algorithm=algorithm,training_seed=configs[algorithm]["training"]["seed"],
                iteration=iteration,real_transitions=samples,
                **{key:values.get(iteration) for key,values in series.items() if key!="simulator_transitions"}))
    assert all(any(row["algorithm"]==name for row in curves) for name in paths)
    with (output/"training_curves.csv").open("w",newline="") as stream:
        fields=list(dict.fromkeys(key for row in curves for key in row))
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(curves)
    metrics=[dict(comparison=name,**row) for name,rows in report.get("metrics",{}).items() for row in rows]
    if metrics:
        with (output/"independent_evaluations.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
    common=min(limits.values())
    for filename,limit in (("common_budget_curves.png",common),("all_logged_curves.png",max(limits.values()))):
        figure,axes=plt.subplots(2,2,figsize=(11,7.5),dpi=170)
        for name,color in (("fastsac","#1369b3"),("supermimic","#a44c12")):
            rows=sorted((row for row in curves if row["algorithm"]==name and row["real_transitions"]<=limit),key=lambda r:r["real_transitions"])
            for axis,key,title in ((axes[0,0],"linear_velocity_tracking","Linear tracking episode score"),
                (axes[0,1],"yaw_tracking_episode_score","Yaw tracking episode score"),
                (axes[1,0],"mean_episode_reward","Training episode return"),
                (axes[1,1],"mean_episode_steps","Training episode length (control steps)")):
                usable=[row for row in rows if row.get(key) is not None]
                axis.plot([row["real_transitions"]/1e6 for row in usable],[row[key] for row in usable],
                    label=("FastSAC (existing)" if name=="fastsac" else "SuperMimic")+"; seed %s"%configs[name]["training"]["seed"],color=color)
                axis.set(xlabel="Real training transitions (millions)",ylabel=title,xlim=(0,limit/1e6))
                axis.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        figure.suptitle("G1 velocity tracking: unchanged task, existing FastSAC baseline\nTraining curves include exploration and curriculum; independent evaluations are separate")
        figure.tight_layout()
        figure.savefig(output/filename)
        plt.close(figure)
    summary=dict(protocol_equal=True,checkpoint_sample_counts=limits,common_curve_limit=common,
        training_seeds={name:config["training"]["seed"] for name,config in configs.items()},
        sample_match_exact=report["sample_match_exact"],sample_difference=report["sample_difference"],
        independent_evaluations=report.get("metrics",{}),
        scope="Actual logged samples only, without extrapolation or smoothing. FastSAC step-zero collection is counted. One training seed per method; simulator evaluation-seed errors are not training-seed uncertainty. Longer SuperMimic results are not sample-matched.")
    (output/"results.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(dict(curve_points=len(curves),sample_limits=limits,analysis=str(output))),flush=True)


if __name__=="__main__":
    main()
