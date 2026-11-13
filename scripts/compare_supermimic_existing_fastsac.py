"""Train SuperMimic and evaluate against an existing unchanged-task FastSAC run.

No FastSAC training command exists in this runner. Its existing checkpoint and
training cost are recorded. Compare the nearest lower SuperMimic sample count
without calling it exact equality; report the longer SuperMimic budget separately.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from supermimic_validation import validate_report
from train_supermimic_isaacgym import make_config


FIELDS=("simulator","robot","terrain","observation","action","command","reward",
    "termination","randomization","curriculum","scene","sensor","plugin")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation",type=Path,required=True)
    parser.add_argument("--fastsac-checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=2)
    parser.add_argument("--samples",type=int,default=51200000)
    parser.add_argument("--evaluation-seeds",nargs="+",type=int,default=[7919,104729,13007])
    parser.add_argument("--preflight-only",action="store_true")
    parser.add_argument("--evaluation-checkpoint",type=Path)
    parser.add_argument("--evaluation-phase",choices=("matched","longer"),default="matched")
    args=parser.parse_args()
    # Direct invocation of a venv Python does not activate its bin directory.
    # gymtorch discovers Ninja via PATH, including in all evaluation children.
    runtime_bin=str(Path(sys.executable).parent)
    os.environ["PATH"]=runtime_bin+os.pathsep+os.environ.get("PATH", "")
    ninja=shutil.which("ninja")
    if not ninja:
        raise RuntimeError("Ninja is missing from the training runtime: "+runtime_bin)
    subprocess.run([ninja,"--version"],check=True)
    repo=Path(__file__).resolve().parents[1]
    validation=(repo/args.validation).resolve()
    checkpoint=(repo/args.fastsac_checkpoint).resolve()
    output=(repo/args.output).resolve()
    gates=validate_report(validation)
    preparation=json.loads(Path(gates["preparation"]).read_text())
    algorithm=preparation["algorithm_config"]
    cfg=make_config(output,4096,args.seed,1)
    from holosoma.utils.safe_torch_import import torch
    import yaml
    old_config=yaml.safe_load((checkpoint.parent/"holosoma_config.yaml").read_text())
    task={}
    for key in FIELDS:
        value=getattr(cfg,key,None)
        task[key]=asdict(value) if hasattr(value,"__dataclass_fields__") else value
    task=json.loads(json.dumps(task))
    differences=[key for key in FIELDS if old_config.get(key)!=task[key]]
    if differences:
        raise ValueError("Existing FastSAC task differs: "+str(differences))
    saved=torch.load(checkpoint,map_location="cpu")
    baseline_samples=(int(saved["global_step"])+1)*old_config["training"]["num_envs"]
    del saved
    ratio=algorithm["update_to_data_ratio"]
    matched_iteration=math.floor(baseline_samples/4096*ratio)
    matched_samples=math.floor(matched_iteration/ratio)*4096
    final_iteration=round(args.samples/4096*ratio)
    assert math.floor(final_iteration/ratio)*4096==args.samples
    assert final_iteration>=matched_iteration>0
    # A logging/checkpoint interval change does not alter optimizer updates.
    assert matched_iteration%algorithm["save_interval"]==0, "Need an explicit saved checkpoint at the comparison budget"
    report=dict(status="preflight_passed",scope=__doc__,protocol_equal=True,
        source_sha256=gates["source_sha256"],validation=str(validation),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline=dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            training_samples=baseline_samples,training_seed=old_config["training"]["seed"],
            reused=True,new_training_transitions=0),
        supermimic=dict(training_seed=args.seed,training_samples=args.samples,cold_start=True,
            matched_iteration=matched_iteration,matched_samples=matched_samples,final_iteration=final_iteration),
        sample_difference=matched_samples-baseline_samples,
        sample_match_exact=matched_samples==baseline_samples,
        comparison_scope="One existing FastSAC seed and one fresh SuperMimic seed; fresh paired simulator evaluation seeds, not multiple training seeds. The longer SuperMimic endpoint is not a matched-sample comparison.",
        evaluation_seeds=args.evaluation_seeds,stages=[],started=time.time())
    if args.evaluation_checkpoint:
        previous=json.loads((output/"comparison.json").read_text())
        assert previous["baseline"]==report["baseline"]
        assert previous["source_sha256"]==report["source_sha256"]
        assert previous["evaluation_seeds"]==args.evaluation_seeds
        report=previous
    else:
        output.mkdir(parents=True,exist_ok=False)
    def write():
        (output/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    def stage(name,script,arguments,cwd=None):
        report.update(status="running",active_stage=name)
        write()
        with (output/(name+".log")).open("w") as stream:
            subprocess.run([sys.executable,"-u",str(repo/"scripts"/script)]+list(map(str,arguments)),
                cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,check=True)
        if name=="supermimic_training":
            # The checkpoint callback published intermediate evaluation results.
            published=json.loads((output/"comparison.json").read_text())
            report.clear()
            report.update(published)
        report["stages"].append(name)
        write()
    write()
    if args.preflight_only:
        print(json.dumps(report,indent=2))
        return
    try:
        release=Path.home()/".local/share/holosoma/figure1-isaacgym/holosoma-9c238cf80f531c0e65d818348c3c1a5cc2764f5b"
        if not args.evaluation_checkpoint:
            configuration=output/"algorithm_config.json"
            configuration.write_text(json.dumps(algorithm,indent=2)+"\n")
            start=time.monotonic()
            stage("supermimic_training","train_supermimic_with_comparison.py",["--base-dir",output/"supermimic/train",
                "--iterations",final_iteration,"--seed",args.seed,"--algorithm-config",configuration,
                "--comparison-output",output,"--matched-iteration",matched_iteration],release)
            pauses=json.loads((output/"training_pauses.json").read_text())
            report["supermimic"]["elapsed_training_seconds"]=time.monotonic()-start-pauses["seconds"]
            report["supermimic"]["evaluation_pause_seconds"]=pauses["seconds"]
            paths=list((output/"supermimic/train").rglob("model_%d.pt"%final_iteration))
            assert len(paths)==1
            checkpoints={"supermimic_longer":paths[0]}
        else:
            checkpoints={"supermimic_"+args.evaluation_phase:(repo/args.evaluation_checkpoint).resolve()}
        report.setdefault("evaluations",{})
        if "fastsac_existing" not in report["evaluations"]:
            checkpoints["fastsac_existing"]=checkpoint
        for label,path in checkpoints.items():
            expected_samples=baseline_samples if label=="fastsac_existing" else matched_samples if label=="supermimic_matched" else args.samples
            evaluations=[]
            for seed in args.evaluation_seeds:
                name=label+"_seed%d"%seed
                stage(name,"evaluate_g1_checkpoint.py",["--algorithm","fastsac" if label=="fastsac_existing" else "supermimic",
                    "--checkpoint",path,"--output",output/name,"--seed",seed,"--num-envs",512],release)
                result=json.loads((output/name/"evaluation.json").read_text())
                assert result["protocol_equal"] and result["censored_episodes"]==0
                assert result["training_transitions"]==expected_samples
                evaluations.append(result)
            report["evaluations"][label]=evaluations
            write()
        import statistics
        report["metrics"]={}
        for label in ("supermimic_matched","supermimic_longer"):
            if label not in report["evaluations"]:
                continue
            values=[]
            for metric in ("episode_return","episode_seconds","episode_planar_velocity_rmse_mps","episode_yaw_velocity_rmse_radps"):
                baseline, candidate=report["evaluations"]["fastsac_existing"],report["evaluations"][label]
                for left,right in zip(baseline,candidate):
                    assert all(left[key]==right[key] for key in ("seed","task_sha256","initial_penalty_scale","num_envs"))
                differences=[right[metric]["mean"]-left[metric]["mean"] for left,right in zip(baseline,candidate)]
                values.append(dict(metric=metric,fastsac_mean=statistics.mean(row[metric]["mean"] for row in baseline),
                    supermimic_mean=statistics.mean(row[metric]["mean"] for row in candidate),
                    difference_mean=statistics.mean(differences),
                    simulator_seed_standard_error=statistics.stdev(differences)/len(differences)**.5))
            report["metrics"][label]=values
        validate_report(validation)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==report["baseline"]["sha256"]
        report.update(status="matched_comparison_completed_training_continues" if args.evaluation_checkpoint else "completed",
            elapsed_seconds=time.time()-report["started"])
        write()
        subprocess.run([sys.executable,str(repo/"scripts/analyze_supermimic_reuse.py"),str(output)],check=True)
    except BaseException as error:
        report.update(status="execution_failed",error=repr(error))
        write()
        raise


if __name__=="__main__":
    main()
