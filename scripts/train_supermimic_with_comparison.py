"""Keep the live training state while evaluating its sample-matched checkpoint."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

from train_supermimic_isaacgym import make_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir",type=Path,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--iterations",type=int,required=True)
    parser.add_argument("--algorithm-config",type=Path,required=True)
    parser.add_argument("--comparison-output",type=Path,required=True)
    parser.add_argument("--matched-iteration",type=int,required=True)
    args=parser.parse_args()
    cfg=make_config(args.base_dir,4096,args.seed,args.iterations)
    algorithm=json.loads(args.algorithm_config.read_text())
    algorithm["num_learning_iterations"]=args.iterations
    cfg=replace(cfg,algo=replace(cfg.algo,config=replace(cfg.algo.config,**algorithm)))
    from holosoma.agents.supermimic.supermimic import SuperMimicAgent
    from holosoma.utils.safe_torch_import import torch
    from holosoma.train_agent import train
    original_save=SuperMimicAgent.save
    completed=False
    def save(agent,path=None,name="last.pt"):
        nonlocal completed
        original_save(agent,path,name)
        if completed or agent.iteration!=args.matched_iteration:
            return
        checkpoint=Path(path) if path else Path(agent.log_dir)/name
        agent.writer.flush()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        cpu_rng,cuda_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
        steps,replay=agent.collected_steps,len(agent.replay)
        report=json.loads((args.comparison_output/"comparison.json").read_text())
        command=[sys.executable,"-u",str(Path(__file__).with_name("compare_supermimic_existing_fastsac.py")),
            "--validation",report["validation"],"--fastsac-checkpoint",report["baseline"]["checkpoint"],
            "--output",str(args.comparison_output),"--seed",str(args.seed),
            "--samples",str(report["supermimic"]["training_samples"]),
            "--evaluation-checkpoint",str(checkpoint),"--evaluation-phase","matched",
            "--evaluation-seeds",*map(str,report["evaluation_seeds"])]
        start=time.monotonic()
        with (args.comparison_output/"matched_callback.log").open("w") as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        elapsed=time.monotonic()-start
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
        assert agent.collected_steps==steps and len(agent.replay)==replay
        (args.comparison_output/"training_pauses.json").write_text(json.dumps(dict(
            iteration=agent.iteration,seconds=elapsed,retained_replay_steps=replay,
            collected_steps=steps,rng_unchanged=True,
            scope="Training paused in its existing process; simulator, FIFO and optimizer state remain resident. Evaluation runs in separate processes."),indent=2)+"\n")
        completed=True
        print("Matched-sample comparison saved; continuing the same live training run",flush=True)
    SuperMimicAgent.save=save
    try:
        train(cfg)
        assert completed, "Expected matched-sample evaluation was never reached"
    finally:
        SuperMimicAgent.save=original_save


if __name__=="__main__":
    main()
