# Less Is More in Opinion Dynamics

Code accompanying the manuscript:

**Dynamic networks can trap collective decisions**

This repository contains Python code to simulate and analyse a minimal opinion-dynamics model in which agents interact on static or rewiring networks to reach consensus between two alternatives of different quality. The model investigates how finite deliberation times and time-varying interaction networks affect collective adaptation to a superior option.

## Overview

The system models a population initially committed to an inferior option. The superior option is introduced through independent discovery. Agents can then spread opinions through pairwise social interactions on a network. When agents adopt a new opinion, they may enter a deliberation state before becoming active spreaders of that opinion.

The code explores how collective adaptation depends on:

- network rewiring probability;
- deliberation time;
- population size;
- network connectivity;
- option-quality asymmetry;
- independent discovery probability.

The main result reproduced by this code is that network rewiring can have opposite effects depending on deliberation time: it accelerates convergence when opinion changes are immediate, but can prevent collective adaptation when agents require time before retransmitting newly acquired information.

---

## Repository structure

```text
LessIsMoreOpinionDynamics/
├── BestOfNOnRewiringNets/
│   ├── bestOf2OnGraphs.py       # main multi-agent simulation script
│   ├── meanfield_eq_plot.py     # mean-field model and plotting
│   └── plotResults.py           # analysis and plotting of simulation outputs
├── config/
│   ├── bestOf2.config           # example configuration file
│   └── bestOf2.template.config  # template configuration file
├── scripts/
│   ├── run_job.sh               # helper script for job execution
│   ├── runexps.sh               # batch experiment launcher
│   └── runexps_singlesrun.sh    # single-run experiment launcher
└── README.md
```

---

## Requirements

The code is written in Python and uses standard scientific-computing libraries.

Main dependencies:

```text
numpy
networkx
matplotlib
scipy
pandas
```

To install the dependencies in a fresh environment:

```bash
python -m venv venv
source venv/bin/activate
pip install numpy scipy pandas matplotlib networkx
```

---

## Running a simulation

Edit the configuration file in `config/bestOf2.config`, then run:

```bash
python BestOfNOnRewiringNets/bestOf2OnGraphs.py config/my_experiment.config
```

The configuration file controls the main model parameters, including:

```ini
[experiment]
randomSeed=12345
outputTxtFile=../data/out.txt
cluster=false
numberOfExperiments=1
maxTime=10000
optionQualities=[1,0.8]
discoveryProbability=0.01
deliberationTime=50

[network]
numberOfNodes=100
netType=erdos-renyi
networkParameter=0.04
rewiringProbability=0.5
```

Here, `optionQualities=[1,0.8]` means that option A has quality `1.0` and option B has quality `0.8`. The parameter `discoveryProbability` controls independent discovery of the superior option. The parameter `deliberationTime` sets the number of timesteps an agent remains in deliberation before committing to a newly encountered opinion. The parameter `rewiringProbability` controls how frequently the interaction network is rewired.

---

## Mean-field analysis

To reproduce the mean-field analysis and associated plots, run:

```bash
python BestOfNOnRewiringNets/meanfield_eq_plot.py
```

This script implements the deterministic mean-field equations used to analyse the population-level dynamics of agents committed to or deliberating on each option.

---

## Reproducing the paper results

The manuscript explores parameter sweeps over:

- group size `N`;
- Erdős–Rényi link probability;
- inferior-option quality;
- discovery probability;
- deliberation duration;
- rewiring probability.

The shell scripts in `scripts/` provide examples for launching multiple simulation runs. Depending on the size of the parameter sweep, reproducing the full dataset may require substantial computation time.

---

## Model summary

At each timestep:

1. One agent is selected uniformly at random.
2. With probability `p_d`, the agent independently discovers the superior option A.
3. Otherwise, the agent interacts socially with a non-deliberating neighbour.
4. The neighbour is selected with probability proportional to the quality of its opinion.
5. If the selected neighbour holds a different opinion, the focal agent enters deliberation.
6. Deliberating agents can still be influenced by others, but they do not transmit their own opinion until deliberation is complete.
7. Network rewiring occurs with probability `rho`, preserving node degrees.

A run is considered successful if the population reaches consensus on the superior option before the maximum simulation time.

---

## Citation

If you use this code, please cite:

```bibtex
@article{KhalufReina2026NetworkRewiring,
  title   = {Dynamic networks can trap collective decisions},
  author  = {Khaluf, Yara and Reina, Andreagiovanni},
  journal = {Manuscript under review},
  year    = {2026}
}
```

---

## Authors

Yara Khaluf  
Andreagiovanni Reina

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.