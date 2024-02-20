<div id="top" align="center">

<p align="center">
  <img src="assets/navsim_transparent.png" width="500">
</p>
    
**NAVSIM:** *Data-Driven **N**on-Reactive **A**utonomous **V**ehicle **Sim**ulation*

</div>


## Highlights <a name="highlight"></a>

🔥 NAVSIM gathers simulation-based metrics (such as progress and time to collision) for end-to-end driving by unrolling simplified bird's eye view abstractions of scenes for a short simulation horizon. It operates under the condition that the policy has no influence on the environment, which enables **efficient, open-loop metric computation** while being **better aligned with closed-loop** evaluations than traditional displacement errors. 

> NAVSIM attempts to address some of the challenges faced by the community:
> 
> 1. **Providing a principled evaluation** (by incorporating ideas + data from nuPlan)
>   - Key Idea: **PDM Score**, a multi-dimensional metric implemented in open-loop with strong correlation to closed-loop metrics
>   - Critical scenario sampling, focusing on situations with intention changes where the ego history cannot be extrapolated into a plan
>   - Official leaderboard on HuggingFace that remains open and prevents ambiguity in metric definitions between projects
> 
> 2. **Maintaining ease of use** (by emulating nuScenes)
>   - Simple data format and reasonably-sized download (<nuPlan’s 5+ TB)
>   - Large-scale publicly available test split for internal benchmarking
>   - Continually-maintained devkit

🏁 **NAVSIM** will serve as a main track in the **`CVPR 2024 Autonomous Grand Challenge`**. For further details, please [stay tuned](https://opendrivelab.com/challenge2024/)!


## Table of Contents
1. [Highlights](#highlight)
2. [Getting started](#gettingstarted)
3. [Changelog](#changelog)
4. [License and citation](#licenseandcitation)
5. [Other resources](#otherresources)


## Getting started <a name="gettingstarted"></a>

- [Download and installation](docs/install.md)
- [Understanding and creating agents](docs/agents.md) 
- [Understanding the data format and classes](docs/cache.md)
- [Understanding the PDM Score](docs/metrics.md)
  
<p align="right">(<a href="#top">back to top</a>)</p>


## Changelog <a name="changelog"></a>

- **`[2024/02/20]`** NAVSIM v0.1 release (initial demo)
  - OpenScene-mini sensor blobs and annotation logs
  - Naive `ConstantVelocity` agent


<p align="right">(<a href="#top">back to top</a>)</p>


## License and citation <a name="licenseandcitation"></a>
All assets and code in this repository are under the [Apache 2.0 license](./LICENSE) unless specified otherwise. The datasets (including nuPlan and OpenScene) inherit their own distribution licenses. Please consider citing our paper and project if they help your research.

```BibTeX
@misc{Contributors2024navsim,
    title={NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation},
    author={NAVSIM Contributors},
    howpublished={\url{https://github.com/autonomousvision/navsim}},
    year={2024}
} 
```

```BibTeX
@inproceedings{Dauner2023CORL,
    title = {Parting with Misconceptions about Learning-based Vehicle Motion Planning},
    author = {Daniel Dauner and Marcel Hallgarten and Andreas Geiger and Kashyap Chitta},
    booktitle = {Conference on Robot Learning (CoRL)},
    year = {2023}
} 
```

<p align="right">(<a href="#top">back to top</a>)</p>


## Other resources <a name="otherresources"></a>

<a href="https://twitter.com/AutoVisionGroup" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/Awesome Vision Group?style=social&color=brightgreen&logo=twitter" />
  </a>
  
- [tuPlan garage](https://github.com/autonomousvision/tuplan_garage) | [CARLA garage](https://github.com/autonomousvision/carla_garage) | [Survey on E2EAD](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving)
- [PlanT](https://github.com/autonomousvision/plant) | [KING](https://github.com/autonomousvision/king) | [TransFuser](https://github.com/autonomousvision/transfuser) | [NEAT](https://github.com/autonomousvision/neat)

<p align="right">(<a href="#top">back to top</a>)</p>
