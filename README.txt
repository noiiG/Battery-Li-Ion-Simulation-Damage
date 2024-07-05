The developed chemo-mechanical model undergoes fatigue failure package is written in FEniCS as a popular free open-source computing platform for solving partial differ- ential equations in which simplifies the implementation of parallel FEM simulations. 

We note that by using MPI running of the FEniCS open platform, we can distribute the computational load for fatigue failure analysis in a multiphysics setting to the multiple cores by: 

	mpirun -np number[cores] python3 name[script].py.

FEniCS is an free open-source computing platform [109] for solving partial differential equations (PDEs). In fact, FEniCS helps developer and scientific researcher to rewrite a serial code into a parallel running code.The FEniCS docker container already includes MPICH and is fundamentally developed for parallel processing on laptops, workstations and also HPC clusters. The user can define the PDEs in Python that FEniCS translates via ffc into a low-level C code, enabling the compilation of highly efficient simulations. A wide variety of elements are supported and meshes from for example GMSH can be implemented and handled very easily. FEniCS makes it easy to change and adapt the code quickly, since for example solver algorithms can be changed with simple commands and parameters. The implementation of a particular algorithm is avoided. In order to summarize the code concisely, only the most important parts of the code are presented here.

Additonally, this open-source code (https://github.com/noiiG) is provided, constituting a convenient platform for future developments, e.g., multi-field coupled problems.

Please Cite the following papers, if you aim to use the following codes:


 @article{noii2024efficient,
  title={An Efficient FEniCS implementation for coupling lithium-ion battery charge/discharge processes with fatigue phase-field fracture},
  author={Noii, Nima and Milijasevic, Dejan and Khodadadian, Amirreza and Wick, Thomas},
  journal={Engineering Fracture Mechanics},
  pages={110251},
  year={2024},
  publisher={Elsevier}
}

@article{noii2024fatigue,
  title={Fatigue failure theory for lithium diffusion induced fracture in lithium-ion battery electrode particles},
  author={Noii, Nima and Milijasevic, Dejan and Waisman, Haim and Khodadadian, Amirreza},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={428},
  pages={117068},
  year={2024},
  publisher={Elsevier}
}

Dr.-Ing Nima Noii
