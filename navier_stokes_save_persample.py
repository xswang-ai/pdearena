# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import os

import h5py
import numpy as np
import torch
from joblib import Parallel, delayed
from phi.flow import (  # SoftGeometryMask,; Sphere,; batch,; tensor,
    Box,
    CenteredGrid,
    Noise,
    StaggeredGrid,
    advect,
    diffuse,
    extrapolation,
    fluid,
)
from phi.math import reshaped_native
from phi.math import seed as phi_seed
from phi.math import Solve
from tqdm import tqdm

import utils

from pde import PDEConfig

logger = logging.getLogger(__name__)


def generate_trajectories_smoke(
    pde: PDEConfig,
    mode: str,
    num_samples: int,
    batch_size: int,
    device: torch.device = torch.device("cpu"),
    dirname: str = "data",
    n_parallel: int = 1,
    seed: int = 42,
    tol: float = 1e-5,
) -> None:
    """
    Generate data trajectories for smoke inflow in bounded domain
    Args:
        pde (PDEConfig): pde at hand [NS2D]
        mode (str): [train, valid, test]
        num_samples (int): how many trajectories do we create
        batch_size (int): batch size
        device: device (cpu/gpu)
    Returns:
        None
    """

    pde_string = str(pde)
    logger.info(f"Equation: {pde_string}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"Device: {device}")
    logger.info(f"Parallel: {n_parallel}")
    save_name = os.path.join(dirname, "_".join([pde_string, mode, str(seed), f"{pde.buoyancy_y:.5f}"]), f"tol{tol:.5f}"+"_" + "nt" + str(pde.nt))
    if mode == "train":
        save_name = save_name + "_" + str(num_samples)
    
    # Check which samples already exist
    existing_samples = []
    for idx in range(num_samples):
        sample_filename = f"{save_name}_sample_{idx:06d}.h5"
        if os.path.exists(sample_filename):
            existing_samples.append(idx)
    
    samples_to_generate = [idx for idx in range(num_samples) if idx not in existing_samples]
    
    if not samples_to_generate:
        logger.info(f"All {num_samples} samples already exist. Nothing to do.")
        return
    
    logger.info(f"Found {len(existing_samples)} existing samples. Need to generate {len(samples_to_generate)} more samples: {samples_to_generate}")

    nt, nx, ny = pde.grid_size[0], pde.grid_size[1], pde.grid_size[2]

    def genfunc(idx, s):
        phi_seed(idx + s)
        smoke = abs(
            CenteredGrid(
                Noise(scale=11.0, smoothness=6.0),
                # extrapolation.PERIODIC,  # PERIODIC boundary conditions
                extrapolation.BOUNDARY,
                x=pde.nx,
                y=pde.ny,
                bounds=Box['x,y', 0 : pde.Lx, 0 : pde.Ly],
            )
        )  # sampled at cell centers
        velocity = StaggeredGrid(
            0, 
            # extrapolation.PERIODIC,
            extrapolation.ZERO,
             x=pde.nx, y=pde.ny, bounds=Box['x,y', 0 : pde.Lx, 0 : pde.Ly]
        )  # sampled in staggered form at face centers
        fluid_field_ = []
        velocity_ = []
        print(f"Solving with relative tolerance {tol}")
        for i in tqdm(range(0, pde.nt + pde.skip_nt)):
            smoke = advect.semi_lagrangian(smoke, velocity, pde.dt)
            buoyancy_force = (smoke * (0, pde.buoyancy_y)).at(velocity)  # resamples smoke to velocity sample points
            velocity = advect.semi_lagrangian(velocity, velocity, pde.dt) + pde.dt * buoyancy_force
            velocity = diffuse.explicit(velocity, pde.nu, pde.dt)
            velocity, _ = fluid.make_incompressible(velocity, solve=Solve(rel_tol=tol))
            fluid_field_.append(reshaped_native(smoke.values, groups=("x", "y", "vector"), to_numpy=True))
            velocity_.append(
                reshaped_native(
                    velocity.staggered_tensor(),
                    groups=("x", "y", "vector"),
                    to_numpy=True,
                )
            )

        fluid_field_ = np.asarray(fluid_field_[pde.skip_nt :]).squeeze()
        # velocity has the shape [nt, nx+1, ny+2, 2]
        velocity_corrected_ = np.asarray(velocity_[pde.skip_nt :]).squeeze()[:, :-1, :-1, :]
        
        fluid_field = fluid_field_[:: pde.sample_rate, ...]
        velocity_corrected = velocity_corrected_[:: pde.sample_rate, ...]
        
        # Save this sample to its own file
        sample_filename = f"{save_name}_sample_{idx:06d}.h5"
        with h5py.File(sample_filename, "w") as sample_h5f:
            sample_h5f.create_dataset("u", data=fluid_field)
            sample_h5f.create_dataset("vx", data=velocity_corrected[..., 0])
            sample_h5f.create_dataset("vy", data=velocity_corrected[..., 1])
            sample_h5f.create_dataset("t", data=np.linspace(pde.tmin, pde.tmax, pde.trajlen))
            sample_h5f.create_dataset("dt", data=pde.dt * pde.sample_rate)
            sample_h5f.create_dataset("x", data=np.linspace(0, pde.Lx, pde.nx))
            sample_h5f.create_dataset("dx", data=pde.dx)
            sample_h5f.create_dataset("y", data=np.linspace(0, pde.Ly, pde.ny))
            sample_h5f.create_dataset("dy", data=pde.dy)
            sample_h5f.create_dataset("buo_y", data=pde.buoyancy_y)
        
        return fluid_field, velocity_corrected

    with utils.Timer() as gentime:
        # Generate random seeds only for the samples we need to create
        rngs = np.random.randint(np.iinfo(np.int32).max, size=len(samples_to_generate))
        # print("rngs", rngs, "rngs type", type(rngs),  rngs  [0].item(), "rngs[0] type", type(rngs[0].item()))
        if torch.cuda.is_available() or n_parallel > 1:
            Parallel(n_jobs=n_parallel)(delayed(genfunc)(idx, rngs[i].item()) for i, idx in enumerate(tqdm(samples_to_generate)))
        else:
            for i, idx in enumerate(tqdm(samples_to_generate)):
                genfunc(idx, rngs[i].item())

    logger.info(f"Took {gentime.dt:.3f} seconds")

    print()
    print(f"Generated {len(samples_to_generate)} new samples")
    print(f"Total samples: {num_samples} ({len(existing_samples)} existing + {len(samples_to_generate)} new)")
    print("Each sample saved to individual HDF5 files")
    print()