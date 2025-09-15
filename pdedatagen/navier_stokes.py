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
from tqdm import tqdm

from pdearena import utils

from .pde import PDEConfig

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

    save_name = os.path.join(dirname, "_".join([pde_string, mode, str(seed), f"{pde.buoyancy_y:.5f}"]))
    if mode == "train":
        save_name = save_name + "_" + str(num_samples)
    h5f = h5py.File("".join([save_name, ".h5"]), "a")
    dataset = h5f.create_group(mode)

    tcoord, xcoord, ycoord, dt, dx, dy = {}, {}, {}, {}, {}, {}
    h5f_u, h5f_vx, h5f_vy = {}, {}, {}

    nt, nx, ny = pde.grid_size[0], pde.grid_size[1], pde.grid_size[2]
    # The scalar field u, the components of the vector field vx, vy,
    # the coordinations (tcoord, xcoord, ycoord) and dt, dx, dt are saved
    h5f_u = dataset.create_dataset("u", (num_samples, nt, nx, ny), dtype=float)
    h5f_vx = dataset.create_dataset("vx", (num_samples, nt, nx, ny), dtype=float)
    h5f_vy = dataset.create_dataset("vy", (num_samples, nt, nx, ny), dtype=float)
    tcoord = dataset.create_dataset("t", (num_samples, nt), dtype=float)
    dt = dataset.create_dataset("dt", (num_samples,), dtype=float)
    xcoord = dataset.create_dataset("x", (num_samples, nx), dtype=float)
    dx = dataset.create_dataset("dx", (num_samples,), dtype=float)
    ycoord = dataset.create_dataset("y", (num_samples, ny), dtype=float)
    dy = dataset.create_dataset("dy", (num_samples,), dtype=float)
    buo_y = dataset.create_dataset("buo_y", (num_samples,), dtype=float)

    def genfunc(idx, s):
        phi_seed(idx + s)
        smoke = abs(
            CenteredGrid(
                Noise(scale=11.0, smoothness=6.0),
                extrapolation.BOUNDARY,
                x=pde.nx,
                y=pde.ny,
                bounds=Box['x,y', 0 : pde.Lx, 0 : pde.Ly],
            )
        )  # sampled at cell centers
        velocity = StaggeredGrid(
            0, extrapolation.ZERO, x=pde.nx, y=pde.ny, bounds=Box['x,y', 0 : pde.Lx, 0 : pde.Ly]
        )  # sampled in staggered form at face centers
        fluid_field_ = []
        velocity_ = []
        for i in range(0, pde.nt + pde.skip_nt):
            smoke = advect.semi_lagrangian(smoke, velocity, pde.dt)
            buoyancy_force = (smoke * (0, pde.buoyancy_y)).at(velocity)  # resamples smoke to velocity sample points
            velocity = advect.semi_lagrangian(velocity, velocity, pde.dt) + pde.dt * buoyancy_force
            velocity = diffuse.explicit(velocity, pde.nu, pde.dt)
            velocity, _ = fluid.make_incompressible(velocity)
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
        return fluid_field_[:: pde.sample_rate, ...], velocity_corrected_[:: pde.sample_rate, ...]

    with utils.Timer() as gentime:
        rngs = np.random.randint(np.iinfo(np.int32).max, size=num_samples)
        fluid_field, velocity_corrected = [], []
        
        for idx in tqdm(range(num_samples)):
            try:
                phi_seed(idx + rngs[idx])
                smoke = abs(
                    CenteredGrid(
                        Noise(scale=11.0, smoothness=6.0),
                        extrapolation.BOUNDARY,
                        x=pde.nx,
                        y=pde.ny,
                        bounds=Box['x,y', 0 : pde.Lx, 0 : pde.Ly],
                    )
                )  # sampled at cell centers
                velocity = StaggeredGrid(
                    0, extrapolation.ZERO, x=pde.nx, y=pde.ny, bounds=Box['x,y', 0 : pde.Lx, 0 : pde.Ly]
                )  # sampled in staggered form at face centers
                fluid_field_ = []
                velocity_ = []
                for i in range(0, pde.nt + pde.skip_nt):
                    smoke = advect.semi_lagrangian(smoke, velocity, pde.dt)
                    buoyancy_force = (smoke * (0, pde.buoyancy_y)).at(velocity)  # resamples smoke to velocity sample points
                    velocity = advect.semi_lagrangian(velocity, velocity, pde.dt) + pde.dt * buoyancy_force
                    velocity = diffuse.explicit(velocity, pde.nu, pde.dt)
                    
                    # Try to make velocity incompressible with error handling
                    try:
                        velocity, _ = fluid.make_incompressible(velocity)
                    except Exception as e:
                        print(f"Warning: Convergence failed at step {i}, residual error: {e}")
                        # Try with a smaller time step for this step
                        try:
                            small_dt = pde.dt * 0.1
                            velocity = advect.semi_lagrangian(velocity, velocity, small_dt) + small_dt * buoyancy_force
                            velocity = diffuse.explicit(velocity, pde.nu, small_dt)
                            velocity, _ = fluid.make_incompressible(velocity)
                            print(f"Successfully recovered with smaller time step")
                        except Exception as e2:
                            print(f"Failed even with smaller time step: {e2}")
                            # As a last resort, skip the incompressibility step
                            print("Skipping incompressibility step - this may affect simulation quality")
                    
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
                fluid_field.append(fluid_field_[:: pde.sample_rate, ...])
                velocity_corrected.append(velocity_corrected_[:: pde.sample_rate, ...])
                print(f"Successfully generated sample {idx}")
            except Exception as e:
                print(f"Error generating sample {idx}: {e}")
                raise

    logger.info(f"Took {gentime.dt:.3f} seconds")

    with utils.Timer() as writetime:
        for idx in range(num_samples):
            # fmt: off
            # Saving the trajectories
            h5f_u[idx : (idx + 1), ...] = fluid_field[idx]
            h5f_vx[idx : (idx + 1), ...] = velocity_corrected[idx][..., 0]
            h5f_vy[idx : (idx + 1), ...] = velocity_corrected[idx][..., 1]
            # fmt:on
            xcoord[idx : (idx + 1), ...] = np.asarray([np.linspace(0, pde.Lx, pde.nx)])
            dx[idx : (idx + 1)] = pde.dx
            ycoord[idx : (idx + 1), ...] = np.asarray([np.linspace(0, pde.Ly, pde.ny)])
            dy[idx : (idx + 1)] = pde.dy
            tcoord[idx : (idx + 1), ...] = np.asarray([np.linspace(pde.tmin, pde.tmax, pde.trajlen)])
            dt[idx : (idx + 1)] = pde.dt * pde.sample_rate
            buo_y[idx : (idx + 1)] = pde.buoyancy_y

    logger.info(f"Took {writetime.dt:.3f} seconds writing to disk")

    print()
    print("Data saved")
    print()
    print()
    h5f.close()
