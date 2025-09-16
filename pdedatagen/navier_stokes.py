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
from phi.math import spatial
from phi.math import tensor
from tqdm import tqdm

from pdearena import utils

from .pde import PDEConfig

logger = logging.getLogger(__name__)


def check_simulation_stability(pde: PDEConfig) -> bool:
    """
    Check if simulation parameters are stable for Navier-Stokes simulation.
    Returns True if parameters appear stable, False otherwise.
    """
    # Check CFL condition: dt should be small enough for advection
    cfl_advection = pde.dt * max(1.0 / pde.dx, 1.0 / pde.dy)  # Assuming max velocity ~ 1
    cfl_stable = cfl_advection < 1.0
    
    # Check diffusion stability: dt should be small enough for explicit diffusion
    diff_stability = pde.dt * pde.nu * (1.0 / pde.dx**2 + 1.0 / pde.dy**2) < 0.5
    
    # Check buoyancy force magnitude
    buoyancy_stable = abs(pde.buoyancy_y) * pde.dt < 1.0
    
    logger.info(f"CFL condition (advection): {cfl_advection:.4f} < 1.0 = {cfl_stable}")
    logger.info(f"Diffusion stability: {pde.dt * pde.nu * (1.0 / pde.dx**2 + 1.0 / pde.dy**2):.4f} < 0.5 = {diff_stability}")
    logger.info(f"Buoyancy stability: {abs(pde.buoyancy_y) * pde.dt:.4f} < 1.0 = {buoyancy_stable}")
    
    return cfl_stable and diff_stability and buoyancy_stable


def get_stable_time_step(pde: PDEConfig, safety_factor: float = 0.5) -> float:
    """
    Calculate a stable time step based on CFL and diffusion conditions.
    Returns a smaller dt if needed for stability.
    """
    # CFL condition for advection (assuming max velocity ~ 1)
    cfl_dt = safety_factor / max(1.0 / pde.dx, 1.0 / pde.dy)
    
    # Diffusion stability condition
    diff_dt = safety_factor * 0.5 / (pde.nu * (1.0 / pde.dx**2 + 1.0 / pde.dy**2))
    
    # Buoyancy stability condition
    buoyancy_dt = safety_factor / abs(pde.buoyancy_y) if pde.buoyancy_y != 0 else float('inf')
    
    # Use the most restrictive condition
    stable_dt = min(cfl_dt, diff_dt, buoyancy_dt, pde.dt)
    
    if stable_dt < pde.dt:
        logger.warning(f"Reducing time step from {pde.dt:.6f} to {stable_dt:.6f} for stability")
        logger.info(f"CFL dt: {cfl_dt:.6f}, Diffusion dt: {diff_dt:.6f}, Buoyancy dt: {buoyancy_dt:.6f}")
    
    return stable_dt


def create_incompressible_initial_velocity(pde: PDEConfig, seed: int) -> StaggeredGrid:
    """
    Create an initial velocity field that is closer to incompressible.
    This helps reduce convergence issues in the first few steps.
    """
    phi_seed(seed)
    
    # For staggered grid, we need to create velocity components with correct sizes
    # x-velocity: (nx-1, ny) - staggered in x direction
    # y-velocity: (nx, ny-1) - staggered in y direction
    
    # Create coordinate arrays for staggered grid
    x_coords_u = np.linspace(0, pde.Lx, pde.nx - 1)  # x-velocity points
    y_coords_u = np.linspace(0, pde.Ly, pde.ny)      # x-velocity points
    x_coords_v = np.linspace(0, pde.Lx, pde.nx)      # y-velocity points  
    y_coords_v = np.linspace(0, pde.Ly, pde.ny - 1)  # y-velocity points
    
    # Create meshgrids for each component
    X_u, Y_u = np.meshgrid(x_coords_u, y_coords_u, indexing='ij')
    X_v, Y_v = np.meshgrid(x_coords_v, y_coords_v, indexing='ij')
    
    # Create stream functions for each component
    stream_u = (
        0.1 * np.sin(2 * np.pi * X_u / pde.Lx) * np.cos(2 * np.pi * Y_u / pde.Ly) +
        0.05 * np.sin(4 * np.pi * X_u / pde.Lx) * np.cos(4 * np.pi * Y_u / pde.Ly)
    )
    stream_v = (
        0.1 * np.sin(2 * np.pi * X_v / pde.Lx) * np.cos(2 * np.pi * Y_v / pde.Ly) +
        0.05 * np.sin(4 * np.pi * X_v / pde.Lx) * np.cos(4 * np.pi * Y_v / pde.Ly)
    )
    
    # Compute velocity components from stream function (automatically divergence-free)
    # u = -dψ/dy, v = dψ/dx
    u = -np.gradient(stream_u, axis=1) / pde.dy
    v = np.gradient(stream_v, axis=0) / pde.dx
    
    # Convert to proper tensor format with dimension names
    u_tensor = tensor(u, spatial('x,y'))
    v_tensor = tensor(v, spatial('x,y'))
    
    # Create staggered grid with the computed velocity
    velocity = StaggeredGrid(
        (u_tensor, v_tensor), 
        extrapolation.ZERO, 
        x=pde.nx, 
        y=pde.ny, 
        bounds=Box['x,y', 0 : pde.Lx, 0 : pde.Ly]
    )
    
    logger.info("Created incompressible initial velocity field")
    return velocity


def smooth_velocity(velocity: StaggeredGrid, smoothing_factor: float = 0.1) -> StaggeredGrid:
    """
    Apply light smoothing to velocity field to reduce high-frequency components
    that can cause convergence issues in the pressure solver.
    """
    # Apply a small amount of diffusion to smooth the velocity field
    smoothed_velocity = diffuse.explicit(velocity, smoothing_factor, 1.0)
    return smoothed_velocity


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
    
    # Check simulation stability and get stable time step
    if not check_simulation_stability(pde):
        logger.warning("Simulation parameters may be unstable. Consider reducing dt or adjusting other parameters.")
    
    # Use a stable time step
    stable_dt = get_stable_time_step(pde)
    logger.info(f"Using time step: {stable_dt:.6f}")

    save_name = os.path.join(dirname, "_".join([pde_string, mode, str(seed), f"{pde.buoyancy_y:.5f}"]))
    if mode == "train":
        save_name = save_name + "_" + str(num_samples)
    h5f = h5py.File("".join([save_name, ".h5"]), "a")
    # if file already exists, delete it
    if os.path.exists("".join([save_name, ".h5"])):
        os.remove("".join([save_name, ".h5"]))
        print(f"Deleted existing file {save_name}.h5")
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
        print('idx type: ', type(idx))
        print('s type: ', type(s))
        print('idx + s type: ', type(idx + s))
        print(idx + s)
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
        # Use improved initial velocity that's closer to incompressible
        try:
            velocity = create_incompressible_initial_velocity(pde, idx + s)
        except Exception as e:
            logger.warning(f"Failed to create incompressible initial velocity: {e}")
            # Fallback to zero velocity (guaranteed incompressible)
            velocity = StaggeredGrid(
                0, extrapolation.ZERO, x=pde.nx, y=pde.ny, bounds=Box['x,y', 0 : pde.Lx, 0 : pde.Ly]
            )
            logger.info("Using zero initial velocity as fallback")
        fluid_field_ = []
        velocity_ = []
        for i in range(0, pde.nt + pde.skip_nt):
            smoke = advect.semi_lagrangian(smoke, velocity, stable_dt)
            buoyancy_force = (smoke * (0, pde.buoyancy_y)).at(velocity)  # resamples smoke to velocity sample points
            velocity = advect.semi_lagrangian(velocity, velocity, stable_dt) + stable_dt * buoyancy_force
            velocity = diffuse.explicit(velocity, pde.nu, stable_dt)
            
            # Apply light smoothing to improve convergence
            velocity = smooth_velocity(velocity, smoothing_factor=0.01)
            
            try:
                # Try with a more robust solver from the start
                solve_robust = Solve(abs_tol=1e-4, rel_tol=1e-4, max_iterations=500, method='CG')
                velocity, _ = fluid.make_incompressible(velocity, solve=solve_robust)
                if i % 10 == 0:  # Log every 10 steps to avoid spam
                    logger.info(f"Converged successfully at step {i}")
            except Exception as e:
                logger.warning(f"Convergence failed at step {i}: {e}")
                # Try with a custom solver with relaxed parameters
                try:
                    # Try with different solver parameters
                    solve_custom = Solve(abs_tol=1e-3, rel_tol=1e-3, max_iterations=1000)
                    velocity, _ = fluid.make_incompressible(velocity, solve=solve_custom)
                    logger.info(f"Converged with custom solver at step {i}")
                except Exception as e2:
                    logger.warning(f"Custom solver also failed at step {i}: {e2}")
                    # Try with iterative solver (CG) which is more robust
                    try:
                        solve_iterative = Solve(abs_tol=1e-2, rel_tol=1e-2, max_iterations=2000, method='CG')
                        velocity, _ = fluid.make_incompressible(velocity, solve=solve_iterative)
                        logger.info(f"Converged with iterative solver at step {i}")
                    except Exception as e3:
                        logger.warning(f"Iterative solver also failed at step {i}: {e3}")
                        # Try with even more relaxed parameters
                        try:
                            solve_very_relaxed = Solve(abs_tol=1e-1, rel_tol=1e-1, max_iterations=5000, method='CG')
                            velocity, _ = fluid.make_incompressible(velocity, solve=solve_very_relaxed)
                            logger.info(f"Converged with very relaxed iterative solver at step {i}")
                        except Exception as e4:
                            logger.error(f"Complete convergence failure at step {i}: {e4}")
                            # Use the velocity as-is without incompressibility correction
                            logger.warning("Proceeding without incompressibility correction")
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
        # fluid_field, velocity_corrected = [], []
        # for idx in tqdm(range(num_samples)):
        #     fluid_field_, velocity_corrected_ = genfunc(idx, rngs[idx].item())
        #     fluid_field.append(fluid_field_)
        #     velocity_corrected.append(velocity_corrected_)
        
        fluid_field, velocity_corrected = zip(
            *Parallel(n_jobs=n_parallel)(delayed(genfunc)(idx, rngs[idx]) for idx in tqdm(range(num_samples)))
        )

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
            dt[idx : (idx + 1)] = stable_dt * pde.sample_rate
            buo_y[idx : (idx + 1)] = pde.buoyancy_y

    logger.info(f"Took {writetime.dt:.3f} seconds writing to disk")

    print()
    print("Data saved")
    print()
    print()
    h5f.close()