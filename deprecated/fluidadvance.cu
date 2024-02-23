__global__ void FluidAdvance(float* rho_np1, float* rhovx_np1, float* rhovy_np1, float* rhovz_np1, float* Bx_np1, float* By_np1, float* Bz_np1, float* e_np1,
     const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, const float* By, const float* Bz, const float* e,
     const float D, const float dt, const float dx, const float dy, const float dz, 
     const int Nx, const int Ny, const int Nz)
    {
        // Execution configuration boilerplate
        int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;
        int xthreads = gridDim.x * blockDim.x;
        int ythreads = gridDim.y * blockDim.y;
        int zthreads = gridDim.z * blockDim.z;

        /* 
        This all is getting re-declared every timestep 
        Can a single thread even handle all of this?
        RTX 2060: 64KB per SM
        */
        // Hoist fluid variables
        float t_rho = 0.0;
        float t_rhov_x = 0.0, t_rhov_y = 0.0, t_rhov_z = 0.0;
        float t_Bx = 0.0, t_By = 0.0, t_Bz = 0.0;
        float t_p = 0.0, t_e = 0.0; 
        float t_KE = 0.0, t_B_sq = 0.0, t_B_dot_u = 0.0;

        // Hoist Fluid Fluxes 
        float t_xflux_rho = 0.0, t_yflux_rho = 0.0, t_zflux_rho = 0.0;
        float t_xflux_rhovx = 0.0, t_yflux_rhovx = 0.0, t_zflux_rhovx = 0.0;
        float t_xflux_rhovy = 0.0, t_yflux_rhovy = 0.0, t_zflux_rhovy = 0.0;
        float t_xflux_rhovz = 0.0, t_yflux_rhovz = 0.0, t_zflux_rhovz = 0.0;
        float t_xflux_Bx = 0.0, t_yflux_Bx = 0.0, t_zflux_Bx = 0.0;
        float t_xflux_By = 0.0, t_yflux_By = 0.0, t_zflux_By = 0.0;
        float t_xflux_Bz = 0.0, t_yflux_Bz = 0.0, t_zflux_Bz = 0.0;
        float t_xflux_e = 0.0, t_yflux_e = 0.0, t_zflux_e = 0.0;
        
        // Hoist Intermediate Variables
        // \bar{Q}_{ijk}
        float t_int_rho = 0.0;
        float t_int_rhov_x = 0.0, t_int_rhov_y = 0.0, t_int_rhov_z = 0.0;
        float t_int_Bx = 0.0, t_int_By = 0.0, t_int_Bz = 0.0;
        float t_int_e = 0.0;
        // float t_int_p = 0.0;
        // float t_int_KE = 0.0, t_int_B_sq = 0.0;

        // \bar{Q}_{i+1,j,k}
        float t_int_rho_ip1 = 0.0;
        float t_int_rhov_x_ip1 = 0.0, t_int_rhov_y_ip1 = 0.0, t_int_rhov_z_ip1 = 0.0;
        float t_int_Bx_ip1 = 0.0, t_int_By_ip1 = 0.0, t_int_Bz_ip1 = 0.0;
        float t_int_e_ip1 = 0.0;
        
        // \bar{Q}_{i,j+1,k}
        float t_int_rho_jp1 = 0.0;
        float t_int_rhov_x_jp1 = 0.0, t_int_rhov_y_jp1 = 0.0, t_int_rhov_z_jp1 = 0.0;
        float t_int_Bx_jp1 = 0.0, t_int_By_jp1 = 0.0, t_int_Bz_jp1 = 0.0;
        float t_int_e_jp1 = 0.0;
        
        // \bar{Q}_{i,j,k+1}
        float t_int_rho_kp1 = 0.0;
        float t_int_rhov_x_kp1 = 0.0, t_int_rhov_y_kp1 = 0.0, t_int_rhov_z_kp1 = 0.0;
        float t_int_Bx_kp1 = 0.0, t_int_By_kp1 = 0.0, t_int_Bz_kp1 = 0.0;
        float t_int_e_kp1 = 0.0;

        //Hoist Intermediate Fluxes
        // With \bar{Q}_{ijk}
        float t_int_xflux_rho = 0.0, t_int_yflux_rho = 0.0, t_int_zflux_rho = 0.0;
        float t_int_xflux_rhovx = 0.0, t_int_yflux_rhovx = 0.0, t_int_zflux_rhovx = 0.0;
        float t_int_xflux_rhovy = 0.0, t_int_yflux_rhovy = 0.0, t_int_zflux_rhovy = 0.0;
        float t_int_xflux_rhovz = 0.0, t_int_yflux_rhovz = 0.0, t_int_zflux_rhovz = 0.0;
        float t_int_xflux_Bx = 0.0, t_int_yflux_Bx = 0.0, t_int_zflux_Bx = 0.0;
        float t_int_xflux_By = 0.0, t_int_yflux_By = 0.0, t_int_zflux_By = 0.0;
        float t_int_xflux_Bz = 0.0, t_int_yflux_Bz = 0.0, t_int_zflux_Bz = 0.0;
        float t_int_xflux_e = 0.0, t_int_yflux_e = 0.0, t_int_zflux_e = 0.0;

        // With \bar{Q}_{i+1,jk}
        float t_int_xflux_rho_ip1 = 0.0;
        float t_int_xflux_rhovx_ip1 = 0.0;
        float t_int_xflux_rhovy_ip1 = 0.0;
        float t_int_xflux_rhovz_ip1 = 0.0;
        float t_int_xflux_Bx_ip1 = 0.0;
        float t_int_xflux_By_ip1 = 0.0;
        float t_int_xflux_Bz_ip1 = 0.0;
        float t_int_xflux_e_ip1 = 0.0;

        // With \bar{Q}_{i,j+1,k}
        float t_int_yflux_rho_jp1 = 0.0;
        float t_int_yflux_rhovx_jp1 = 0.0;
        float t_int_yflux_rhovy_jp1 = 0.0;
        float t_int_yflux_rhovz_jp1 = 0.0;
        float t_int_yflux_Bx_jp1 = 0.0;
        float t_int_yflux_By_jp1 = 0.0;
        float t_int_yflux_Bz_jp1 = 0.0;
        float t_int_yflux_e_jp1 = 0.0;

        // With \bar{Q}_{i,j,k+1}
        float t_int_zflux_rho_kp1 = 0.0;
        float t_int_zflux_rhovx_kp1 = 0.0;
        float t_int_zflux_rhovy_kp1 = 0.0;
        float t_int_zflux_rhovz_kp1 = 0.0;
        float t_int_zflux_Bx_kp1 = 0.0;
        float t_int_zflux_By_kp1 = 0.0;
        float t_int_zflux_Bz_kp1 = 0.0;
        float t_int_zflux_e_kp1 = 0.0;

        // Handle B.Cs separately
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){ // THIS LOOP ORDER IS FOR CONTIGUOUS MEMORY ACCESS
            for (int i = tidx + 1; i < Nx - 1; i += xthreads){ 
                for (int j = tidy + 1; j < Ny - 1; j += ythreads){
                    /* 
                    Compute p, B^2, \vec{B}\dot\vec{u}, and the hoisted fluid variables 
                    */
                    t_rho = rho[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_rhov_x = rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_rhov_y = rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_rhov_z = rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_KE = KE(i, j, k, rho, rhov_x, rhov_y, rhov_z, Nx, Ny, Nz); // I know this is overly verbose, current concern is get it working and correct 
                    
                    t_Bx = Bx[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_By = By[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_Bz = Bz[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_B_sq = B_sq(i, j, k, Bx, By, Bz, Nx, Ny, Nz);
                    
                    t_e = e[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_p = p(i, j, k, e, t_B_sq, t_KE, Nx, Ny, Nz);
                    t_B_dot_u = B_dot_u(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, Nx, Ny, Nz);
                    
                    /* 
                    Compute fluid fluxes 
                    */
                    t_xflux_rho = XFluxRho(i, j, k, rhov_x, Nx, Ny, Nz);
                    t_yflux_rho = YFluxRho(i, j, k, rhov_y, Nx, Ny, Nz);
                    t_zflux_rho = ZFluxRho(i, j, k, rhov_z, Nx, Ny, Nz);

                    t_xflux_rhovx = XFluxRhoVX(t_rho, t_rhov_x, t_Bx, t_B_sq, t_p);
                    t_yflux_rhovx = YFluxRhoVX(t_rho, t_rhov_x, t_rhov_y, t_Bx, t_By);
                    t_zflux_rhovx = ZFluxRhoVX(t_rho, t_rhov_x, t_rhov_z, t_Bx, t_Bz);

                    t_xflux_rhovy = XFluxRhoVY(t_rho, t_rhov_x, t_rhov_y, t_Bx, t_By);
                    t_yflux_rhovy = YFluxRhoVY(t_rho, t_rhov_y, t_By, t_B_sq, t_p);
                    t_zflux_rhovy = ZFluxRhoVY(t_rho, t_rhov_y, t_rhov_z, t_By, t_Bz);

                    t_xflux_rhovz = XFluxRhoVZ(t_rho, t_rhov_x, t_rhov_z, t_Bx, t_Bz);
                    t_yflux_rhovz = YFluxRhoVZ(t_rho, t_rhov_y, t_rhov_z, t_By, t_Bz);
                    t_zflux_rhovz = ZFluxRhoVZ(t_rho, t_rhov_z, t_Bz, t_B_sq, t_p);

                    t_xflux_Bx = XFluxBX();
                    t_yflux_Bx = YFluxBX(t_rho, t_rhov_x, t_rhov_y, t_Bx, t_By); 
                    t_zflux_Bx = ZFluxBX(t_rho, t_rhov_x, t_rhov_z, t_Bx, t_Bz); 

                    t_xflux_By = XFluxBY(t_rho, t_rhov_x, t_rhov_y, t_Bx, t_By); 
                    t_yflux_By = YFluxBY(); 
                    t_zflux_By = ZFluxBY(t_rho, t_rhov_y, t_rhov_z, t_By, t_Bz); 

                    t_xflux_Bz = XFluxBZ(t_rho, t_rhov_x, t_rhov_z, t_Bx, t_Bz); 
                    t_yflux_Bz = YFluxBZ(t_rho, t_rhov_y, t_rhov_z, t_By, t_Bz); 
                    t_zflux_Bz = ZFluxBZ(); 

                    t_xflux_e = XFluxE(t_rho, t_rhov_x, t_Bx, t_e, t_p, t_B_sq, t_B_dot_u); 
                    t_yflux_e = YFluxE(t_rho, t_rhov_y, t_By, t_e, t_p, t_B_sq, t_B_dot_u); 
                    t_zflux_e = ZFluxE(t_rho, t_rhov_z, t_Bz, t_e, t_p, t_B_sq, t_B_dot_u); 

                    /* 
                    Compute intermediate variables 
                    \bar{Q}_{ijk} = Q^{n}_{ijk} - (dt / dx) * (F^{n}_{ijk} - F^{n}_{i-1,j,k}) 
                                        - (dt / dy) * (G^{n}_{ijk} - G^{n}_{i,j-1,k})
                                        - (dt / dz) * (H^{n}_{ijk} - H^{n}_{i,j,k-1})
                    
                    NOTE - ALSO NEED:
                    \bar{Q}_{i+1,j,k}
                    \bar{Q}_{i,j+1,k}
                    \bar{Q}_{i,j,k+1}
                    For fluid variable update
                    */
                    // THESE ARE ALL \bar{Q}_{ijk}
                    t_int_rho = t_rho 
                        - (dt / dx) * (t_xflux_rho - XFluxRho(i-1, j, k, rhov_x, Nx, Ny, Nz))
                        - (dt / dy) * (t_yflux_rho - YFluxRho(i, j-1, k, rhov_y, Nx, Ny, Nz))   
                        - (dt / dz) * (t_zflux_rho - ZFluxRho(i, j, k-1, rhov_z, Nx, Ny, Nz));
                    
                    t_int_rhov_x = t_rhov_x
                        - (dt / dx) * (t_xflux_rhovx 
                            - XFluxRhoVX(i-1, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz))
                        - (dt / dy) * (t_yflux_rhovx
                            - YFluxRhoVX(i, j-1, k, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz))
                        - (dt / dz) * (t_zflux_rhovx
                            - ZFluxRhoVX(i, j, k-1, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz));
                    
                    t_int_rhov_y = t_rhov_y
                        - (dt / dx) * (t_xflux_rhovy
                            - XFluxRhoVY(i-1, j, k, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz))
                        - (dt / dy) * (t_yflux_rhovy
                            - YFluxRhoVY(i, j-1, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz))
                        - (dt / dz) * (t_zflux_rhovy
                            - ZFluxRhoVY(i, j, k-1, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz));
                    
                    t_int_rhov_z = t_rhov_z
                        - (dt / dx) * (t_xflux_rhovz
                            - XFluxRhoVZ(i-1, j, k, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz))
                        - (dt / dy) * (t_yflux_rhovz
                            - YFluxRhoVZ(i, j-1, k, rho, rhov_x, rhov_z, By, Bz, Nx, Ny, Nz))
                        - (dt / dz) * (t_zflux_rhovz
                            - ZFluxRhoVZ(i, j, k-1, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz));
                    
                    t_int_Bx = t_Bx
                        - (dt / dx) * (t_xflux_Bx 
                            - XFluxBX())
                        - (dt / dy) * (t_yflux_Bx
                            - YFluxBX(i, j-1, k, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz))
                        - (dt / dz) * (t_zflux_Bx
                            - ZFluxBX(i, j, k-1, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz));
                    
                    t_int_By = t_By
                        - (dt / dx) * (t_xflux_By 
                            - XFluxBY(i-1, j, k, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz))
                        - (dt / dy) * (t_yflux_By
                            - YFluxBY())
                        - (dt / dz) * (t_zflux_By
                            - ZFluxBY(i, j, k-1, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz));

                    t_int_Bz = t_Bz
                        - (dt / dx) * (t_xflux_Bz 
                            - XFluxBZ(i-1, j, k, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz))
                        - (dt / dy) * (t_yflux_Bz
                            - YFluxBZ(i, j-1, k, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz))
                        - (dt / dz) * (t_zflux_Bz
                            - ZFluxBZ());   
                    
                    t_int_e = t_e
                        - (dt / dx) * (t_xflux_e
                            - XFluxE(i-1, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz))
                        - (dt / dy) * (t_yflux_e
                            - YFluxE(i, j-1, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz))
                        - (dt / dz) * (t_zflux_e
                            - ZFluxE(i, j, k-1, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz));

                    /* ADD \bar{Q}_{i+1,j,k} */
                    /* ADD \bar{Q}_{i,j+1,k} */
                    /* ADD \bar{Q}_{i,j,k+1} */

                    /*
                    TODO: Compute intermediate fluxes
                    Call flux functions on intermediate variables
                    */
                    t_int_xflux_rho = 0.0; 
                    t_int_xflux_rho_ip1 = 0.0;
                    
                    t_int_yflux_rho = 0.0;
                    t_int_yflux_rho_jp1 = 0.0;

                    t_int_zflux_rho = 0.0;
                    t_int_zflux_rho_kp1 = 0.0;

                    t_int_xflux_rhovx = 0.0; 
                    t_int_xflux_rhovx_ip1 = 0.0;
                    
                    t_int_yflux_rhovx = 0.0;
                    t_int_yflux_rhovx_jp1 = 0.0;

                    t_int_zflux_rhovx = 0.0;
                    t_int_zflux_rhovx_kp1 = 0.0;
                    
                    t_int_xflux_rhovy = 0.0; 
                    t_int_xflux_rhovy_ip1 = 0.0;
                    
                    t_int_yflux_rhovy = 0.0;
                    t_int_yflux_rhovy_jp1 = 0.0;

                    t_int_zflux_rhovy = 0.0;
                    t_int_zflux_rhovy_kp1 = 0.0;

                    t_int_xflux_rhovz = 0.0; 
                    t_int_xflux_rhovz_ip1 = 0.0;
                    
                    t_int_yflux_rhovz = 0.0;
                    t_int_yflux_rhovz_jp1 = 0.0;

                    t_int_zflux_rhovz = 0.0;
                    t_int_zflux_rhovz_kp1 = 0.0;
                    
                    t_int_xflux_Bx = 0.0; 
                    t_int_xflux_Bx_ip1 = 0.0; 

                    t_int_yflux_Bx = 0.0; 
                    t_int_yflux_Bx_jp1 = 0.0; 

                    t_int_zflux_Bx = 0.0;
                    t_int_zflux_Bx_kp1 = 0.0;

                    t_int_xflux_By = 0.0; 
                    t_int_xflux_By_ip1 = 0.0; 

                    t_int_yflux_By = 0.0; 
                    t_int_yflux_By_jp1 = 0.0; 

                    t_int_zflux_By = 0.0;
                    t_int_zflux_By_kp1 = 0.0;
                    
                    t_int_xflux_Bz = 0.0;
                    t_int_xflux_Bz_ip1 = 0.0;

                    t_int_yflux_Bz = 0.0; 
                    t_int_yflux_Bz_jp1 = 0.0; 

                    t_int_zflux_Bz = 0.0;
                    t_int_zflux_Bz_kp1 = 0.0;
                    
                    t_int_xflux_e = 0.0; 
                    t_int_xflux_e_ip1 = 0.0; 

                    t_int_yflux_e = 0.0; 
                    t_int_yflux_e_jp1 = 0.0; 

                    t_int_zflux_e = 0.0;
                    t_int_zflux_e_kp1 = 0.0;

                    /* 
                    TODO: Update fluid variables 
                    Q^{n+1}_{ijk} = 0.5 * (Q^{n}_{ijk} + \bar{Q}_{ijk})
                        - (dt / 2*dx) * (\bar{F}_{i+1,j,k} - \bar{F}_{ijk})
                        - (dt / 2*dy) * (\bar{G}_{i,j+1,k} - \bar{G}_{ijk})
                        - (dt / d*dz) * (\bar{H}_{i,j,k+1} - \bar{H}_{ijk})
                    */
                }
            }
        } 
        return;
    }