#include <functional>
#include <map>
#include <stdexcept>
#include <string>

#include "initialize_od.cuh"
#include "kernels_od.cuh"
#include "kernels_fluidbcs.cuh"

// I don't want to have a separate runtime file for each problem
class SimulationInitializer {
    private:
        using KernelLauncher = std::function<void(float*, const InitConfig&)>;
        std::map<std::string, KernelLauncher> initFunctions;
        InitConfig config;
    
    public:
        SimulationInitializer(const InitConfig& config) : config(config) {
            initFunctions["screwpinch"] = [this](float* data, const InitConfig& cfg) {
                LaunchScrewPinch(data, cfg); // Do not want to pass cfg to GPU or make this code less readable by passing long list of cfg parameters
            };
            initFunctions["screwpinch-stride"] = [this](float* data, const InitConfig& cfg) {
                LaunchScrewPinchStride(data, cfg);
            };
            /* ADD OTHER INITIALIZERS */
        } 
 
        void initialize(const std::string& simType, float* data){
            auto it = initFunctions.find(simType);
            if (it == initFunctions.end()) {
                throw std::runtime_error("Unknown simulation type: " + simType);
            }
            it->second(data, config);
        }
 };
 
// I don't want to have a separate runtime file for each possible choice of megakernels / microkernels
// Due to structure, it looks like I will need to create multiple separate instances of this class
class KernelConfigurer {
    private:
        using KernelLauncher = std::function<void(float*, const float*, const KernelConfig& kcfg)>;
        std::map<std::string, KernelLauncher> kernelFunctions;
        KernelConfig config;

    public:
        KernelConfigurer(const KernelConfig& kcfg) : config(kcfg) {
            kernelFunctions["fluidadvancelocal-nodiff"] = [this](float* fluidvars, const float *intvars, const KernelConfig& kcfg) {
                LaunchFluidAdvanceLocalNoDiff(fluidvars, intvars, kcfg); // Do not want to pass kcfg to GPU or make this code less readable by passing long list of params
            };
            /* ADD MORE BUNDLES OF KERNELS TO RUN 
            Examples: 
            (1) Block computation of Predictor variables
            (2) Different methods for the computation of Corrector Variables
            */
        }

        void LaunchKernels(const std::string& kBundle, float* fvars_or_intvars, const float* intvars_or_fvars){
            auto it = kernelFunctions.find(kBundle);
            if (it == kernelFunctions.end()) {
                throw std::runtime_error("Unknown kernel bundle selected: " + kBundle);
            }
            it->second(fvars_or_intvars, intvars_or_fvars, config);
        }
};
 
// See documentation for what each of these Boundary Conditions specify exactly
/* 
WHERE (?) 

There aren't that many different kinds of BCs, really. Two basic situations that I know are relevant to Ideal MHD are:
(1) Rigid, perfectly-conducting walls (PCRWs) + PBC (X-, Y-, or Z-direction)
(2) ALL PBCs

The first kind are particularly relevant to studying fusion equilibria, and the second are relevant to infinite problems like the Orszag-Tang vortex.

The reason for having a class, and wrappers, that call these kernels variously is to make the action of solving the boundary conditions modular. 
This provides the benefit of not needing a separate runtime file for each unique set of boundary conditions that are implemented.
*/
class BoundaryConfigurer {
    private:
        using KernelLauncher = std::function<void(float*, const int, const int, const int, const BoundaryConfig& bcfg)>;
        std::map<std::string, KernelLauncher> boundaryFunctions;
        BoundaryConfig config;

    public:
        BoundaryConfigurer(const BoundaryConfig& bcfg) : config(bcfg) {
            boundaryFunctions["pcrw-xy"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPCRWXY(fluidvars, Nx, Ny, Nz, bcfg); // Perfectly-conducting, rigid walls boundary conditions in x- and y-directions
            };
            /* FINALIZE */
            boundaryFunctions["pcrw-yz"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPCRWYZ(fluidvars, Nx, Ny, Nz, bcfg);
            }
            /* FINALIZE */
            boundaryFunctions["pcrw-xz"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPCRWXZ(fluidvars, Nx, Ny, Nz, bcfg);
            }
            boundaryFunctions["pbc-x"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPBCX(fluidvar, Nx, Ny, Nz, bcfg);
            };
            /* FINALIZE */
            boundaryFunctions["pbc-y"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPBCY(fluidvar, Nx, Ny, Nz, bcfg);
            };
            boundaryFunctions["pbc-z"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPBCZ(fluidvars, Nx, Ny, Nz, bcfg); // Periodic in z 
            };
            /* ADD MORE BOUNDARY CONDITIONS 
            For example, PBCs in every direction! (Orszag-Tang)
            */
        }

        /* Could we pick a better name for `bcBundle` (?) */
        void LaunchKernels(const std::string& bcBundle, float* fvars_or_intvars, const int Nx, const int Ny, const int Nz){
            auto it = boundaryFunctions.find(bcBundle);
            if (it == boundaryFunctions.end()) {
                throw std::runtime_error("Unknown bcs selected: " + bcBundle);
            }
            it->second(fvars_or_intvars, Nx, Ny, Nz, config);
        }
};