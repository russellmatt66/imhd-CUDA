#include <functional>
#include <map>
#include <stdexcept>
#include <string>

#include "initialize_od.cuh"
#include "kernels_od.cuh"
#include "kernels_fluidbcs.cuh"
#include "kernels_od_intvar.cuh"
#include "kernels_intvarbcs.cuh"

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

/* 
(CORRECTOR) KERNEL CONFIGURER 
*/
// I don't want to have a separate runtime file for each possible choice of megakernels / microkernels
class FluidKernelConfigurer {
    private:
        using KernelLauncher = std::function<void(float*, const float*, const KernelConfig& kcfg)>;
        std::map<std::string, KernelLauncher> kernelFunctions;
        KernelConfig config;

    public:
        FluidKernelConfigurer(const KernelConfig& kcfg) : config(kcfg) {
            kernelFunctions["fluidadvancelocal-nodiff"] = [this](float* fluidvars, const float *intvars, const KernelConfig& kcfg) {
                LaunchFluidAdvanceLocalNoDiff(fluidvars, intvars, kcfg); // Do not want to pass kcfg to GPU or make this code less readable by passing long list of params
            };
            /* ADD MORE BUNDLES OF KERNELS TO RUN 
            Examples: 
            (1) Block computation of Predictor variables
            (2) Different methods for the computation of Corrector Variables
            */
        }

        void LaunchKernels(const std::string& kBundle, float* fluidvars, const float* intvars){
            auto it = kernelFunctions.find(kBundle);
            if (it == kernelFunctions.end()) {
                throw std::runtime_error("Unknown kernel bundle selected: " + kBundle);
            }
            it->second(fluidvars, intvars, config);
        }
};
 
/*
(PREDICTOR) KERNEL CONFIGURER
The predictor step (speaking specifically about Lax-Wendroff) "looks" in the opposite direction of the corrector step for the finite difference
*/
class PredictorKernelConfigurer {
    private:
        /* COMPLETE */
        using KernelLauncher = std::function<void()>;
        std::map<std::string, KernelLauncher> kernelFunctions;
        KernelConfig config;

    public:
        /* COMPLETE */
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
class FluidBoundaryConfigurer {
    private:
        using KernelLauncher = std::function<void(float*, const int, const int, const int, const BoundaryConfig& bcfg)>;
        std::map<std::string, KernelLauncher> boundaryFunctions;
        BoundaryConfig config;

    public:
        FluidBoundaryConfigurer(const BoundaryConfig& bcfg) : config(bcfg) {
            boundaryFunctions["pbc-x"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPBCX(fluidvars, Nx, Ny, Nz, bcfg);
            };
            boundaryFunctions["pbc-y"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPBCY(fluidvars, Nx, Ny, Nz, bcfg);
            };
            boundaryFunctions["pbc-z"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPBCZ(fluidvars, Nx, Ny, Nz, bcfg); // Periodic in z 
            };
            boundaryFunctions["pcrw-front"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPCRWFront(fluidvars, Nx, Ny, Nz, bcfg);
            };
            boundaryFunctions["pcrw-back"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPCRWBack(fluidvars, Nx, Ny, Nz, bcfg);
            };
            boundaryFunctions["pcrw-left"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPCRWLeft(fluidvars, Nx, Ny, Nz, bcfg);
            };
            boundaryFunctions["pcrw-right"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPCRWRight(fluidvars, Nx, Ny, Nz, bcfg);
            };
            boundaryFunctions["pcrw-top"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPCRWTop(fluidvars, Nx, Ny, Nz, bcfg);
            };
            boundaryFunctions["pcrw-bottom"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPCRWBottom(fluidvars, Nx, Ny, Nz, bcfg);
            };
            boundaryFunctions["pcrw-xy"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPCRWXY(fluidvars, Nx, Ny, Nz, bcfg); // Perfectly-conducting, rigid walls boundary conditions in x- and y-directions
            };
            boundaryFunctions["pcrw-yz"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPCRWYZ(fluidvars, Nx, Ny, Nz, bcfg);
            };
            boundaryFunctions["pcrw-xz"] = [this](float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg) {
                LaunchFluidBCsPCRWXZ(fluidvars, Nx, Ny, Nz, bcfg);
            };
            /* ADD MORE BOUNDARY CONDITIONS 
            For example, PBCs in every direction! (Orszag-Tang)
            */
        }

        void LaunchKernels(const std::string& bcBundle, float* fluidvars, const int Nx, const int Ny, const int Nz){
            auto it = boundaryFunctions.find(bcBundle);
            if (it == boundaryFunctions.end()) {
                throw std::runtime_error("Unknown bcs selected: " + bcBundle);
            }
            it->second(fluidvars, Nx, Ny, Nz, config);
        }
};

/*
The predictor variables must be calculated on the boundaries due to the presence of fluxes in the system. 

Depending on the boundary conditions there are some points in the computational domain which might not influence the global solution, however, 
it is best practice to calculate the values everywhere because these points will change with the B.Cs so doing otherwise would invite subtle bugs.

The method by which the predictor variables are calculated on the boundaries depends on the specific way that the fluid (corrector) variables
are handled on the boundaries. 

An additional constraint exists due to the execution model of an NVIDIA GPU. Performant computation of the boundaries in CUDA requires microkernels
whose execution configurations are adjusted to compute along linear dimensions (1D), rather than rectangular as in the case of fluid BCs (2D), 
or rectangular prism as in the case of the predictor / corrector megakernels (3D). The way that an NVIDIA GPU works at a fundamental level (warps) means 
that it is very inefficient to assign thread teams to a problem that is lower-dimension than the threadblock they originate from.

For example, a 3D execution configuration that is launched with a megakernel to calculate the predictor variables EVERYWHERE in the computational domain
will either require so many (and convoluted) if statements to ensure that all the proper rules are implemented that the performance will evaporate 
in the face of thread divergence. If, rather than implementing a megakernel to calculate the intvars everywhere, instead a 3D grid of 
3D threadblocks was launched to compute the boundaries the performance would again dissipate due to the introduction of a large number of wasteful
memory access cycles as - due to the mismatched geometry - only a small fraction of the threads in each team would receive the necessary data each read.

Furthermore, it is not satisfactory to fuse the microkernels into one that is amenable to a 2D execution configuration, as can be done in the fluid BC case,
because the fluid variables are being SPECIFIED on the boundaries whereas the intermediate (predictor) variables are being calculated there so that the
nearest-neighbor interior points can get the fluxes that they need in order to be updated in a valid manner. 

Consequently, there is a different rule necessary for how to update the "spatchcocked" sets of points in the corners of the cartesian computational grid 
which find themselves nearest neighbors to infinity on two, and sometimes even three, sides. Care must be taken to avoid OOB accesses depending on 
the exact advance equations that are implemented. 
*/
// Non-blocking launchers - don't want to place this code in main file, and don't want to introduce CUDA dependencies here for synchronization
class PredictorBoundaryConfigurer {
    private:
        using KernelLauncher = std::function<void(const float* , float*, const int, const int, const int, const IntvarBoundaryConfig&)>;
        std::map<std::string, KernelLauncher> intvarBoundaryFunctions;
        IntvarBoundaryConfig config;
    public:
        /* COMPLETE */
        PredictorBoundaryConfigurer(const IntvarBoundaryConfig& ibcfg) : config(ibcfg) {
            intvarBoundaryFunctions["pbc-z"] = [this](const float* fluidvars, float* intvars, const int Nx, const int Ny, const int Nz, const IntvarBoundaryConfig& ibcfg) {
                LaunchIntvarsBCsPBCZ(fluidvars, intvars, Nx, Ny, Nz, ibcfg); // Blocking, two-step, microkernel mixture   
            };
        }

        void LaunchKernels(const std::string& intvarsBCsBundle, const float* fluidvars, float* intvars, const int Nx, const int Ny, const int Nz){
            auto it = intvarBoundaryFunctions.find(intvarsBCsBundle);
            if (it == intvarBoundaryFunctions.end()) {
                throw std::runtime_error("Unknown bcs selected: " + intvarsBCsBundle);
            }
            it->second(fluidvars, intvars, Nx, Ny, Nz, config);
        }
};