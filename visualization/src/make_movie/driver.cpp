#include <iostream>
#include <string>

#include <vtkCamera.h>
#include <vtkColorTransferFunction.h>
#include <vtkFFMPEGWriter.h>
#include <vtkImageImport.h>
#include <vtkNew.h>
#include <vtkPiecewiseFunction.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkScalarBarActor.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>
#include <vtkWindowToImageFilter.h>

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char* argv[]){
    std::cout << "Inside make_movie driver" << std::endl;

    std::string path_to_data = argv[1];
    std::string dset_name = argv[2]; // specific fluid variable to visualize
    std::string filename_grid = argv[3];
    size_t Nt = atoi(argv[4]);

    float frametext_x = atof(argv[5]);
    float frametext_y = atof(argv[6]);
    int frametext_fontsize = atoi(argv[7]);
    float frametext_red = atof(argv[8]);
    float frametext_green= atof(argv[9]);
    float frametext_blue = atof(argv[10]);
    int scalarbar_textpad = atoi(argv[11]);
    int scalarbar_numlabels = atoi(argv[12]);
    int scalarbar_maxnumcolors = atoi(argv[13]);
    float scalarbar_barratio = atof(argv[14]);
    float scalarbar_xll = atof(argv[15]);
    float scalarbar_yll = atof(argv[16]);
    float scalarbar_dxur = atof(argv[17]);
    float scalarbar_dyur = atof(argv[18]);
    int scalarbar_labelfontsize = atoi(argv[19]);
    int scalarbar_titlefontsize = atoi(argv[20]);
    float camera_x = atof(argv[21]);
    float camera_y = atof(argv[22]);
    float camera_z = atof(argv[23]);
    float camera_focalx = atof(argv[24]);
    float camera_focaly = atof(argv[25]);
    float camera_focalz = atof(argv[26]);
    float camera_viewupx = atof(argv[27]);
    float camera_viewupy = atof(argv[28]);
    float camera_viewupz = atof(argv[29]);
    float renderer_backgroundred = atof(argv[30]);
    float renderer_backgroundgreen = atof(argv[31]);
    float renderer_backgroundblue = atof(argv[32]);
    int videowriter_fps = atoi(argv[33]);
    float opacity_min = atof(argv[34]);
    float opacity_max = atof(argv[35]);

    std::string dt = argv[36];
    std::string D = argv[37];

    /* REPLACE THIS WITH LIBRARY FUNCTION */
    // Get mesh dimensions, and spacing
    // Allocate shared memory that will store the necessary attributes
    // = Nx, Ny, Nz, dx, dy, dz
    // Shared memory = workaround b/c VTK and .h5 files don't mix without effort to mix them
    std::string shm_gridattr_name = "/shm_grid_attributes";
    int shm_fd_gridattr = shm_open(shm_gridattr_name.data(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_gridattr == -1){
        std::cerr << "Failed to create shared memory for grid attributes" << std::endl;
        return 1;
    }

    size_t gridattr_data_size = sizeof(float) * 6;
    ftruncate(shm_fd_gridattr, gridattr_data_size); // Nx, Ny, Nz, ...

    float* shm_gridattr = (float*)mmap(0, gridattr_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_gridattr, 0);
    if (shm_gridattr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory for grid attributes" << std::endl;
        return 1;
    }

    // Fork process to obtain necessary grid attributes
    std::string gridattr_command = "./read_grid_attr " + shm_gridattr_name + " " + filename_grid + " " + std::to_string(gridattr_data_size);
    std::cout << "Forking to process for obtaining grid attributes" << std::endl;
    int ret = std::system(gridattr_command.data());
    if (ret != 0) {
        std::cerr << "Error executing command: " << gridattr_command << std::endl;
        return 1;
    }

    int Nx = (int)shm_gridattr[0], Ny = (int)shm_gridattr[1], Nz = (int)shm_gridattr[2];
    float dx = shm_gridattr[3], dy = shm_gridattr[4], dz = shm_gridattr[5];

    std::cout << "Returned successfully from process. Attribute values are: " << std::endl;
    std::cout << "(Nx, Ny, Nz) = " << "(" << Nx << "," << Ny << "," << Nz << ")" << std::endl;
    std::cout << "(dx, dy, dz) = " << "(" << dx << "," << dy << "," << dz << ")" << std::endl; 

    /* REPLACE THIS WITH LIBRARY FUNCTION */
    // Allocate shared memory that will contain the fluidvar data 
    std::cout << "Allocating shared memory for fluidvar data" << std::endl;
    std::string shm_fluidvar_name = "/shm_fluidvar_data";
    int shm_fd_fluidvar = shm_open(shm_fluidvar_name.data(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_fluidvar == -1){
        std::cerr << "Failed to create shared memory for fluidvar data" << std::endl;
        return 1;
    }

    size_t fluid_data_size = sizeof(float) * Nx * Ny * Nz;
    ftruncate(shm_fd_fluidvar, fluid_data_size);

    float* shm_fluidvar = (float*)mmap(0, fluid_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_fluidvar, 0);
    if (shm_fluidvar == MAP_FAILED) {
        std::cerr << "Failed to map shared memory for fluidvar data" << std::endl;
        return 1;
    }
    std::cout << "Shared memory for fluidvar data successfully allocated" << std::endl;

    // VTK
    std::cout << "Instantiating VTK objects" << std::endl; 

    std::cout << "Instantiating frame_data" << std::endl;
    vtkNew<vtkImageImport> frame_data;
    frame_data->SetDataSpacing(dx, dy, dz);
    frame_data->SetDataOrigin(0, 0, 0);
    frame_data->SetWholeExtent(0, Nx - 1, 0, Ny - 1, 0, Nz - 1);
    frame_data->SetDataExtentToWholeExtent();
    frame_data->SetDataScalarTypeToFloat();  
    frame_data->SetNumberOfScalarComponents(1);
    std::cout << "frame_data instantiated" << std::endl;

    std::cout << "Instantiating volumeMapper" << std::endl;
    vtkNew<vtkSmartVolumeMapper> volumeMapper;
    volumeMapper->SetInputConnection(frame_data->GetOutputPort());
    std::cout << "volumeMapper instantiated, input connection to frame_data set" << std::endl;

    std::cout << "Instantiating volumeActor" << std::endl;
    vtkNew<vtkVolume> volumeActor;
    volumeActor->SetMapper(volumeMapper);
    std::cout << "volumeActor instantiated, mapper set to volumeMapper" << std::endl;

    std::cout << "Instantiating textActor" << std::endl;
    vtkNew<vtkTextActor> frameText;
    frameText->GetPositionCoordinate()->SetCoordinateSystemToNormalizedDisplay();
    frameText->SetPosition(frametext_x, frametext_y);
    frameText->GetTextProperty()->SetFontSize(frametext_fontsize);
    frameText->GetTextProperty()->SetColor(frametext_red, frametext_green, frametext_blue);
    std::cout << "textActor instantiated" << std::endl;

    /* Didn't show up */
    // std::cout << "Instantiating textActor for simulation parameters" << std::endl;
    // vtkNew<vtkTextActor> frameText_simparams;
    // frameText_simparams->GetPositionCoordinate()->SetCoordinateSystemToNormalizedDisplay();
    // frameText_simparams->SetPosition(frametext_x, frametext_y - 0.1);
    // frameText_simparams->GetTextProperty()->SetFontSize(frametext_fontsize);
    // frameText_simparams->GetTextProperty()->SetColor(frametext_red, frametext_green, frametext_blue);
    // std::string simparams_text = "dt=" + dt + "\nD=" + D + "\n";
    // std::cout << "textActor for simparams instantiated" << std::endl;

    std::cout << "Instantiating colorTransfer" << std::endl;
    vtkNew<vtkColorTransferFunction> colorTF;
    std::cout << "colorTransfer instantiated" << std::endl;

    // Create color bar and set its properties
    vtkNew<vtkScalarBarActor> scalarBar;
    scalarBar->SetLookupTable(colorTF);
    scalarBar->SetTitle(dset_name.data());

    scalarBar->SetTextPositionToPrecedeScalarBar();
    scalarBar->SetTextPad(scalarbar_textpad);

    scalarBar->SetNumberOfLabels(scalarbar_numlabels);
    scalarBar->SetMaximumNumberOfColors(scalarbar_maxnumcolors);
    scalarBar->SetBarRatio(scalarbar_barratio);
    scalarBar->SetUnconstrainedFontSize(true);
    scalarBar->SetPosition(scalarbar_xll, scalarbar_yll);
    scalarBar->SetPosition2(scalarbar_dxur, scalarbar_dyur);

    vtkTextProperty* scalarBarLabelProp = scalarBar->GetLabelTextProperty();
    scalarBarLabelProp->SetFontSize(scalarbar_labelfontsize);

    vtkTextProperty* scalarBarTitleProp = scalarBar->GetTitleTextProperty();
    scalarBarTitleProp->SetFontSize(scalarbar_titlefontsize);
    // scalarBarTitleProp->SetJustification(VTK_TEXT_CENTERED);

    std::cout << "Instantiating opacityTransfer" << std::endl;
    vtkNew<vtkPiecewiseFunction> opacityTF;
    // opacityTransfer->AddPoint(min_val, 0.0);  // Adjust opacity points based on your data
    // opacityTransfer->AddPoint(max_val, 1.0);
    std::cout << "opacityTransfer instantiated" << std::endl;

    volumeActor->GetProperty()->SetScalarOpacity(opacityTF);
    volumeActor->GetProperty()->SetColor(colorTF);

    std::cout << "Instantiating renderer, and renderWindow" << std::endl;
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> renderWindow;
    renderWindow->AddRenderer(renderer);
    // renderWindow->Render();
    std::cout << "renderer, and renderWindow instantiated" << std::endl;

    std::cout << "Instantiating camera" << std::endl;
    vtkNew<vtkCamera> camera;
    camera->SetPosition(camera_x, camera_y, camera_z);
    camera->SetFocalPoint(camera_focalx, camera_focaly, camera_focalz);
    camera->SetViewUp(camera_viewupx, camera_viewupy, camera_viewupz);
    std::cout << "camera instantiated" << std::endl;

    std::cout << "Connecting camera, and volumeActor to renderer" << std::endl;
    renderer->SetActiveCamera(camera);
    renderer->SetBackground(renderer_backgroundred, renderer_backgroundgreen, renderer_backgroundblue);
    renderer->AddVolume(volumeActor);
    renderer->AddActor(frameText);
    renderer->AddActor2D(scalarBar);
    std::cout << "camera and volumeActor connected to renderer" << std::endl;
    
    std::cout << "Instantiating windowToImageFilter" << std::endl;
    vtkNew<vtkWindowToImageFilter> windowToImageFilter;
    windowToImageFilter->SetInput(renderWindow);
    // windowToImageFilter->Update();
    std::cout << "windowToImageFilter instantiated - renderWindow connected" << std::endl;

    std::cout << "Instantiating videoWriter" << std::endl;
    std::string video_filename = "../" + dset_name 
        + "_Nx" + std::to_string(Nx) + "Ny" + std::to_string(Ny) + "Nz" + std::to_string(Nz) 
        + "_x" + std::to_string(static_cast<int>(camera_x)) + "y" + std::to_string(static_cast<int>(camera_y)) + "z" + std::to_string(static_cast<int>(camera_z)) 
        + ".avi"; // Being run from inside 'build/'
    vtkNew<vtkFFMPEGWriter> videoWriter;
    videoWriter->SetInputConnection(windowToImageFilter->GetOutputPort());
    videoWriter->SetFileName(video_filename.data());
    videoWriter->SetRate(videowriter_fps); /* Speed this up later */
    videoWriter->Start();
    std::cout << "videoWriter instantiated" << std::endl;   

    // Loop over frames, and make video
    std::string load_fluidvar_command = "";
    std::string fluidvar_filename = ""; // container for frame text data in string form

    float min_val = std::numeric_limits<float>::lowest();
    float max_val = std::numeric_limits<float>::max();
    float val_delta = max_val - min_val;

    for (int i = 0; i <= Nt; i++){
        std::cout << "Writing frame " << i << " to video" << std::endl;

        // Fork process to load fluidvar data from .h5 file
        fluidvar_filename = path_to_data + "fluidvars_" + std::to_string(i) + ".h5";
        std::cout << "fluidvar_filename is: " << fluidvar_filename << std::endl;

        load_fluidvar_command = "./read_fluid_data " + shm_fluidvar_name + " " + dset_name + " " + fluidvar_filename 
            + " " + std::to_string(Nx) + " " + std::to_string(Ny) + " " + std::to_string(Nz);

        std::cout << "Forking to read_fluid_data binary with: " << load_fluidvar_command << std::endl;
        ret = std::system(load_fluidvar_command.data());
        if (ret != 0) {
           std::cerr << "Error executing command: " << load_fluidvar_command << std::endl;
            return 1;
        }

        min_val = *std::min_element(shm_fluidvar, shm_fluidvar + Nx * Ny * Nz);
        max_val = *std::max_element(shm_fluidvar, shm_fluidvar + Nx * Ny * Nz);

        std::cout << "Min val is: " << min_val << std::endl;
        std::cout << "Max val is: " << max_val << std::endl;

        val_delta = max_val - min_val;

        colorTF->RemoveAllPoints();
        colorTF->AddRGBPoint(min_val, 1.0, 0.0, 0.0);   // Red
        colorTF->AddRGBPoint(min_val + 0.1667 * val_delta, 1.0, 0.5, 0.0); // Orange
        colorTF->AddRGBPoint(min_val + 0.3333 * val_delta, 1.0, 1.0, 0.0); // Yellow
        colorTF->AddRGBPoint(min_val + 0.5 * val_delta, 0.0, 1.0, 0.0);    // Green
        colorTF->AddRGBPoint(min_val + 0.6667 * val_delta, 0.0, 0.0, 1.0); // Blue
        colorTF->AddRGBPoint(min_val + 0.8333 * val_delta, 0.5, 0.0, 1.0); // Indigo
        colorTF->AddRGBPoint(max_val, 1.0, 0.0, 1.0);       // Violet

        opacityTF->RemoveAllPoints();
        opacityTF->AddPoint(min_val, opacity_min);
        opacityTF->AddPoint(max_val, opacity_max);

        // Modify data, and write window to video
        frame_data->SetImportVoidPointer(shm_fluidvar);
        frame_data->Modified();

        fluidvar_filename += "\n"; // No reason not to use this as the container - maybe needs a better name
        fluidvar_filename += dset_name + " frame " + std::to_string(i) + " dt=" + dt + " D=" + D;
        frameText->SetInput(fluidvar_filename.data());
        // frameText_simparams->SetInput(simparams_text.data());

        renderWindow->Render();

        windowToImageFilter->Modified();
        windowToImageFilter->Update();

        videoWriter->Write();
    }

    std::cout << "Frames successfully written, ending videoWriter" << std::endl;
    videoWriter->End();

    // Manage memory
    std::cout << "Freeing shared memory" << std::endl;
    shm_unlink(shm_gridattr_name.data());
    shm_unlink(shm_fluidvar_name.data());
    munmap(shm_gridattr, gridattr_data_size);
    munmap(shm_fluidvar, fluid_data_size);
    close(shm_fd_gridattr);
    close(shm_fd_fluidvar);
    return 0;
}