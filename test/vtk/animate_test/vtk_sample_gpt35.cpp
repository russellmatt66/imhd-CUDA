#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkOBJReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkAnimationCue.h>
#include <vtkAnimationScene.h>
#include <vtkPNGWriter.h>

/* CHAT-GPT 3.5 Example Code for VTK Visualization Pipeline to Animate 3D Data */
int main(int, char *[])
{
    // Read 3D dataset (replace 'path/to/your/dataset.obj' with the actual path)
    vtkSmartPointer<vtkOBJReader> reader =
        vtkSmartPointer<vtkOBJReader>::New();
    reader->SetFileName("path/to/your/dataset.obj");

    // Map dataset to graphics primitives
    vtkSmartPointer<vtkPolyDataMapper> mapper =
        vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());

    // Create an actor
    vtkSmartPointer<vtkActor> actor =
        vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Create a renderer and add the actor
    vtkSmartPointer<vtkRenderer> renderer =
        vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(actor);

    // Create a render window
    vtkSmartPointer<vtkRenderWindow> renderWindow =
        vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    // Create an interactor
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
        vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Set up the animation
    vtkSmartPointer<vtkAnimationScene> scene =
        vtkSmartPointer<vtkAnimationScene>::New();
    scene->SetModeToSequence();

    // Set up camera positions (replace with your desired camera positions)
    vtkSmartPointer<vtkAnimationCue> cue =
        vtkSmartPointer<vtkAnimationCue>::New();
    for (int i = 0; i < 360; i += 10)
    {
        vtkSmartPointer<vtkCamera> camera =
            vtkSmartPointer<vtkCamera>::New();
        camera->Azimuth(i);
        cue->AddCamera(camera);
        cue->SetTime(i, i + 1); // Set time duration for each frame
    }

    scene->AddCue(cue);

    // Add a writer to capture frames
    vtkSmartPointer<vtkPNGWriter> writer =
        vtkSmartPointer<vtkPNGWriter>::New();
    writer->SetFileName("output/frame.png");

    scene->SetRenderer(renderer);
    scene->SetWriter(writer);
    scene->SetInteractor(renderWindowInteractor);

    // Start the animation
    scene->Play();

    // Start the rendering loop
    renderWindow->Render();
    renderWindowInteractor->Start();

    return 0;
}
