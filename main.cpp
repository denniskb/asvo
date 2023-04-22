#include <vector_functions.h>
#include <vector_types.h>

#include <memory>
#include <string>

#include "Glue.h"
#include "Light.h"
#include "Object3D.h"
#include "Renderer.h"
#include "extended_helper_math.h"

int main(int argc, char **argv) {
  // TODO @user: Replace hard-coded values
  int const frameWidthInPixels = 550;
  int const frameHeightInPixels = 800;
  double const aspectRatio = ((double)frameWidthInPixels) / frameHeightInPixels;
  bool const shadowMapping = true;

  // Set up the renderer
  auto renderer =
      new Renderer(frameWidthInPixels, frameHeightInPixels, shadowMapping);

  // Load an asvo from file.
  std::string path(argv[0]);
  int lastSlash = path.find_last_of("\\");
  path.resize(lastSlash + 1);
  path.append("../content/");

  auto model = new Object3D(
      new BFSOctree((path + "imrod.asvo").c_str(),
                    (path + "diffuse.raw").c_str(),
                    (path + "illum.raw").c_str(), (path + "spec.raw").c_str(),
                    (path + "normal.raw").c_str()),
      true);
  float3 rotAxis = make_float3(1.f, 0.f, 0.f);
  model->assignTransform(make_rotation(rotAxis, -1.5707f));

  // Set up the light.
  float3 lightPosition = make_float3(-1.f, -0.5f, 0.5f);
  float lightDiffusePower = 0.8;
  Light light(lightPosition, lightDiffusePower);

  // Set up the camera.
  float3 position = make_float3(0.f, 25.f, -100.f);
  float3 lookAt = make_float3(0.f, 0.f, 0.f);
  float fov = 1;
  Camera camera(position, lookAt, fov, aspectRatio, 10, 200);

  // Initialize the GLUT framework.
  if (!Glue::init(argc, argv, frameWidthInPixels, frameHeightInPixels, renderer,
                  model, light, camera)) {
    return 1;
  }

  // Start the main render-and-update loop
  Glue::startGlutMainLoop();

  Glue::cleanUp();

  return 0;
}