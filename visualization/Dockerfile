FROM ubuntu:22.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl ninja-build pkg-config \
        libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
        libgl1-mesa-dev libgl1-mesa-dri libhdf5-dev libx11-dev libxt-dev \
        python3 xauth xvfb \
    && rm -rf /var/lib/apt/lists/*

ARG VTK_VERSION=9.3.1
ARG BUILD_JOBS=2
RUN curl -fsSL "https://github.com/Kitware/VTK/archive/refs/tags/v${VTK_VERSION}.tar.gz" -o /tmp/vtk.tar.gz \
    && mkdir /tmp/vtk \
    && tar -xzf /tmp/vtk.tar.gz -C /tmp/vtk --strip-components=1 \
    && cmake -G Ninja -S /tmp/vtk -B /tmp/vtk-build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/vtk \
        -DVTK_BUILD_TESTING=OFF -DVTK_BUILD_EXAMPLES=OFF \
        -DVTK_BUILD_ALL_MODULES=OFF -DVTK_USE_X=ON \
        -DVTK_GROUP_ENABLE_Rendering=DONT_WANT \
        -DVTK_GROUP_ENABLE_StandAlone=DONT_WANT \
        -DVTK_MODULE_ENABLE_VTK_FiltersSources=YES \
        -DVTK_MODULE_ENABLE_VTK_IOImage=YES \
        -DVTK_MODULE_ENABLE_VTK_IOFFMPEG=YES \
        -DVTK_MODULE_ENABLE_VTK_RenderingAnnotation=YES \
        -DVTK_MODULE_ENABLE_VTK_RenderingContextOpenGL2=YES \
        -DVTK_MODULE_ENABLE_VTK_RenderingFreeType=YES \
        -DVTK_MODULE_ENABLE_VTK_RenderingGL2PSOpenGL2=YES \
        -DVTK_MODULE_ENABLE_VTK_RenderingVolumeOpenGL2=YES \
    && cmake --build /tmp/vtk-build --parallel "${BUILD_JOBS}" \
    && cmake --install /tmp/vtk-build \
    && rm -rf /tmp/vtk /tmp/vtk-build /tmp/vtk.tar.gz

WORKDIR /opt/imhd-cuda
COPY visualization/src/ visualization/src/
COPY visualization/build/launcher_makemovie.py visualization/build/makemovie.inp visualization/build/
COPY build/on-device/input.inp build/on-device/

RUN cmake -G Ninja -S visualization/src -B visualization/build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/vtk \
    && cmake --build visualization/build --parallel "${BUILD_JOBS}" \
    && mkdir /output

ENV LIBGL_ALWAYS_SOFTWARE=1 IMHD_MOVIE_OUTPUT_DIR=/output
WORKDIR /opt/imhd-cuda/visualization/build
ENTRYPOINT ["xvfb-run", "-a", "python3", "launcher_makemovie.py"]
CMD ["/data/", "rhovz"]