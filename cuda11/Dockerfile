# syntax=docker/dockerfile:1

FROM gcr.io/tensorflow-testing/nosla-cuda11.8-cudnn8.6-ubuntu20.04-manylinux2014-multipython

WORKDIR /root

RUN git clone https://github.com/google/jax

# install jaxlib ---------------------------------------------------------------

# build jaxlib from source
RUN git clone https://github.com/openxla/xla
WORKDIR jax

# RUN bazel build --verbose_failures=true --override_repository=xla=/root/xla --config=avx_posix --config=mkl_open_source_only --config=rbe_linux_cuda11.8_nvcc_py3.9 --config=tensorflow_testing_rbe_linux -- //build:build_wheel

RUN python3.9 -m pip install build
RUN python3.9 build/build.py --enable_cuda --bazel_options=--override_repository=xla=$HOME/xla
RUN python3.9 -m pip install dist/*.whl

# install jax ------------------------------------------------------------------
RUN pip install -e .

# install test and build deps --------------------------------------------------
RUN "python3.9" -m pip install -U numpy=="1.21.6" scipy=="1.7.3" wheel pytest-xdist absl-py opt-einsum msgpack colorama portpicker matplotlib

# install other useful stuff ---------------------------------------------------
RUN "python3.9" -m pip install -U ipython ipdb && sudo apt update && sudo apt install tmux -y
RUN echo 'set -g history-limit 999999999' >> ~/.tmux.conf
