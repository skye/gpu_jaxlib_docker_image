# syntax=docker/dockerfile:1

FROM gcr.io/tensorflow-testing/nosla-cuda12.0.1-cudnn8.8-ubuntu20.04-manylinux2014-multipython

WORKDIR /root

RUN git clone https://github.com/google/jax

# install jaxlib ---------------------------------------------------------------

# build jaxlib from source
RUN git clone https://github.com/openxla/xla
WORKDIR jax
RUN python3.10 build/build.py --enable_cuda --bazel_options=--override_repository=xla=$HOME/xla
RUN python3.10 -m pip install dist/*.whl

# release cuda 12 install (works)
# RUN python3.10 -m pip install -U "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# release cuda 11 install
#python3.10 -m pip install -U "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html;

# other install commands
# python3.10 -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# python3.10 -m pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# install jax ------------------------------------------------------------------
RUN pip install -e .

# install test and build deps --------------------------------------------------
RUN "python3.10" -m pip install -U numpy=="1.21.6" scipy=="1.7.3" wheel pytest-xdist absl-py opt-einsum msgpack colorama portpicker matplotlib

# install other useful stuff ---------------------------------------------------
RUN "python3.10" -m pip install -U ipython ipdb
