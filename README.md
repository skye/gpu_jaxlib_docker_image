# jaxlib GPU docker image

This is my attempt at creating a repeatable setup for building jaxlib from
source and interactively running tests on a GPU VM.

I don't really know what I'm doing, especially when it comes to docker.

## Build image

```bash
sudo docker build -t gpu_jaxlib
```

Building GPU jaxlib takes a long time, so do this on a beefy machine (doesn't
have to have GPUs attached).

## Save image to tar file

```bash
sudo docker save -o <tar file> gpu_jaxlib:latest
```

It needs lots of disk space (~25 GB).

## Load image from tar file

```bash
sudo docker load -i <tar file>
```

## One-time setup on GPU VM to run image

I'm not sure how many of these are necessary.

```bash
sudo apt install nvidia-docker2
sudo apt install nvidia-container-toolkit

sudo nvidia-ctk runtime configure

sudo systemctl restart docker
```

## Run image

```bash
sudo docker run -it --rm --shm-size=16g --gpus all  gpu_jaxlib:latest
```

The image starts in a jax checkout that's locally installed via `pip install -e
.`, with a jaxlib built from source from ~/xla and installed.

You can run all unit tests with (may need tweaking):

```bash
export TF_CPP_MIN_LOG_LEVEL=0
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export LD_LIBRARY_PATH="/usr/local/cuda:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/tensorrt/lib"
export NCCL_DEBUG=WARN

python3.10 -m pytest -n 8 --tb=short --maxfail=1 tests examples --deselect=tests/linalg_test.py::LaxLinalgTest::test_tridiagonal_solve1 --deselect=tests/multi_device_test.py::MultiDeviceTest::test_computation_follows_data --deselect=tests/xmap_test.py::XMapTest::testCollectivePermute2D
```
