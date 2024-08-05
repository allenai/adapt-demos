# This dockerfile is forked from ai2/pytorch2.3.1-cuda12.1-python3.11
# To get the latest id, run `beaker image pull ai2/pytorch2.3.1-cuda12.1-python3.11` 
# and then `docker image list`, to verify docker image is pulled
# e.g. `Image is up to date for gcr.io/ai2-beaker-core/public/cncl3kcetc4q9nvqumrg:latest`
# FROM gcr.io/ai2-beaker-core/public/cqmg8j0fd26tmjbv4tc0:latest
FROM gcr.io/ai2-beaker-core/public/cqgl31u2ba5vrtuc91og:latest

RUN apt update && apt install -y openjdk-8-jre-headless

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get -y install git-lfs

WORKDIR /stage/

RUN pip install --upgrade pip setuptools wheel
# torch included in image
# RUN pip3 install torch torchvision torchaudio

# pinned to nemotron PR, TODO update to next release
# RUN pip install git+https://github.com/vllm-project/vllm.git@07278c37ddd898d842bbddc382e4f67ac08dae35

# RUN export VLLM_VERSION=0.5.3.post1
# RUN export VLLM_COMMIT=16a1cc9bb2b4bba82d78f329e5a89b44a5523ac8
# RUN pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/${VLLM_COMMIT}/vllm-${VLLM_VERSION}-cp38-abi3-manylinux1_x86_64.whl
RUN pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/16a1cc9bb2b4bba82d78f329e5a89b44a5523ac8/vllm-0.5.3.post1-cp38-abi3-manylinux1_x86_64.whl
RUN pip install vllm-flash-attn

# for interactive session
RUN chmod -R 777 /stage/