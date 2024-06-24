
# OE Adapt Demo Tools

TLDR: We need to do vibe checks on the models we’re using. Here’s a document that can help you easily get set up to talk with a trained model via Gradio UI (locally).

## References & Setup

- [VLLM endpoint docs](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) 

- Image for now: 

  - [nathanl/rb\_v16](https://beaker.org/im/01HW5Y0TXPAHCGTMH8MTA14QKH/details) (has latest VLLM from source), but TODO make a new lightweight image for this

  - [nathanl/vllm\_image](https://beaker.org/im/01HW8PSRJ9CHVG8HMV5RESK265/details) (for VLLM + latest OLMo models - transformers v4.40)

- [Exposing port with Beaker](https://beaker-docs.apps.allenai.org/interactive/configuration.html#exposing-ports)  --port 8000

- [Gradio app we use](https://huggingface.co/spaces/ai2-adapt-dev/chat-demo-example) (please contribute improvements!)


## Developing

To develop in this library, first make a new Conda environement:
```
conda create -n chat_tools python=3.10
```
Next, install with editable mode.
```


### TODO

Some important features are needed in the gradio app (preferably via gui rather than argparse)

1. Ability to edit system prompts

2. Ability to edit sampling parameters

## Workflow

### Beaker interactive session

**Creating a session**

    beaker session create --image beaker://nathanl/rb_v16 --gpus 1 --budget ai2/oe-adapt --port 8000
    beaker session create --image beaker://nathanl/vllm_image --gpus 2 --budget ai2/oe-adapt --port 8000 # for newer olmo models

Port mapping outputs (key part on last line, you need this later):

    Defaulting to workspace ai2/nathanl
    Starting session 01HW8DZJFR6QEAFAABCTCR2BY2 with at least 2 GPUs on node 01HQX3BZZ194B0VYF2HAAAM9Z9... (Press Ctrl+C to cancel)
    See more information at https://beaker.org/job/01HW8DZJFR6QEAFAABCTCR2BY2
    Waiting for session to start............. Done!
    Reserved 2 GPUs, 31 CPUs
    Exposed Ports: 0.0.0.0:32800->8000/tcp


### Download model (e.g. beaker Dataset or from HuggingFace)

This isn’t needed for HuggingFace model, just bypass directly to “**Starting Server**”

**Beaker Dataset**

**(**[Dataset docs](https://beaker-docs.apps.allenai.org/concept/datasets.html)) Grab a model from Beaker datasets, or run one on NFS already. Keep track of the URL you create.

    beaker dataset fetch hamishivi/tulu_2_llama_3_8b_dpo -o models/

### Starting Server 

NOTE: If your model doesn’t have the chat template in the tokenizer config, see the below snipper.

    python -m vllm.entrypoints.openai.api_server --model {path or huggingface} --tensor-parallel-size {number of GPUs in session}
    python -m vllm.entrypoints.openai.api_server --model /net/nfs.cirrascale/allennlp/nathanl/models/tulu_2_llama_3_8b_dpo/ --tensor-parallel-size 2
    python -m vllm.entrypoints.openai.api_server --model /net/nfs.cirrascale/aristo/oyvindt/olmo-models/olmo-70b-1T-step160500-hf/ --tensor-parallel-size 2

**No chat template?** If there is not chat template in the tokenizer, you must pass a chat template to vllm with `--chat\_template={template.jinja}`. Some are provided in the [gradio repository](https://huggingface.co/spaces/ai2-adapt-dev/chat-demo-example), which you can clone and pass the local file in when using vllm, as so:

    cd /net/nfs.cirrascale/allennlp/nathanl/chat-demo-example
    python -m vllm.entrypoints.openai.api_server --model /net/nfs.cirrascale/aristo/oyvindt/olmo-models/olmo-70b-1T-step160500-hf/ --tensor-parallel-size 3 --chat-template=templates/none.jinja

### Set up Gradio UI ON A BEAKER MACHINE

_Note: the sharing functionality doesn’t work locally, but the gradio app still works. You need to set up port forwarding (see blow)_

Clone this repo example: <https://huggingface.co/spaces/ai2-adapt-dev/chat-demo-example> 

Requirements: Gradio, openai

```

pip install gradio openai

```

**Run the model!**

    python app.py --port 32800 --model /net/nfs.cirrascale/allennlp/nathanl/models/tulu_2_llama_3_8b_dpo/

It’ll print something like\
Running on local URL:  http\://127.0.0.1:7860

Running on public URL: https\://07f435dc8d326c0254.gradio.live

The second one is what you share with your team 🙂

**Note – these links are public, don’t share them!**

Using multiple models will mean you need to pass --port to vllm (the default vllm port is 8000, so if using multiple demo’s on one machine, run a different port with vllm and the beaker app!)


## Example Dockerfile

    # TODO: Update this when releasing RewardBench publicly
    # This dockerfile is forked from ai2/cuda11.8-cudnn8-dev-ubuntu20.04
    # To get the latest id, run `beaker image pull ai2/cuda11.8-cudnn8-dev-ubuntu20.04`
    # and then `docker image list`, to verify docker image is pulled
    # e.g. `Image is up to date for gcr.io/ai2-beaker-core/public/cncl3kcetc4q9nvqumrg:latest`
    FROM gcr.io/ai2-beaker-core/public/cojd4q5l9jpqudh7p570:latest

    RUN apt update && apt install -y openjdk-8-jre-headless

    RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
    RUN apt-get -y install git-lfs

    WORKDIR /stage/

    RUN pip install --upgrade pip setuptools wheel
    RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    RUN pip install transformers
    RUN pip install flash-attn==2.5.2 --no-build-isolation
    RUN pip install jinja2 
    RUN pip install anthropic
    RUN pip install openai
    RUN pip install git+https://github.com/Isotr0py/vllm.git@olmo

    # for interactive session
    RUN chmod -R 777 /stage/


## Extra Instructions (that didn’t work)

**Set Up Port Forwarding**

To attached to the machine, set up a port forwarding link when authenticated via VPN (**leave this window open)**:

```

ssh -L \<local\_port>:\<remote\_host>:\<remote\_port> {beaker machine}

```

For example:

```

ssh -L 32800:localhost:32800 allennlp-cirrascale-01.reviz.ai2.in

```

**Fixing Sharing (not currently working on local machines)**

Gradio will error the first time you try and share. **Note: sharing links take**

```

Could not create share link. Missing file: /Users/nathanl/miniconda3/envs/misc/lib/python3.10/site-packages/gradio/frpc\_darwin\_arm64\_v0.2. 

Please check your internet connection. This can happen if your antivirus software blocks the download of this file. You can install manually by following these steps: 

1\. Download this file: https\://cdn-media.huggingface.co/frpc-gradio-0.2/frpc\_darwin\_arm64

2\. Rename the downloaded file to: frpc\_darwin\_arm64\_v0.2

3\. Move the file to this location: /Users/nathanl/miniconda3/envs/misc/lib/python3.10/site-packages/gradio

```

Fixed with

```

mv \~/Downloads/frpc\_darwin\_arm64 /Users/nathanl/miniconda3/envs/misc/lib/python3.10/site-packages/gradio

mv /Users/nathanl/miniconda3/envs/misc/lib/python3.10/site-packages/gradio/frpc\_darwin\_arm64 /Users/nathanl/miniconda3/envs/misc/lib/python3.10/site-packages/gradio/frpc\_darwin\_arm64\_v0.2 

```
