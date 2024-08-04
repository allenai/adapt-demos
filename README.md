
# AI2 Adapt Demo Tools

An important part of modern language model (LM) research is making tools that are easy to use and effective.
In this repo, we have our lightweight Gradio tools for completing many needed tasks for vibe-checking your model.

The core features we implemented included:
* Default Gradio ChatInterface behavior plugged into a VLLM OpenAI endpoint.
* Inclusion of safety-filter models, including classifying prompt or generation, re-writing, and other behavior.
* Saving of conversations with the user.
* Side-by-side comparison of multiple models with custom system prompts.

Here is an example:

![Screenshot 2024-06-27 at 3 48 51 PM](https://github.com/allenai/reward-bench/assets/10695622/2781011a-6b5e-4faf-8cc0-e972a2e8457f)

With additional UX elements not shown for the safety filter. 
When the safety filter is enabled, the expanded window will show:
```
Harmful Request: {Yes/No} # if the prompt is harmful
Response Refusal: {Yes/No} # if the model refused a harmful prompt
Harmful Response: {Yes/No} # if the response was harmful
```
And optionally, it will rewrite the prompt to the model telling it to refuse the request.
See [the prompts](https://github.com/allenai/adapt-demos/blob/main/demo_tools/prompts.py) in the repo.

The basic system diagram is shown below.
![Frame 25](https://github.com/allenai/reward-bench/assets/10695622/9fae34a6-af49-47c2-be98-9622c78de559)

---

TLDR: Here’s a document that can help you easily get set up to talk with a trained model via Gradio UI (locally).

## Repository Structure

This repository is largely an expansion of [Gradio's ChatInterface](https://www.gradio.app/docs/gradio/chatinterface) abstraction.
We also add OpenAI endpoint compatible tools for debugging chat demos.

```
├── README.md                   <- Top-level guidance to use this project
├── demo_tools/                
│   ├── templates/              <- Jinja chat templates for vllm
│   ├── dummy_chatbot.py        <- OpenAI compatible debugging model
│   ├── interface.py            <- Enhanced Gradio Code
│   ├── model_client.py         <- Predict functions & model wrappers for Gradio
│   ├── prompts.py              <- Prompts 
│   └── style.py                <- Styling code
├── LICENSE                     
├── Makefile                    <- Style tools
├── app-dual.py                 <- Side-by-side demo
├── app.py                      <- Base demo
└── pyproject.toml              <- Requirements and setup
```

## References & Setup

* [VLLM endpoint docs](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) 

* Image for now: 

  * [nathanl/rb\_v20](https://beaker.org/im/01J15RB4DZC5QJJ4R51GMHMD00/details) (has latest VLLM from source), but TODO make a new lightweight image for this

  * TODO Recreate [nathanl/vllm\_image](https://beaker.org/im/01HW8PSRJ9CHVG8HMV5RESK265/details) (for VLLM + latest OLMo models - transformers v4.40)

* [Exposing port with Beaker](https://beaker-docs.apps.allenai.org/interactive/configuration.html#exposing-ports)  --port 8000

* [WildGuard Repo](https://github.com/allenai/wildguard) More information on the safety models

    * [WildGaurd Model](https://huggingface.co/allenai/wildguard) based on Mistral 7B

    * More models coming soon

## Developing

To develop in this library, first make a new Conda environement:
```
conda create -n chat_tools python=3.10
```
Next, install with editable mode (or not).
```
pip install -e .
```
Requirements are in [`pyproject.toml`](https://github.com/allenai/adapt-demos/blob/main/pyproject.toml).

### Additions?

_To add or request new features, please [open an issue](https://github.com/allenai/adapt-demos/issues/new/choose)._

### Docker

For base usage, consider using [VLLM's images](https://hub.docker.com/r/vllm/vllm-openai/tags), but they are not built pre-release for cutting edge models.
They can be loaded in Beaker with:
```
beaker sesion ... --image=docker://image/name
```

To update the image, run these commands in the root directory of this repo:
```
docker build -t <local_image_name> . --platform linux/amd64
beaker image create <local_image_name> -n <beaker_image_name>
```

### Debug mode
We have added a debug mode for working on the interface without using GPUs. 
With `app.py` or `app-dual.py` pass the arg `--debug` as follows:
```
python app.py --debug --port 1234 --model tmp --safety_filter_port 1234 --safety_model luca
python app-dual.py --model_one tmp --model_two TMEPPP --port_one 1234 --port_two 2333 --debug
```


## Workflow

Much of this workflow is specific to AI2 tools, but largely the system can be extended arbitrarily. 

### Beaker interactive session

**Creating a session**

With the right image and number of gpus.
```
beaker session create --image beaker://nathanl/rb_v16 --gpus 1 --budget ai2/oe-adapt --port 8000
```

Port mapping outputs (key part on last line, you need this later):

```
Defaulting to workspace ai2/nathanl
Starting session 01HW8DZJFR6QEAFAABCTCR2BY2 with at least 2 GPUs on node 01HQX3BZZ194B0VYF2HAAAM9Z9... (Press Ctrl+C to cancel)
See more information at https://beaker.org/job/01HW8DZJFR6QEAFAABCTCR2BY2
Waiting for session to start............. Done!
Reserved 2 GPUs, 31 CPUs
Exposed Ports: 0.0.0.0:32800->8000/tcp
```
Port 32800 will be passed to the Gradio demo.

**For Another Model (e.g. safety / side-by-side)**:  You need to create another beaker session, run either of these commands in the same machine depending on the size of your model (note the port needs to be different if on the same machine, but cross machine tunneling of ports via SSH can also be used):

```
beaker session create --image beaker://nathanl/rb_v16 --gpus 1 --budget ai2/oe-adapt --port 8001
```

Make sure to take note of the port mapping output similar to above.

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
    python -m vllm.entrypoints.openai.api_server --model /net/nfs.cirrascale/mosaic/liweij/auto_jailbreak/src/EasyLM/models/v3/tulu2mix-all-vani_b-50000-vani_h-50000-adv_b-50000-adv_h-50000 --tensor-parallel-size 3 --chat-template=templates/none.jinja

**For safety models** You need to run an API server (e.g., adding a wildguard model), MODIFY

    python -m vllm.entrypoints.openai.api_server --model allenai/wildguard --tensor-parallel-size 2 --port 8001

This one does not need a chat template as it is handled elsewhere in the code (for now).

### Set up Gradio UI ON A BEAKER MACHINE

_Note: the sharing functionality doesn’t work locally, but the gradio app still works. You need to set up port forwarding (see blow)_

Clone this repo example: <https://huggingface.co/spaces/ai2-adapt-dev/chat-demo-example> 

Requirements: Gradio, openai

```

pip install gradio openai

```

**Run the model!**

    python app.py --port 32800 --model /net/nfs.cirrascale/allennlp/nathanl/models/tulu_2_llama_3_8b_dpo/

**For Safety Models** run the following command: if you run on the same machine on which you created the beaker session, no need to provide a port:

    python app.py --model /net/nfs.cirrascale/mosaic/liweij/auto_jailbreak/src/EasyLM/models/v3/tulu2mix-all-vani_b-50000-vani_h-50000-adv_b-50000-adv_h-50000 --safety_model allenai/wildguard 

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


## Extra Notes

### Debugging

**Unexpected generator object**: Gradio's `ChatInterface` [inspected the generation function](https://github.com/allenai/adapt-demos/blob/c3e3e97b0c6e4a9e15cf46982276f95343bec9a7/demo_tools/interface.py#L159) to see if it is designed for streaming or not.
Some of our code is parametrized to either stream or return the whole thing, which will make the object from the predict function appear as a generator sometimes.
This is nasty to debug, good luck.

**HuggingFace Token in session for VLLM**: If you sign in with `huggingface-cli login` in one session on a node, you should be able to just do this once. Alternatively, you can do the following.
```
mkdir -p "/root/.cache/huggingface" && echo -n "${HUGGING_FACE_HUB_TOKEN}" > "/root/.cache/huggingface/token"
```

### Useful Commands / Ideas

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

Gradio will error the first time you try and share. 

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
