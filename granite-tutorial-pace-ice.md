# Granite Tutorial Supplement (BETA)

This page covers added resources to run and work with the open source [IBM Granite Workshop material](https://ibm.github.io/granite-workshop) on Georgia Tech's [PACE ICE cluster](https://pace.gatech.edu/ice-cluster/). Please consider this a replacement for the Granite Workshop's Lab 0 prelab.

## Related Videos
See these videos for an overview of the tutorial as well as a demo of how to run it on PACE ICE using Ollama:
- [IBM Granite Workshop Overview - by BJ Hargrave - June 4th](https://mediaspace.gatech.edu/media/Granite+Workshop+for+GT+OSPO+VSIP+-+June+4th%2C+2025/1_uiym8a8f)
- [GT Office Hours for this PACE ICE training - June 5th](https://mediaspace.gatech.edu/media/OSPO+VSIP+-+Running+Granite+on+PACE+ICE/1_w5fsoz1e)

___
> [!IMPORTANT]
> You will need access to the GA Tech VPN - this can be installed locally, or accessed entirely via your web browser here: [vpn.gatech.edu](https://vpn.gatech.edu/)

## Running on PACE-ICE
Students with the VSIP program have access to the ICE cluster, which has CPU and GPU backends as well as an Ollama instance supported through Open OnDemand. This Ollama instance can be used in place of running locally on your laptop or Replicate, which uses a GPU backend.

[DOCS: Using Ollama + Jupyter in Open OnDemand](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0044442) (Requires GT login)

> [!NOTE] 
> PACE supports Ollama endpoints via Slurm scheduled jobs rather than as a locally (your PC) or remotely hosted (Replicate) service. You can use the Open OnDemand (OOD) application on PACE ICE to schedule a Slurm job that then has access to a set of shared models, including the latest Granite 3 models.

## Part 1: Setting Up Your PACE Virtual Environment
First, [**Login here by opening this link in a new tab**](https://ondemand-ice.pace.gatech.edu/) (**Georgia Tech VPN Required**) - or right click and copy the link for use on the [VPN's web interface here.](https://vpn.gatech.edu/)

Now, we'll need to set up a "clean" venv (or virtual environment) via the terminal to run the workshop notebooks. To do this, we'll need to drop into a login shell at the ICE cluster. Once you log in to the OnDemand instance, you can select the following: `Clusters -> ICE Shell Access`.

![SS1](https://github.com/user-attachments/assets/e27536d6-963a-4c49-83d1-8f8269f2556b)

A new terminal window should open, you can now run the following commands to setup your venv:

> [!WARNING]
> Below is a shell prompt - these lines show both the prompt and the command. Only copy the command portion (after the $), not the entire line!

- Step 1. Load our (stable) Python version
```
[login-ice ~]$ module load python/3.12
```

- Step 2. Declare the virtual environment
```
[login-ice ~]$ python -m venv --upgrade-deps --clear granite_env
```
> [!NOTE]
> We've named our venv "granite_venv" here but you can name it whatever you like - just adjust accordingly!

- Step 3. Source the newly declared environment to use it on your terminal
```
[login-ice ~]$ source ~/granite_env/bin/activate
```
> [!NOTE]
> Make sure this venv has jupyter installed to use with the PACE OOD environment.

- Step 4. Install the required python-pip libraries
```
(granite_env) [login-ice ~]$ python -m pip install --require-virtualenv notebook jupyter ipywidgets
```

- Step 5. Clone The IBM Granite Workshop Repo
```
(granite_env) [login-ice ~]$ git clone https://github.com/IBM/granite-workshop.git
```

## Part 2: Starting an Open OnDemand instance
The easiest way to use Ollama with Jupyter notebooks is to use the [(DOCS) Open OnDemand instance for ICE](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042133) (Requires GT login for docs and OOD interface). Once you log in to the OnDemand instance, you can select the following: `Interactive Applications -> Ollama + Jupyter (Beta)`

<img width="756" alt="SS1" src="https://github.com/user-attachments/assets/9eaeabae-6f94-4a9f-9ad8-1ba1931f16be" />

Now you need to create an instance, for your convience - we recommend running with the following parameters:
- Python environment: Custom virtual environment
    - Virtual environment path: Wherever you created your virtual environment (see above).  In our example, we used `/home/hice1/<gburdell>/granite_venv`
- Ollama models directory: `PACE Shared Models`
- Jupyter interface: Whichever you prefer
- Node type: <Choose 1 of "NVIDIA L40s", "NVIDIA GPU H100 (HGX)", or "NVIDIA GPU (first avail)">
- Number of CPUs (cores): `4`
- Number of GPUs: `1`
- Total memory: <32 GB or 64 GB>
- Time: <start with 1 hour and expand as needed>
- Quality of service: `Default (none)`

After some time, you'll see something like following populate under `My Interactive Sessions` - this is your PACE-ICE node.

<img width="756" alt="SS3" src="https://github.com/user-attachments/assets/191e5063-ac41-481c-a6ce-5f484eeb3c0c" />

Now when you start your Jupyter session, you'll then use the Jupyter file browser to navigate to `granite-workshop/notebooks`.

## Part 3: ICE Specific Changes
Through testing, we've discovered a change that must be made in regards to how the notebooks starts an Ollama session. Our approach on ICE is actually simpler than the original notebook.

In the original [Summarize example notebook](https://ibm.github.io/granite-workshop/lab-1/) from Lab 1, the session is started in this cell. 
 
```Python
import os
import requests
from langchain_ollama.llms import OllamaLLM
from langchain_community.llms import Replicate
from ibm_granite_community.notebook_utils import get_env_var

model_path = "ibm-granite/granite-3.3-8b-instruct"
try: # Look for a locally accessible Ollama server for the model
    response = requests.get(os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    model = OllamaLLM(
        model="granite3.3:8b",
        num_ctx=65536, # 64K context window
    )
    model = model.bind(raw=True) # Client side controls prompt
except Exception: # Use Replicate for the model
    model = Replicate(
        model=model_path,
        replicate_api_token=get_env_var('REPLICATE_API_TOKEN'),
        model_kwargs={
            "max_tokens": 2000, # Set the maximum number of tokens to generate as output.
            "min_tokens": 200, # Set the minimum number of tokens to generate as output.
            "temperature": 0.75,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
    )
```

**However,** you must replace this cell with the code below in order to proceed with the rest of the notebook.

```Python
import os
from langchain_ollama.llms import OllamaLLM
from ibm_granite_community.notebook_utils import get_env_var

model_path = "ibm-granite/granite-3.3-8b-instruct"
model = OllamaLLM(
    model="granite3.3:8b",
    num_ctx=65536, # 64K context window
)
model = model.bind(raw=True) # Client side controls prompt
```

> [!TIP] 
> Once you've made the change, you can print out all the available models for your Ollama instance by running the following in a new cell:
> ```
> import ollama
> for m in ollama.list().models:
>     print(m.model)
> ```

#### You're all set! Now, you can rock and roll with Ollama+Granite via your browser 🚀🤘

### Contributors
Many thanks to [Alex Jenkins](https://github.com/rockenman1234) and [Ron Rahaman](https://github.com/RonRahaman) for their contributions to earlier versions of this material!
