import random
import base64
import io
import os
from pathlib import Path
from typing import Union
from modal import Image, Stub, method

stub = Stub("controlnet")

TRANSFORMERS_CACHE = "/cache"


def set_scheduler(model, scheduler: str):
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        DEISMultistepScheduler,
        HeunDiscreteScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        UniPCMultistepScheduler,
    )

    SCHEDULER_MAP = {
        "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(
            config, use_karras=True, algorithm_type="sde-dpmsolver++"
        ),
        "DPM++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(
            config, use_karras=True
        ),
        "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
        "Euler a": lambda config: EulerAncestralDiscreteScheduler.from_config(config),
        "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
        "DDIM": lambda config: DDIMScheduler.from_config(config),
        "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
        "UniPCMultistep": lambda config: UniPCMultistepScheduler.from_config(config),
    }

    SCHEDULER_MAP["DPM++ 2M Karras"] = SCHEDULER_MAP["DPM++ Karras"]

    scheduler_fn = SCHEDULER_MAP.get(scheduler, None)

    if not scheduler_fn:
        raise ValueError(
            f"Sampler {scheduler} not found. Valid options are:"
            f" {list(SCHEDULER_MAP.keys())}"
        )

    model.scheduler = scheduler_fn(model.scheduler.config)
    return scheduler_fn


def download_controlnet():
    import torch
    from diffusers import ControlNetModel

    for model in [
        "monster-labs/control_v1p_sd15_qrcode_monster:v2",
    ]:
        if ":" in model:
            model, subfolder = model.split(":")
        else:
            subfolder = None
        controlnet = ControlNetModel.from_pretrained(
            model,
            subfolder=subfolder,
            torch_dtype=torch.float16,
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
            use_safetensors=True,
            cache_dir=TRANSFORMERS_CACHE,
        )
        controlnet.save_pretrained(TRANSFORMERS_CACHE, safe_serialization=True)


def download_sd():
    import torch
    from diffusers import StableDiffusionPipeline

    for model in [
        "runwayml/stable-diffusion-v1-5",
        "SG161222/Realistic_Vision_V1.4",
        "Lykon/DreamShaper",
    ]:
        print(f"Loading model {model}...")
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
            cache_dir=TRANSFORMERS_CACHE,
        )
        sd_pipe.save_pretrained(TRANSFORMERS_CACHE, safe_serialization=True)


image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate",
        "diffusers[torch]>=0.15.1",
        "ftfy",
        "torchvision",
        "transformers",
        "triton",
        "safetensors",
        "opencv-python-headless",
        "torch>=2.0.1",
    )
    .run_function(download_controlnet)
    .run_function(download_sd)
)
stub.image = image


def load_stable_diffusion(model_id: str, controlnet=None, **kwargs):
    import torch
    from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline
    import os

    if controlnet is not None:
        stable_diffusion = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            controlnet=controlnet,
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
            cache_dir=TRANSFORMERS_CACHE,
            **kwargs,
        )

    else:
        stable_diffusion = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
            cache_dir=TRANSFORMERS_CACHE,
            **kwargs,
        )
    return stable_diffusion


def load_controlnet(model_id: str, **kwargs):
    import torch
    from diffusers import ControlNetModel
    import os

    if ":" in model_id:
        model, subfolder = model_id.split(":")
    else:
        model = model_id
        subfolder = None

    controlnet = ControlNetModel.from_pretrained(
        model,
        subfolder=subfolder,
        torch_dtype=torch.float16,
        use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
        use_safetensors=True,
        cache_dir=TRANSFORMERS_CACHE,
        **kwargs,
    )
    return controlnet


@stub.cls(gpu="a10g")
class StableDiffusion:
    def setup_models(
        self,
        sd_model: str,
        controlnet_model: str = None,
        # lora_model: str = None,
        scheduler: str = None,
    ):
        import torch

        if scheduler is None:
            # speed up diffusion process with faster scheduler and memory optimization
            scheduler = "UniPCMultistep"

        torch.backends.cuda.matmul.allow_tf32 = True

        if controlnet_model is not None:
            controlnet = load_controlnet(controlnet_model)
            stable_diffusion = load_stable_diffusion(sd_model, controlnet=controlnet)
        else:
            stable_diffusion = load_stable_diffusion(sd_model)

        set_scheduler(stable_diffusion, scheduler)

        stable_diffusion.enable_model_cpu_offload()

        self.pipe = stable_diffusion

        return self.pipe

    @method()
    def run_inference(
        self,
        prompt: str,
        control_image: str,
        negative_prompt: str = None,
        seed: Union[int, list[int]] = None,
        steps: int = None,
        batch_size: int = 1,
        # high values: bias toward the literal prompt
        guidance_scale: float = None,
        # high values: bias toward the control image
        controlnet_conditioning_scale: float = None,
        scheduler: str = None,
        width: int = None,
        height: int = None,
        sd_model: str = None,
        controlnet_model: str = None,
    ) -> list[bytes]:
        import torch
        import PIL.Image

        if guidance_scale is None:
            guidance_scale = 7.5
        if controlnet_conditioning_scale is None:
            controlnet_conditioning_scale = 1.3
        if steps is None:
            steps = 30

        decoded_image = base64.b64decode(control_image)
        image = PIL.Image.open(io.BytesIO(decoded_image))

        self.setup_models(
            sd_model=sd_model,
            controlnet_model=controlnet_model,
            scheduler=scheduler,
        )

        # set seed
        if seed is None:
            seeds = [
                torch.manual_seed(random.getrandbits(64)) for _ in range(batch_size)
            ]
        elif not isinstance(seed, list):
            seeds = [torch.manual_seed(seed) for _ in range(batch_size)]
        else:
            if len(seed) != batch_size:
                raise ValueError(
                    f"Number of seeds ({len(seed)}) does not match batch size"
                    f" ({batch_size})"
                )
            seeds = [torch.manual_seed(s) for s in seed]
        print(f"Using seeds: {[s.initial_seed() for s in seeds]}")

        # generate image
        output = self.pipe(
            [prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size if negative_prompt else None,
            num_inference_steps=steps,
            image=image,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=seeds,
            width=width,
            height=height,
        )

        # Convert to PNG bytes
        results = []
        for image in output.images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                results.append(buf.getvalue())
        return results


@stub.local_entrypoint()
def entrypoint(
    prompt: str,
    control_image_path: str,
    output_path: str = None,
    negative_prompt: str = None,
    seed: int = None,
    steps: int = 30,
    n: int = 1,
    sd_model: str = "SG161222/Realistic_Vision_V1.4",
    controlnet_model: str = "monster-labs/control_v1p_sd15_qrcode_monster:v2",
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.45,
):
    # load bytes from the control image path
    with open(control_image_path, "rb") as f:
        control_image_bytes = f.read()
    control_image_b64 = base64.b64encode(control_image_bytes).decode()

    if output_path is None:
        output_path = "/tmp/aimages"
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(exist_ok=True, parents=True)

    prompt = ", ".join(
        [prompt, "stunning quality", "highly detailed", "8k", "cinematic"]
    )
    negative_prompt = ", ".join(
        [
            negative_prompt or "",
            "ugly",
            "bad hands",
            "bad",
            "low quality",
            "cartoon",
            "clipart",
            "boring",
            "dull",
            "uninteresting",
        ]
    )

    sd = StableDiffusion()
    images = sd.run_inference.call(
        prompt=prompt,
        negative_prompt=negative_prompt,
        batch_size=n,
        steps=steps,
        seed=seed,
        control_image=control_image_b64,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        sd_model=sd_model,
        controlnet_model=controlnet_model,
    )

    output_index = len(list(output_path.glob("*")))
    for i, image_bytes in enumerate(images):
        image_output_path = output_path / f"aimage_{output_index + 1}_{i + 1}.png"
        with open(image_output_path, "wb") as f:
            f.write(image_bytes)
        print(f"Image written to {image_output_path}")

        # load the file in the default image viewer
        try:
            os.system(f"open {image_output_path}")
        except Exception:
            pass
