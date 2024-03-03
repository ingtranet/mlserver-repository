import json

from typing import Dict, Any
from mlserver import MLModel, types
from mlserver.codecs import StringCodec


class Orion14BChatInt4(MLModel):
    MODEL_NAME = "OrionStarAI/Orion-14B-Chat-Int4"
    BASE_GENERATION_CONFIG = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_new_tokens": 1024,
        "pad_token_id": 0,
        "repetition_penalty": 1.05
    }

    async def load(self) -> bool:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation.utils import GenerationConfig
        from transformers.utils.logging import disable_progress_bar, set_verbosity_debug
        disable_progress_bar(); set_verbosity_debug()
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(self.MODEL_NAME)
        return await super().load()

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        from transformers.generation.utils import GenerationConfig
        messages = self._parse_request(payload)["messages"]
        generation_config = self._parse_request(payload).get("generation_config", "default")
        if generation_config == "nosample":
            gcon = GenerationConfig(do_sample=False, **self.BASE_GENERATION_CONFIG)
        elif generation_config == "beam":
            gcon = GenerationConfig(
                temperature=0.3,
                top_k=5,
                top_p=0.9,
                num_beams=3,
                **self.BASE_GENERATION_CONFIG
            )
        elif generation_config == "exact":
            gcon = GenerationConfig(
                temperature=0.1,
                top_k=3,
                top_p=0.5,
                **self.BASE_GENERATION_CONFIG
            )
        elif generation_config == "creative":
            gcon = GenerationConfig(
                temperature=0.9,
                top_k=7,
                top_p=0.95,
                **self.BASE_GENERATION_CONFIG
            )
        else:
            gcon = self.model.generation_config
        response = {
            "role": "assistant",
            "content": self.model.chat(self.tokenizer, messages, streaming=False, generation_config=gcon)
        }
        response_bytes = json.dumps(response, ensure_ascii=False).encode("UTF-8")
        return types.InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                types.ResponseOutput(
                    name="generated_text",
                    shape=[len(response_bytes)],
                    datatype="BYTES",
                    data=[response_bytes],
                    parameters=types.Parameters(content_type="str"),
                )
            ],
        )

    def _parse_request(self, payload: types.InferenceRequest) -> Dict[str, Any]:
        inputs = {}
        for inp in payload.inputs:
            inputs[inp.name] = json.loads(
                "".join(self.decode(inp, default_codec=StringCodec))
            )
        return inputs