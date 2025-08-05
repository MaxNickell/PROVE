from PIL import Image

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

class DeepSeekVL2:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-vl2-tiny") -> None:
        self.vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_name)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()
    

    def list_and_bound_objects(self, image_path: str) -> str:
        conversation = [
            {
                "role": "<|User|>",
                "content": "This is the image: <image>\n"
                        "<|grounding|>Identify all objects in the image and output them in bounding boxes.",
                "images": [
                    image_path,
                ],
            },
            {
                "role": "<|Assistant|>", 
                "content": ""
            },
        ]
        
        return self.run_inference(conversation)   
    
    def run_inference(self, conversation) -> str:
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(self.vl_gpt.device)

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer
        