from torchtitan.config_manager import JobConfig
from transformers import AutoTokenizer


def build_hf_tokenizer(job_config: JobConfig):
    tokenizer = AutoTokenizer.from_pretrained(job_config.model.tokenizer_path)

    original_encode_func = tokenizer.encode

    def new_encode_func(*args, **kwargs):
        bos = kwargs.pop("bos")
        eos = kwargs.pop("eos")
        t = original_encode_func(*args, **kwargs)
        if bos:
            t.insert(0, tokenizer.bos_token_id)
        if eos:
            t.append(tokenizer.eos_token_id)
        return t

    tokenizer.encode = new_encode_func

    return tokenizer
