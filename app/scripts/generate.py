import os
import sys

sys.path.append(os.path.abspath(".."))

from validation.Inputs import Inputs


def generate(model, tokenizer, inputs: Inputs, max_length=256, device='cpu'):
    inputs = Inputs.model_validate(inputs)

    prompt = (
        f"Аddress: {inputs.address}\n"
        f"Name: {inputs.name}\n"
        f"Rating: {inputs.rating}\n"
        f"Keywords: {inputs.rubrics}\n"
        f"Review: "
    )

    tokens = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **tokens,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        num_beams=1,
        temperature=0.95,
        top_k=10,
        top_p=0.95,
    )

    # Отбрасываем текст запроса из выхода модели
    response = outputs[0][tokens["input_ids"].shape[-1] :]

    # Почистим gpu память
    tokens.to("cpu")

    return tokenizer.decode(response, skip_special_tokens=True)
