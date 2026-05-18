import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        result = []

        for i in range(new_chars):

            # 1. Crop context to context_length if it exceeds it: context[:, -context_length:]
            if context.shape[1] > context_length:
                context = context[:, -context_length:]
            # 2. Run model(context) -> take last position's logits -> apply softmax(dim=-1)
            logits = model(context)
            probs = nn.functional.softmax(logits[:,-1,:], dim=-1)
            # 3. Sample next token with torch.multinomial(probs, 1, generator=generator)
            next_token = torch.multinomial(probs, 1, generator=generator)
            generator.set_state(initial_state)
            # 4. Append sampled token to context with torch.cat
            context = torch.cat((context, next_token), dim=-1)
            # 5. Map token to character using int_to_char and accumulate result
            token_map = int_to_char[next_token.item()]
            result.append(token_map)

        # Once your code passes the test, check out the Colab link to see your code generate new Drake lyrics!
        return ''.join(result)