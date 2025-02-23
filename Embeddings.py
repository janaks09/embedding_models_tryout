import torch
import torch.nn.functional as F
import voyageai

from openai import OpenAI
from angle_emb import AnglE
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

class ModelInfo:
    def __init__(self, provider, dimension):
        self.provider = provider
        self.dimension = dimension


class Embeddings:

    def get_embeddingModelInfo(self, model):
        
        if not model:
            raise Exception("Model name can not be empty.")

        match model:
            case "openai":
                return ModelInfo(model, 1536)
            case "intfloat":
                return ModelInfo(model, 4096)
            case "uae":
                return ModelInfo(model, 1024)
            case "gte":
                return ModelInfo(model, 768)
            case _:
                raise Exception("Invalid provider selection")
        

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]
            
    def average_pool(self, last_hidden_states: Tensor,
        attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_openai_embeddings(self, text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        openAiClient = OpenAI()
        return openAiClient.embeddings.create(input=text, model=model).data[0].embedding

    def get_UAE_embeddings(self, text, model="WhereIsAI/UAE-Large-V1"):
        text = text.replace("\n", " ")
        angle = AnglE.from_pretrained(model, pooling_strategy="cls").cuda()
        vec = angle.encode(text, to_numpy=True)
        return vec[0]
    
    def get_GTE_embeddings(self, text, model="thenlper/gte-base"):
        tokenizer = AutoTokenizer.from_pretrained(model)
        m = AutoModel.from_pretrained(model)

        # Tokenize the input texts
        batch_dict = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')

        outputs = m(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy()[0]

    def get_intfloat_embeddings(self, text, model="intfloat/e5-mistral-7b-instruct"):
        texts = [text.replace("\n", " ")]

        # Initialize the tokenizer and model with the pre-trained identifiers
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModel.from_pretrained(model)

        # Tokenize input texts and adjust for model input requirements
        max_length = 4096
        batch_dict = tokenizer(
            texts,
            max_length=max_length - 1,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [
            input_ids + [tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]
        ]
        batch_dict = tokenizer.pad(
            batch_dict, padding=True, return_attention_mask=True, return_tensors="pt"
        )

        # Obtain model outputs
        outputs = model(**batch_dict)
        embeddings = self.last_token_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )

        # Normalize embeddings
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        return normalized_embeddings

    def get_embeddings(self, text, provider):

        if not provider:
            raise Exception("Provider can not be empty.")

        match provider:
            case "openai":
                return self.get_openai_embeddings(text)
            case "intfloat":
                return self.get_intfloat_embeddings(text)
            case "uae":
                return self.get_UAE_embeddings(text)
            case "gte":
                return self.get_GTE_embeddings(text)
            case _:
                raise Exception("Invalid provider selection")
