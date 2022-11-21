import torch
from transformers import (
    ElectraForMaskedLM,
    ElectraForPreTraining,
)

from transformers.modeling_utils import PreTrainedModel


class ElectraForLanguageModelingModel(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super(ElectraForLanguageModelingModel, self).__init__(config, **kwargs)
        if "generator_config" in kwargs:
            generator_config = kwargs["generator_config"]
        else:
            generator_config = config
        self.generator_model = ElectraForMaskedLM(generator_config)
        if "discriminator_config" in kwargs:
            discriminator_config = kwargs["discriminator_config"]
        else:
            discriminator_config = config
        self.discriminator_model = ElectraForPreTraining(discriminator_config)
        self.vocab_size = generator_config.vocab_size
        if kwargs.get("tie_generator_and_discriminator_embeddings", True):
            self.tie_generator_and_discriminator_embeddings()

    def tie_generator_and_discriminator_embeddings(self):
        self.discriminator_model.set_input_embeddings(self.generator_model.get_input_embeddings())

    def forward(self, inputs, masked_lm_labels, attention_mask=None, token_type_ids=None):
        d_inputs = inputs.clone()

        # run masked LM.
        g_out = self.generator_model(
            inputs, masked_lm_labels=masked_lm_labels, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        # get samples from masked LM.
        sample_probs = torch.softmax(g_out[1], dim=-1, dtype=torch.float32)
        sample_probs = sample_probs.view(-1, self.vocab_size)

        sampled_tokens = torch.multinomial(sample_probs, 1).view(-1)
        sampled_tokens = sampled_tokens.view(d_inputs.shape[0], -1)

        # labels have a -100 value to mask out loss from unchanged tokens.
        mask = masked_lm_labels.ne(-100)

        # replace the masked out tokens of the input with the generator predictions.
        d_inputs[mask] = sampled_tokens[mask]

        # turn mask into new target labels.  1 (True) for corrupted, 0 otherwise.
        # if the prediction was correct, mark it as uncorrupted.
        correct_preds = sampled_tokens == masked_lm_labels
        d_labels = mask.long()
        d_labels[correct_preds] = 0

        # run token classification, predict whether each token was corrupted.
        d_out = self.discriminator_model(
            d_inputs, labels=d_labels, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        g_loss = g_out[0]
        d_loss = d_out[0]
        g_scores = g_out[1]
        d_scores = d_out[1]
        return g_loss, d_loss, g_scores, d_scores, d_labels

