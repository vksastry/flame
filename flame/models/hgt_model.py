import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
import pdb
class HGTModelWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, C: int, H: int, W: int):
        super().__init__()
        self.model = base_model

        self.C = C
        self.H = H
        self.W = W
        self.hidden_size = self.model.config.hidden_size
        
        # simple projection: (C*H*W) -> d_model
        self.proj = nn.Linear(C * H * W, self.hidden_size)
        self.forecast_head = nn.Linear(self.hidden_size, C * H * W)
        self.criterion = nn.MSELoss()
    # --------- Delegations so the training harness sees a HF-like model ---------
    @property
    def config(self):
        return self.model.config

    def post_init(self):
        # flame/train_hgt.py calls model.post_init()
        if hasattr(self.model, "post_init"):
            return self.model.post_init()
    """
    @property
    def layers(self):
        if hasattr(self.model, "layers"):
            return self.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        return None

    @property
    def tok_embeddings(self):
        # Not all models use this name; we try common patterns.
        if hasattr(self.model, "tok_embeddings"):
            return self.model.tok_embeddings
        if hasattr(self.model, "embeddings"):
            return self.model.embeddings
        if hasattr(self.model, "model") and hasattr(self.model.model, "embeddings"):
            return self.model.model.embeddings
        return None

    @property
    def norm(self):
        if hasattr(self.model, "norm"):
            return self.model.norm
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        return None


    @property
    def lm_head(self):
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        return None
    """ 
    def forward(
    self,
    inputs=None,        # HGT data: (B, T, C, H, W)
    input_ids=None,     # keep HF-style API
    inputs_embeds=None,
    labels=None,
    **kwargs,           # position_ids, cu_seqlens, etc.
    ):
        # If we got HGT inputs, project them to embeddings
        if inputs is not None:
            # inputs: (B, T, C, H, W)
            B, T, C, H, W = inputs.shape
            assert C == self.C and H == self.H and W == self.W, (
            f"Expected (C,H,W)=({self.C},{self.H},{self.W}), "
            f"got ({C},{H},{W})"
            )

            x = inputs.view(B, T, C * H * W)   # (B, T, C*H*W)
            inputs_embeds = self.proj(x)       # (B, T, hidden_size)
            input_ids = None                   # make sure base model doesn't use embedding()
            """
            print("inputs stats:",
                  inputs.mean().item(),
                  inputs.std().item(),
                  inputs.min().item(),
                  inputs.max().item(),
                  flush=True)
            print("x stats:",
                  x.mean().item(),
                  x.std().item(),
                  x.min().item(),
                  x.max().item(),
                  flush=True)
            print("inputs_embeds stats:",
                  inputs_embeds.mean().item(),
                  inputs_embeds.std().item(),
                  inputs_embeds.min().item(),
                  inputs_embeds.max().item(),
                  flush=True)
            """
        # Now call the base HF/FLA model with whatever we have
        output = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,          # forward position_ids, cu_seqlens, etc.
        )
        hidden = output.hidden_states[-1]
        logits = None
        loss = None

        # 3. If labels provided (HGT targets), compute regression forecast loss
        if labels is not None:
            B, T_out, C, H, W = labels.shape
            assert C == self.C and H == self.H and W == self.W
            #print("hidden:", hidden.shape, "labels:", labels.shape, flush=True)
            # Take last T_out timesteps from hidden as forecast "slots"
            hidden_last = hidden[:, -T_out:, :]                  # (B, T_out, hidden_size)
            
            pred_flat = self.forecast_head(hidden_last)          # (B, T_out, C*H*W)
            pred = pred_flat.view(B, T_out, C, H, W)             # (B, T_out, C, H, W)

            logits = pred                                        # treat logits as forecast fields
            loss = self.criterion(pred, labels)
        """
        with torch.no_grad():
            print("labels stats:",
                  labels.mean().item(),
                  labels.std().item(),
                  labels.min().item(),
                  labels.max().item(),
                  )
            print("pred stats:",
                  pred.mean().item(),
                  pred.std().item(),
                  pred.min().item(),
                  pred.max().item(),
                  )
            diff = pred - labels
            print("RMSE:", torch.sqrt((diff ** 2).mean()).item())
            #print("RMSE:", loss.sqrt().item())
            pdb.set_trace()
        #pdb.set_trace()
        """
        # 4. Package like HF CausalLMOutputWithPast so train loop can use output.loss
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


    """
    def forward(self, x, labels=None, **unused):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        assert C == self.C and H == self.H and W == self.W

        # flatten spatial dims into feature dim
        x = x.view(B, T, C * H * W)      # (B, T, C*H*W)
        x = self.proj(x)                # (B, T, d_model)

        # now feed as inputs_embeds
        return self.model(inputs_embeds=x, labels=labels)
    """
