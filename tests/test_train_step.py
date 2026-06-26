import torch

from networks.models.glio_ode.glio_ode import GlioODE
from networks.models.glio_ode.diffusion import GlioDiffusion


def test_one_optimizer_step_changes_parameters():
    model = GlioODE(
        crop_size=(8, 16, 16),
        in_channels=2,
        num_classes=2,
        patch_size=(2, 4, 4),
        d_model=12,
        num_heads=2,
        mlp_ratio=2.0,
        ode_steps=2,
        decoder_channels=(12, 6, 6, 6),
        diffusion_timesteps=10,
    )
    diff = GlioDiffusion(model, image_channels=2, mask_channels=2, timesteps=10)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    pre = {n: p.detach().clone() for n, p in model.named_parameters()}

    z0 = torch.randn(2, 4, 8, 16, 16)
    optim.zero_grad()
    loss = diff(z0)
    loss.backward()
    optim.step()

    changed = 0
    for n, p in model.named_parameters():
        if p.requires_grad and not torch.equal(pre[n], p.detach()):
            changed += 1
    assert changed > 0, "optimizer.step() did not move any parameter"
