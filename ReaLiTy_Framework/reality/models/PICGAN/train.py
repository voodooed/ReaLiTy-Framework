import torch
import config
from tqdm import tqdm

# S - Simulated Domain
# R - Real Domain


def train_fn(
    disc_S, disc_R, gen_R, gen_S, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler,
    phy_transform=None, extra_generator_loss=None, real_label=1.0
):
    """
    phy_transform         : optional callable applied to fake_real before the
                            physics comparison only. The physics term compares the
                            generated intensity's position within the target
                            distribution against phy's position within the physics
                            distribution, so both operands must be in z-space. When
                            gen_R emits [0, 1] data units instead of z-scores, this
                            puts it back into z-space for that one comparison and
                            leaves the term's meaning unchanged. None (default)
                            reproduces the original behaviour exactly.
    extra_generator_loss  : optional callable (fake_real, real, sim, phy) -> scalar
                            tensor, added to the generator objective. Used for the
                            distributional (Wasserstein) term; None by default.
    real_label            : target the discriminators are trained towards on real
                            samples. 1.0 (default) is the original behaviour; a
                            value below 1.0 is one-sided label smoothing, which
                            weakens an over-confident discriminator. It is applied
                            to the discriminator step only -- the generator still
                            aims at 1.0 -- so no loss term is added or reweighted.
    """
    R_reals = 0
    R_fakes = 0
    # Per-term accumulators. These only observe the losses that are already
    # computed below; no term is added, removed or reweighted.
    totals = {"D_loss": 0.0, "G_adversarial": 0.0, "cycle_real": 0.0,
              "cycle_sim": 0.0, "physics": 0.0, "extra": 0.0, "G_loss": 0.0}
    n_batches = 0
    loop = tqdm(loader, leave=True)

    for idx, (sim, real, phy) in enumerate(loop):
        sim = sim.to(config.DEVICE) # 3 channels - Depth, IA, Reflectance
        real = real.to(config.DEVICE) # 1 channel - Real intensity
        phy = phy.to(config.DEVICE) # 1 channel - Physics-based intensity

        # Train Discriminators R and S
        with torch.cuda.amp.autocast():  #For float16 training


            fake_real = gen_R(sim)
            D_R_real = disc_R(real)
            D_R_fake = disc_R(fake_real.detach())
            R_reals += D_R_real.mean().item()
            R_fakes += D_R_fake.mean().item()
            D_R_real_loss = mse(D_R_real, torch.full_like(D_R_real, real_label))
            D_R_fake_loss = mse(D_R_fake, torch.zeros_like(D_R_fake))
            D_R_loss = D_R_real_loss + D_R_fake_loss

            fake_sim = gen_S(real)
            D_S_real = disc_S(sim)
            D_S_fake = disc_S(fake_sim.detach())
            D_S_real_loss = mse(D_S_real, torch.full_like(D_S_real, real_label))
            D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake))
            D_S_loss = D_S_real_loss + D_S_fake_loss

            # put it together
            D_loss = (D_R_loss + D_S_loss) / 2


        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators R and S
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_S_fake = disc_S(fake_sim)
            D_R_fake = disc_R(fake_real)
            loss_G_S = mse(D_S_fake, torch.ones_like(D_S_fake))
            loss_G_R = mse(D_R_fake, torch.ones_like(D_R_fake))

            # cycle loss
            cycle_real = gen_R(fake_sim)
            cycle_sim = gen_S(fake_real)
            cycle_real_loss = l1(real, cycle_real)
            cycle_sim_loss = l1(sim, cycle_sim)

            """
            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_real = gen_R(real)
            identity_sim = gen_S(sim)
            identity_real_loss = l1(real, identity_real)
            identity_sim_loss = l1(sim, identity_sim)
            """
            #Physics Loss
            #Loss between generated intensity(real domain) with physics-based intensity.
            #Both operands must sit in the same (z-scored) space; phy_transform puts
            #fake_real there when the generator emits data units instead.
            phy_loss = l1(fake_real if phy_transform is None else phy_transform(fake_real),
                          phy)

            # add all together
            G_loss = (
                loss_G_R
                + loss_G_S
                + cycle_real_loss * config.LAMBDA_CYCLE
                + cycle_sim_loss * config.LAMBDA_CYCLE
               #+ identity_real_loss * config.LAMBDA_IDENTITY
               #+ identity_sim_loss * config.LAMBDA_IDENTITY
                + phy_loss * config.LAMBDA_Physics
            )

            extra_loss = None
            if extra_generator_loss is not None:
                extra_loss = extra_generator_loss(fake_real, real, sim, phy)
                G_loss = G_loss + extra_loss

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        n_batches += 1
        totals["D_loss"]        += D_loss.item()
        totals["G_adversarial"] += (loss_G_R + loss_G_S).item()
        totals["cycle_real"]    += cycle_real_loss.item()
        totals["cycle_sim"]     += cycle_sim_loss.item()
        totals["physics"]       += phy_loss.item()
        totals["extra"]         += 0.0 if extra_loss is None else extra_loss.item()
        totals["G_loss"]        += G_loss.item()

        loop.set_postfix(R_real=R_reals / (idx + 1), R_fake=R_fakes / (idx + 1))

    # Mean of each term over the epoch, so training dynamics are diagnosable.
    return {name: value / max(n_batches, 1) for name, value in totals.items()}